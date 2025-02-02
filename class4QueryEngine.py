import os
import sys
import heapq
import numpy as np
import networkx as nx
import spacy
from typing import List, Tuple, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain.document_loaders import PyPDFLoader

os.environ['OPENAI_API_KEY'] = "Your OpenAi key""

from class1VietnameseEmbeddings import *
from class2DocumentProcessor import *
from class3KnowledgeGraph import *



class AnswerCheck(BaseModel):
    is_complete: bool = Field(description = "Whether the current context provides a complete answer to the query")
    answer: str = Field(description = "The current answer based on the context, if any")

class QueryEngine:
    def __init__(self, vector_store, knowledge_graph, llm):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.max_context_length = 4000
        self.answer_check_chain = self._create_answer_check_chain()

    def _create_answer_check_chain(self):
        """
        Tạo một chuỗi để kiểm tra xem ngữ cảnh có cung cấp câu trả lời đầy đủ cho truy vấn hay không.

        Tham số:
        - None

        Trả về:
        - Chain: Một chuỗi để kiểm tra xem ngữ cảnh có cung cấp câu trả lời đầy đủ không.
        """
        answer_check_prompt = PromptTemplate(
            input_variables = ["query", "context"],
            template = "Given the query: '{query}'\n\nAnd the current context:\n{context}\n\nDoes this context provide a complete answer to the query? If yes, provide the answer. If no, state that the answer is incomplete.\n\nIs complete answer (Yes/No):\nAnswer (if complete):"
        )
        return answer_check_prompt | self.llm.with_structured_output(AnswerCheck)

    def _check_answer(self, query: str, context: str) -> Tuple[bool, str]:
        """
        Kiểm tra xem ngữ cảnh hiện tại có cung cấp câu trả lời đầy đủ cho truy vấn không.

        Tham số:
        - query (str): Truy vấn cần trả lời.
        - context (str): Ngữ cảnh hiện tại.

        Returns:
        - tuple: Một tuple chứa:
          - is_complete (bool): Liệu ngữ cảnh có cung cấp câu trả lời đầy đủ không.
          - answer (str): Câu trả lời dựa trên ngữ cảnh, nếu đầy đủ.
        """
        response = self.answer_check_chain.invoke({"query": query, "context": context})
        return response.is_complete, response.answer

    def _expand_context(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:
        """
        Mở rộng ngữ cảnh bằng cách duyệt qua knowledge graph theo cách tiếp cận giống Dijkstra. 
        Phương pháp này triển khai một phiên bản sửa đổi của thuật toán Dijkstra để khám phá knowledge graph,
        ưu tiên thông tin có liên quan nhất. Thuật toán hoạt động như sau:
        1. Khởi tạo:
           - Bắt đầu với các nodes tương ứng với các documents có liên quan nhất.
           - Sử dụng hàng đợi ưu tiên (priority queue) để quản lý thứ tự duyệt, trong đó độ ưu tiên dựa trên connection strength.
           - Duy trì một dictionary các "khoảng cách" tốt nhất (đảo ngược của connection strength) đến mỗi nodes.
        2. Traverse:
           - Luôn khám phá nodes có độ ưu tiên cao nhất (kết nối mạnh nhất) tiếp theo.
           - Với mỗi node, kiểm tra xem chúng ta đã tìm được câu trả lời đầy đủ chưa.
           - Khám phá các node neighbor, cập nhật độ ưu tiên của chúng nếu phát hiện kết nối mạnh hơn.
        3. Xử lý các khái niệm:
           - Theo dõi các khái niệm đã duyệt qua để hướng quá trình đi khám phá các thông tin mới có liên quan.
           - Mở rộng sang các neighbor chỉ khi chúng có các khái niệm mới.
        4. Kết thúc:
           - Dừng lại nếu đã tìm được câu trả lời đầy đủ.
           - Tiếp tục cho đến khi hàng đợi ưu tiên rỗng (tất cả các node có thể tiếp cận đã được khám phá).

        Tham số:
        - query (str): Truy vấn cần trả lời.
        - relevant_docs (List[Document]): Danh sách các tài liệu có liên quan để bắt đầu quá trình duyệt.

        Returns:
        - tuple: Một tuple chứa:
          - expanded_context (str): Ngữ cảnh tích lũy từ các node đã duyệt.
          - traversal_path (List[int]): Dãy các chỉ số node đã duyệt qua.
          - filtered_content (Dict[int, str]): Bản đồ các chỉ số node với nội dung của chúng.
          - final_answer (str): Câu trả lời cuối cùng nếu có.
        """
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""
        priority_queue = []
        distances = {}

        print("\nTraversing the knowledge graph:")

        # Khởi tạo hàng đợi ưu tiên với các node gần nhất với các tài liệu có liên quan
        for doc in relevant_docs:
            # Tìm node gần tương tự nhất trong knowledge graph cho mỗi tài liệu có liên quan
            closest_nodes = self.vector_store.similarity_search_with_score(doc.page_content, k=1)
            closest_node_content, similarity_score = closest_nodes[0]

            # Lấy node tương ứng in knowledge graph
            closest_node = next(n for n in self.knowledge_graph.graph.nodes if
                                self.knowledge_graph.graph.nodes[n]['content'] == closest_node_content.page_content)

            # Khởi tạo độ ưu tiên (đảo ngược của điểm tương đồng để có hành vi min-heap)
            priority = 1 / similarity_score
            heapq.heappush(priority_queue, (priority, closest_node))
            distances[closest_node] = priority

        step = 0
        while priority_queue:
            # Lấy node có độ ưu tiên cao nhất (giá trị khoảng cách thấp nhất)
            current_priority, current_node = heapq.heappop(priority_queue)

            # Bỏ qua đã có đường đi tốt hơn đến node này
            if current_priority > distances.get(current_node, float('inf')):
                continue

            if current_node not in traversal_path:
                step += 1
                traversal_path.append(current_node)
                node_content = self.knowledge_graph.graph.nodes[current_node]['content']
                node_concepts = self.knowledge_graph.graph.nodes[current_node]['concepts']

                # Thêm nội dung node vào ngữ cảnh tích lũy
                filtered_content[current_node] = node_content
                expanded_context += "\n" + node_content if expanded_context else node_content

                # Ghi lại bước hiện tại để debug và visuallize
                print(f"\nStep {step} - Node {current_node}:")
                print(f"Content: {node_content[:100]}...")
                print(f"Concepts: {', '.join(node_concepts)}")
                print("-" * 50)

                # Kiểm tra xem có câu trả lời đầy đủ với ngữ cảnh hiện tại không  
                is_complete, answer = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    break

                # Xử lý các khái niệm của node hiện tại
                node_concepts_set = set(self.knowledge_graph._lemmatize_concept(c) for c in node_concepts)
                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)

                    # Khám phá các neighbors
                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]
                        edge_weight = edge_data['weight']

                        # Tính khoảng cách (độ ưu tiên) mới đến neighbor
                        # Sử dụng 1 / edge_weight vì các trọng số cao hơn có nghĩa là kết nối mạnh hơn
                        distance = current_priority + (1 / edge_weight)

                        # Nếu tìm thấy một kết nối mạnh hơn đến neighbor, cập nhật khoảng cách của nó
                        if distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))

                            # Xử lý node neighbor nếu nó chưa có trong traversal path
                            if neighbor not in traversal_path:
                                step += 1
                                traversal_path.append(neighbor)
                                neighbor_content = self.knowledge_graph.graph.nodes[neighbor]['content']
                                neighbor_concepts = self.knowledge_graph.graph.nodes[neighbor]['concepts']

                                filtered_content[neighbor] = neighbor_content
                                expanded_context += "\n" + neighbor_content if expanded_context else neighbor_content

                                # Ghi lại thông tin neighbor node
                                print(f"\nStep {step} - Node {neighbor} (neighbor of {current_node}):")
                                print(f"Content: {neighbor_content[:100]}...")
                                print(f"Concepts: {', '.join(neighbor_concepts)}")
                                print("-" * 50)

                                # Kiểm tra xem chúng ta có câu trả lời đầy đủ sau khi thêm nội dung của neighbor không   
                                is_complete, answer = self._check_answer(query, expanded_context)
                                if is_complete:
                                    final_answer = answer
                                    break

                                # Xử lý các khái niệm của neighbor
                                neighbor_concepts_set = set(
                                    self.knowledge_graph._lemmatize_concept(c) for c in neighbor_concepts)
                                if not neighbor_concepts_set.issubset(visited_concepts):
                                    visited_concepts.update(neighbor_concepts_set)

                # Nếu tìm được câu trả lời cuối cùng, thoát khỏi vòng lặp chính
                if final_answer:
                    break

        # Nếu chưa tìm được câu trả lời đầy đủ, tạo câu trả lời bằng LLM
        if not final_answer:
            print("\nGenerating final answer...")
            response_prompt = PromptTemplate(
                input_variables = ["query", "context"],
                template = "Based on the following context, please answer the query in vietnamese.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
            )
            response_chain = response_prompt | self.llm
            input_data = {"query": query, "context": expanded_context}
            final_answer = response_chain.invoke(input_data)

        return expanded_context, traversal_path, filtered_content, final_answer

    def query(self, query: str, return_full_info: bool = True) -> Union[str, Tuple[str, List[int], Dict[int, str]]]:
        """
        Xử lý một query bằng cách truy xuất các tài liệu liên quan, mở rộng ngữ cảnh và tạo ra câu trả lời cuối cùng.

        Tham số:
        - query (str): query cần trả lời.
        - return_full_info (bool): Nếu True, trả về tất cả thông tin; nếu False, chỉ trả về câu trả lời cuối cùng.

        Trả về:
        - final_answer (str) nếu return_full_info là False, nếu True trả về một tuple chứa:
            - final_answer (str): Câu trả lời cuối cùng cho truy vấn với tổng số token.
            - traversal_path (list): Đường đi của các node trong knowledge graph.
            - filtered_content (dict): Nội dung đã lọc của các node.
        """
        with get_openai_callback() as cb:
            print(f"\nProcessing query: {query}")
            relevant_docs = self._retrieve_relevant_documents(query)
            expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(query, relevant_docs)

            if not final_answer:
                print("\nGenerating final answer...")
                response_prompt = PromptTemplate(
                    input_variables = ["query", "context"],
                    template = "Based on the following context, please answer the query in vietnamese.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
                )

                response_chain = response_prompt | self.llm
                input_data = {"query": query, "context": expanded_context}
                response = response_chain.invoke(input_data)
                final_answer = response
            else:
                print("\nComplete answer found during traversal.")
            
            total_tokens = cb.total_tokens
            print(f"\nFinal Answer: {final_answer}")
            print(f"\nTotal Tokens: {total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            
            # Sửa câu trả lời cuối cùng để bao gồm tổng số token
            final_answer_with_tokens = f"{final_answer} total_tokens={total_tokens}"

        if return_full_info:
            # Trả về thông tin đầy đủ với câu trả lời cuối cùng đã sửa
            return final_answer_with_tokens, traversal_path, filtered_content 
        else:
            # Chỉ trả về câu trả lời cuối cùng với tổng số token
            return final_answer_with_tokens 


    def _retrieve_relevant_documents(self, query: str):
        """
        Truy xuất các tài liệu liên quan dựa trên truy vấn sử dụng vector store.

        Tham số:
        - query (str): Truy vấn cần được trả lời.

        Trả về:
        - list: Một list các tài liệu liên quan.
        """
        print("\nRetrieving relevant documents...")
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        return compression_retriever.invoke(query)