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



class Concepts(BaseModel):
    concepts_list: List[str] = Field(description="List of concepts")
class KnowledgeGraph:
    def __init__(self,edges_threshold=0.8):
        """
        Khởi tạo KnowledgeGraph với đồ thị, lemmatizer và mô hình NLP.

        Tham số:
        - graph: đồ thị networkx.
        - lemmatizer: WordNetLemmatizer.
        - concept_cache: dictionary để lưu cache các khái niệm đã trích xuất.
        - nlp: spaCy NLP model.
        - edges_threshold: Một giá trị float để đặt threshold khi thêm các edges dựa trên độ tương đồng.
        """
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}
        self.nlp = self._load_spacy_model()
        self.edges_threshold = edges_threshold

    def build_graph(self, splits, llm, embedding_model):
        """
        Tạo knowledge graph bằng cách thêm các nodes, tạo các embeddings, trích xuất các concepts, và thêm các edges.

        Tham số:
        - splits (list): list các document splits.
        - llm: large language model.
        - embedding_model: embedding model.

        Trả về:
        - None
        """
        self._add_nodes(splits)
        embeddings = self._create_embeddings(splits, embedding_model)
        self._extract_concepts(splits, llm)
        self._add_edges(embeddings)

    def _add_nodes(self, splits):
        """
        Thêm nodes vào graph từ các document splits.

        Tham số:
        - splits (list): list các document splits.

        Trả về:
        - None
        """
        for i, split in enumerate(splits):
            self.graph.add_node(i, content=split.page_content)

    def _create_embeddings(self, splits, embedding_model):
        """
        Tạo embeddings cho các document splits sử dụng embedding model.

        Tham số:
        - splits (list): list các document splits.
        - embedding_model: embedding model.

        Trả về:
        - numpy.ndarray: Một array embeddings cho các document splits.
        """
        texts = [split.page_content for split in splits]
        return embedding_model.embed_documents(texts)

    def _compute_similarities(self, embeddings):
        """
        Tính cosine similarity matrix cho các embeddings.

        Tham số:
        - embeddings (numpy.ndarray): Một array các embeddings.

        Trả về:
        - numpy.ndarray: cosine similarity matrix cho các embeddings.
        """
        return cosine_similarity(embeddings)

    def _load_spacy_model(self):
        """
        Load spaCy NLP model

        Tham số:
        - None

        Trả về:
        - spacy.Language: spaCy NLP model.
        """
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def _extract_concepts_and_entities(self, content, llm):
        """
        Trích xuất các concepts và named entities từ content sử dụng spaCy và 1 llm.

        Tham số:
        - content (str): Nội dung để trích xuất concepts và entities.
        - llm: large language model.

        Trả về:
        - list: Danh sách các concepts và entities đã trích xuất.
        """
        if content in self.concept_cache:
            return self.concept_cache[content]

        # Trích xuất named entities sử dụng spaCy
        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]

        # Trích xuất các concepts chung sử dụng LLM
        concept_extraction_prompt = PromptTemplate(
            input_variables = ["text"],
            template = "Extract key concepts (excluding named entities) from the following text:\n\n{text}\n\nKey concepts:"
        )
        concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts)
        general_concepts = concept_chain.invoke({"text": content}).concepts_list

        # Kết hợp các named entities và các concepts chung
        all_concepts = list(set(named_entities + general_concepts))

        self.concept_cache[content] = all_concepts
        return all_concepts

    def _extract_concepts(self, splits, llm):
        """
        Trích xuất concepts cho tất cả document splits sử dụng đa luồng (multi-threading).

        Tham số:
        - splits (list): list các document splits.
        - llm: large language model.

        Trả về:
        - None
        """
        with ThreadPoolExecutor() as executor:
            future_to_node = {executor.submit(self._extract_concepts_and_entities, split.page_content, llm): i
                              for i, split in enumerate(splits)}

            for future in tqdm(as_completed(future_to_node), total = len(splits),
                               desc = "Extracting concepts and entities"):
                node = future_to_node[future]
                concepts = future.result()
                self.graph.nodes[node]['concepts'] = concepts

    def _add_edges(self, embeddings):
        """
        Thêm các edges vào graph dựa trên similarity của embeddings và các concepts chung.

        Tham số:
        - embeddings (numpy.ndarray): Một array các embeddings cho các document splits.

        Trả về:
        - None
        """
        similarity_matrix = self._compute_similarities(embeddings)
        num_nodes = len(self.graph.nodes)

        for node1 in tqdm(range(num_nodes), desc = "Adding edges"):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]
                if similarity_score > self.edges_threshold:
                    shared_concepts = set(self.graph.nodes[node1]['concepts']) & set(
                                          self.graph.nodes[node2]['concepts'])
                    edge_weight = self._calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    self.graph.add_edge(node1, node2, weight = edge_weight,
                                        similarity = similarity_score,
                                        shared_concepts = list(shared_concepts))

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts,alpha=0.8, beta=0.2):
        """
        Tính toán weight của một cạnh dựa trên similarity score và các khái niệm chung.

        Tham số:
        - node1 (int): Node đầu tiên.
        - node2 (int): Node thứ hai.
        - similarity_score (float): The similarity score giữa các nodes.
        - shared_concepts (set): set các khái niệm chung giữa các nodes.
        - alpha (float, optional): Trọng số của similarity score.
        - beta (float, optional):Trọng số của các shared concepts.

        Trả về:
        - float: Trọng số tính toán được của cạnh.
        """
        max_possible_shared = min(len(self.graph.nodes[node1]['concepts']), len(self.graph.nodes[node2]['concepts']))
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        return alpha * similarity_score + beta * normalized_shared_concepts

    def _lemmatize_concept(self, concept):
        """
        Lemmatize một khái niệm đã cho.

        Tham số:
        - concept (str): Khái niệm cần lemmatize.

        Trả về:
        - str: Khái niệm đã được lemmatize.
        """
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])