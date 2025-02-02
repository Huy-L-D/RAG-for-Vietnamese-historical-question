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
from class4QueryEngine import *
from class5Visualizer import *



class GraphRAG:
    def __init__(self, documents, chunk_size=500, chunk_overlap=200, edges_threshold=0.8):
        """
        Khởi tạo hệ thống GraphRAG với các thành phần để xử lý tài liệu, xây dựng knowledge graph, truy vấn và hiển thị.

        Tham số:
        - documents (list of str): Danh sách các tài liệu cần được xử lý.
        """
        # Thiết lập mô hình ngôn ngữ và embeddings PhoBERT
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
        # Sử dụng VietnameseEmbeddings thay vì OpenAIEmbeddings
        self.embedding_model = VietnameseEmbeddings()  
        self.document_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 
        self.knowledge_graph = KnowledgeGraph(edges_threshold=edges_threshold)
        self.query_engine = None
        self.visualizer = Visualizer()
        self.process_documents(documents)

    def process_documents(self, documents):
        """
        Xử lý một list các document bằng cách chia chúng thành các đoạn, embedding chúng và xây dựng knowledge graph.
        """
        splits, vector_store = self.document_processor.process_documents(documents)
        # Pass VietnameseEmbeddings cho knowledge graph
        self.knowledge_graph.build_graph(splits, self.llm, self.embedding_model)
        self.query_engine = QueryEngine(vector_store, self.knowledge_graph, self.llm)

    def query(self, query: str):
        """
        Xử lý một truy vấn bằng cách truy xuất thông tin liên quan từ knowledge graph và hiển thị traversal path.

        Tham số:
        - query (str): Truy vấn cần được trả lời.

        Trả về:
        - str: Phản hồi cho truy vấn.
        """
        # Đảm bảo truy vấn trong QueryEngine đang sử dụng đúng embedding_model
        response, traversal_path, filtered_content= self.query_engine.query(query)

        if traversal_path:
            self.visualizer.visualize_traversal(self.knowledge_graph.graph, traversal_path)
        else:
            print("No traversal path to visualize.")

        return response