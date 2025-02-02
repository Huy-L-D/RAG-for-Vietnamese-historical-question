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



class VietnameseEmbeddings:
    def __init__(self, model_name="dangvantuan/vietnamese-embedding"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """
        Tạo embeddings cho list các văn bản tiếng Việt đã được tokenize.

        Tham số: 
        - texts (list of str): list các tài liệu văn bản cần tạo embeddings.
        
        Trả về: 
        - list of numpy.ndarray: Danh sách các embeddings cho từng tài liệu.
        """
        tokenized_texts = [tokenize(text) for text in texts]
        embeddings = self.model.encode(tokenized_texts)
        return embeddings
    
    def embed_query(self, text):
        """Tạo embedding chỉ cho query text đầu tiên."""
        return self.model.encode([tokenize(text)])[0]

    def __call__(self, text):
        """Hàm khi gọi sẽ trả về embedding của text"""
        return self.embed_query(text)