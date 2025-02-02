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



class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=200):
        """
        Khởi tạo DocumentProcessor với text splitter và PhoBERT embeddings.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n", " ", ".", ",", ";", ":", "?", "!"]
        )
        self.embeddings = VietnameseEmbeddings()

    def process_documents(self, documents):
        """
        Xử lý list các documents bằng cách chia chúng thành các chunk nhỏ hơn và tạo một vector store.
        """
        splits = self.text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(splits, self.embeddings) 
        return splits, vector_store

    def create_embeddings_batch(self, texts, batch_size=32):
        """
        Tạo embeddings cho list các texts theo từng batches.
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def compute_similarity_matrix(self, embeddings):
        """
        Tính cosine similarity matrix cho một bộ embeddings đã cho.
        """
        return cosine_similarity(embeddings)