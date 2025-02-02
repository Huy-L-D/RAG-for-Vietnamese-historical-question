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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.environ['OPENAI_API_KEY'] = "Your OpenAi key"

from class1VietnameseEmbeddings import *
from class2DocumentProcessor import *
from class3KnowledgeGraph import *
from class4QueryEngine import *



class Visualizer:
    @staticmethod
    def visualize_traversal(graph, traversal_path):
        """
        Hiển thị traversal path trên knowledge graph

        Tham số:
        - graph (networkx.Graph): Knowledge graph với nodes và edges
        - traversal_path (list of int): Danh sách chỉ mục của các node đại diện cho traversal path.

        Trả về:
        - None
        """
        # Tạo một đồ thị mới chỉ chứa các node và các cạnh từ traversal path
        traversal_graph = nx.DiGraph()

        # Thêm các nodes và edges từ traversal path
        for i in range(len(traversal_path) - 1):
            start_node = traversal_path[i]
            end_node = traversal_path[i + 1]
            
            # Thêm nodes
            if start_node in graph.nodes:
                traversal_graph.add_node(start_node, **graph.nodes[start_node])
            if end_node in graph.nodes:
                traversal_graph.add_node(end_node, **graph.nodes[end_node])

            # Thêm edges
            if graph.has_edge(start_node, end_node):
                traversal_graph.add_edge(start_node, end_node, **graph.get_edge_data(start_node, end_node))

        fig, ax = plt.subplots(figsize=(16, 12))

        # Tạo vị trí cho tất cả các node trong traversal graph
        pos = nx.spring_layout(traversal_graph, k=1, iterations=50)

        # Vẽ cạnh
        nx.draw_networkx_edges(traversal_graph, pos,
                               edge_color='red',
                               width=2,
                               arrowstyle="->",
                               arrowsize=15,
                               connectionstyle="arc3,rad=0.3",
                               ax=ax)

        # Vẽ node
        nx.draw_networkx_nodes(traversal_graph, pos,
                               node_color='lightblue',
                               node_size=3000,
                               ax=ax)

        # Nhãn cho các node
        labels = {node: f"{i + 1}. {graph.nodes[node].get('concepts', [''])[0]}"
                  for i, node in enumerate(traversal_path) if node in traversal_graph.nodes}
        nx.draw_networkx_labels(traversal_graph, pos, labels, font_size=10, font_weight="bold", ax=ax)

        # Highlight node đầu và cuối
        start_node = traversal_path[0]
        end_node = traversal_path[-1]
        if start_node in traversal_graph.nodes:
            nx.draw_networkx_nodes(traversal_graph, pos,
                                   nodelist=[start_node],
                                   node_color='lightgreen',
                                   node_size=3000,
                                   ax=ax)
        if end_node in traversal_graph.nodes:
            nx.draw_networkx_nodes(traversal_graph, pos,
                                   nodelist=[end_node],
                                   node_color='lightcoral',
                                   node_size=3000,
                                   ax=ax)

        ax.set_title("Graph Traversal Flow")
        ax.axis('off')

        plt.tight_layout()
        plt.show()