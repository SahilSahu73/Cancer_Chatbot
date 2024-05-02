from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
client = QdrantClient(
    url= 'https://428bf2a1-62eb-495a-8a2f-90919827ca0b.us-east4-0.gcp.cloud.qdrant.io:6333',
    api_key='Z5w9lFdKr30tZk6E6tbScx6nXLFBAGf4uJsWRlPoZuvSlucdaMmfXw'
)

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

loader = DirectoryLoader('data/', glob='**/*.pdf', show_progress=True, loader_cls=UnstructuredFileLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

list_Documents = text_splitter.split_documents(documents)

# texts = [doc.page_content for doc in list_Documents]

# url = 'http://localhost:6333'

qdrant = Qdrant.from_documents(
    list_Documents,
    embeddings,
    url= 'https://428bf2a1-62eb-495a-8a2f-90919827ca0b.us-east4-0.gcp.cloud.qdrant.io:6333',
    api_key='Z5w9lFdKr30tZk6E6tbScx6nXLFBAGf4uJsWRlPoZuvSlucdaMmfXw',
    prefer_grpc=False,
    collection_name='CancerData'
)

print("Cancer Data is Created.")