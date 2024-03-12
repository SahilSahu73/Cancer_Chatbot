from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
client = QdrantClient(
    url=os.getenv("Qdrant_HOST"), 
    api_key=os.getenv("Qdrant_API_KEY")
)

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

loader = DirectoryLoader('data/', glob='**/*.pdf', show_progress=True, loader_cls=UnstructuredFileLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

texts = text_splitter.split_documents(documents)

# url = 'http://localhost:6333'

qdrant = Qdrant.from_texts(
    texts,
    embeddings,
    url=os.getenv("Qdrant_HOST"),
    api_key=os.getenv("Qdrant_API_KEY"),
    prefer_grpc=False,
    collection_name=os.getenv('Qdrant_Collection_name')
)

print("Cancer Data is Created.")