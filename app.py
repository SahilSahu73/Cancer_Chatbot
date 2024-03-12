from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp 
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dotenv import load_dotenv
import json
import os

load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory='templates')

app.mount("/static", StaticFiles(directory='static'), name='static')

local_llm = r"C:\Users\Sahil Sahu\Desktop\Cancer_ChatBot\models\BioMistral-7B.Q4_K_M.gguf"

llm = LlamaCpp(
    model_path = local_llm,
    temperature = 0.2,
    max_tokens = 2048,
    top_p = 2
)

print("LLM Initialized......")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say don't know.
Context: {context}
Question: {question}
Only return helpful answer in detail.
Helpful answer:
"""

# # Initialize translation model and tokenizer
# translation_model_name = "ai4bharat/IndicBARTSS"
# translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)
# translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)

# def translate_to_indic_language(text, target_language):
#     inputs = translation_tokenizer(text, return_tensors="pt")
#     translated_tokens = translation_model.generate(**inputs)
#     translated_text = translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
#     return translated_text


embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = os.getenv("Qdrant_HOST")

client = QdrantClient(
    url=url,
    api_key=os.getenv("Qdrant_API_KEY"),
    prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name='CancerDataPart1')

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k":1})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
    response = qa(query)
    print(response)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['_collection_name']

    # if target_language:
    #     translated_answer = translate_to_indic_language(answer, target_language)
    # else:
    #     translated_answer = answer  # No translation if no language chosen

    response_data = jsonable_encoder(json.dumps({
        "answer": answer,
        # "translated_answer":translated_answer, 
        "source_document": source_document, 
        "doc": doc
    }))
    
    res = Response(response_data)
    return res