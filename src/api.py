from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.core import PromptTemplate
import pandas as pd
from dotenv import load_dotenv
import uvicorn
import faiss
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

prompt_template = PromptTemplate(
    """You are a bartender. Answer directly and concisely.

    Context: {context_str}
    Question: {query_str}

    Rules:
    - For ingredient questions, ONLY list: "1. [Name]"
    - For recipe questions, provide full details
    - Never repeat the list
    - Never use phrases like "Based on" or "Therefore"
    - Never explain your answer
    - Just give the information requested
    """
)

llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    token=os.getenv('HUGGINGFACE_API_KEY'),
    query_wrapper_prompt=prompt_template
)

columns = {
    'id': int,
    'name': str,
    'alcoholic': str,
    'category': str,
    'glassType': str,
    'instructions': str,
    'drinkThumbnail': str,
    'text': str,
}

converters = {
    'ingredients': lambda x: x.strip("[]").split(", "),
    'ingredientMeasures': lambda x: x.strip("[]").split(", "),
}

class Query(BaseModel):
    text: str

app = FastAPI(title="Cocktail Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

FAISS_INDEX_PATH = "storage/faiss_index.bin"
DOCS_INDEX_PATH = "storage/docs"

def initialize_index():
    os.makedirs("storage", exist_ok=True)
    
    try:
        
        print("Creating new index...")
        faiss_index = faiss.IndexFlatL2(384)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
       
        try:
            df = pd.read_csv("./dataset/final_cocktails.csv", dtype=columns, converters=converters)
        except FileNotFoundError:
            df = pd.read_csv("../dataset/final_cocktails.csv", dtype=columns, converters=converters)
        
        documents = [
            Document(
                text=f"""Cocktail Name: {row['name']}
                Type: {row['alcoholic']}
                Category: {row['category']}
                Glass: {row['glassType']}
                
                Ingredients:
                {', '.join(f'{ing}: {meas}' for ing, meas in zip(row['ingredients'], row['ingredientMeasures']))}
                
                Instructions:
                {row['instructions']}
                
                Additional Information:
                {row['text']}""",
                metadata={
                    "name": row['name'],
                    "alcoholic": row['alcoholic'],
                    "category": row['category'],
                    "thumbnail": row['drinkThumbnail']
                }
            ) for _, row in df.iterrows()
        ]
        
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            vector_store=vector_store,
            show_progress=True
        )
        
        return index
        
    except Exception as e:
        print(f"Error initializing index: {str(e)}")
        raise

index = initialize_index()

@app.post("/chat")
async def chat(query: Query):
    try:
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=10,
            response_mode="compact",
            structured_answer_filtering=True,
            node_postprocessors=[],
        )
        
        response = query_engine.query(query.text)
        
        if not str(response).strip():
            return {
                "response": "I couldn't generate a response. Please try a different question.",
                "status": "no_response"
            }
            
        return {
            "response": str(response),
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return {
            "response": "Sorry, I encountered an error processing your request.",
            "error": str(e)
        }

@app.options("/chat")
async def options_chat():
    return JSONResponse(
        status_code=200,
        content={"message": "OK"}
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="localhost", port=8000)


