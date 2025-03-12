from fastapi import FastAPI
import chromadb
from sentence_transformers import SentenceTransformer
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Initialize FastAPI
app = FastAPI()

# Allow requests from any frontend (change this for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains for security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="medical_chatbot")

# Function to retrieve answers
def get_answer(user_query):
    query_embedding = model.encode(user_query, convert_to_tensor=True).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=1
    )

    if results["documents"]:
        return results["metadatas"][0][0]["answer"]
    else:
        return "I'm sorry, I don't have an answer for that."

# API endpoint for chatbot
@app.post("/chatbot")
async def chatbot(message: dict):
    user_message = message.get("message", "")
    return {"response": f"Echo: {user_message}"}

@app.get("/")
def read_root():
    return {"message": "Hello, Healthcare Chatbot is running!"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)