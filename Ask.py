
from fastapi import FastAPI, Request
import generate_embeding
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import signal
import sys
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Google API Key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=google_api_key)

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print('\nShutting down Findbest Course API')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Model for user query input
class QueryRequest(BaseModel):
    query: str

# Generate a RAG prompt based on user query and context
def generate_rag_prompt(query, context):
    escaped = context.replace("'", " ").replace('"', ' ').replace('\n', ' ')
    prompt = (f"""
    Based on the user's query: "{query}", please find the most relevant courses and details from the information below.

    User's query: {query}

    Extracted course details:

    {escaped}

    From the list of course details above, provide a list of the top recommended courses that match the user's query. 
    Highlight the course title, faculty, semester fee, total fee, email, and contact number for each recommended course.
    If You haven't found any course, please provide a message to the user indicating that no relevant courses were found based on the query.
    If the context is not sufficient to generate a response, please provide a message to the user indicating that more information is needed to generate a response.
    """)
    return prompt

# Function to get relevant context from the database using vector search
def get_relevant_context_from_db(query):
    context = ""
    embeddings_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./findcourseDB", embedding_function=embeddings_function)
    
    # Perform similarity search
    search_results = vector_db.similarity_search(query, k=10)
    for result in search_results:
        context += result.page_content + "\n"
    return context

# Function to generate answer using Google's Gemini model
def generate_answer(prompt):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# API route to handle user queries
@app.post("/query")
async def handle_query(request: QueryRequest):
    query = request.query
    context = get_relevant_context_from_db(query)
    prompt = generate_rag_prompt(query=query, context=context)
    answer = generate_answer(prompt=prompt)
    return {"query": query, "answer": answer}

# Root API route
@app.get("/")
async def root():
    return {"message": "Welcome to the Findbest Course API"}
