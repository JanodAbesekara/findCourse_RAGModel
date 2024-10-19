
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
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust to match your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers (Authorization, etc.)
)


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
# def generate_rag_prompt(query, context):
#     escaped = context.replace("'", " ").replace('"', ' ').replace('\n', ' ')
#     prompt = (f"""
#     Based on the user's query: "{query}", please find the most relevant courses and details from the information below.

#     User's query: {query}

#     Extracted course details:

#     {escaped}

#     From the list of course details above, provide a list of the top recommended courses that match the user's query. 
#     Highlight the course title, faculty, semester fee, total fee, email, and contact number for each recommended course.
#     If You haven't found any course, please provide a message to the user indicating that no relevant courses were found based on the query.
#     If the context is not sufficient to generate a response, please provide a message to the user indicating that more information is needed to generate a response.
#     """)
#     return prompt


def generate_rag_prompt(query, context):
    # Escape problematic characters and preserve formatting
    escaped_context = context.replace("'", " ").replace('"', ' ').replace('\n', ' ')
    
    # Check if the query is asking about Software Engineering (SE)
    if query.lower() in ['se', 'software engineering']:
        prompt = f"""
        The user is asking about "Software Engineering" degrees (or SE). Please assist by finding the most relevant courses in Software Engineering from the provided context.

        User's Query: "{query}"

        Context (Course Details):
        {escaped_context}

        Task:
        From the provided context, identify and recommend the top Software Engineering courses that match the user's query. For each recommended course, include the following information:
          - Course Title
          - Faculty
          - Semester Fee
          - Total Fee
          - Email
          - Contact Number

        If no relevant Software Engineering courses are found based on the query, kindly inform the user by stating: "No relevant Software Engineering courses were found."
        """
    # Check if the query is asking about Quantity Surveying (QS)
    elif query.lower() in ['qs', 'quantity surveying']:
        prompt = f"""
        The user is asking about "Quantity Surveying" degrees (or QS). Please assist by finding the most relevant courses in Quantity Surveying from the provided context.

        User's Query: "{query}"

        Context (Course Details):
        {escaped_context}

        Task:
        From the provided context, identify and recommend the top Quantity Surveying courses that match the user's query. For each recommended course, include the following information:
          - Course Title
          - Faculty
          - Semester Fee
          - Total Fee
          - Email
          - Contact Number

        If no relevant Quantity Surveying courses are found based on the query, kindly inform the user by stating: "No relevant Quantity Surveying courses were found."
        """
   # Check if the user is only asking for a list of faculties
    if 'faculty' in query.lower() and ('list' in query.lower() or 'faculties' in query.lower() or 'all' in query.lower() or 'computing' in query.lower()):
       prompt = f"""
    The user is asking for a list of faculties at the university, specifically including the Faculty of Computing. Here are some example queries that would trigger this response:
    
    - "What are the faculties at [University Name]?"
    - "Give all faculties at [University Name]."
    - "Provide a list of faculties in [University Name]."
    - "Show me the faculties at [University Name]."
    - "Give all faculties including Faculty of Computing at [University Name]."
    - "What are the faculties related to computing at [University Name]?"

    User's Query: "{query}"

    Context (Course Details):   
    {escaped_context}

    Task:
    From the provided context, extract and list only the faculty names, including the **Faculty of Computing**. The list should be in a simple bullet point format, such as:
      - Faculty Name

    Do not include any course titles, fees, or other details.

    If no faculties, or the Faculty of Computing, are found in the context, kindly inform the user by stating: "No faculties or the Faculty of Computing were found in the provided context."
    """
    # Check if the user is asking for faculty details with courses
    elif 'faculty' in query.lower() and 'details' in query.lower():
        prompt = f"""
        The user is asking for faculty details along with the associated courses. Please extract the faculty names and related course details from the provided context.

        User's Query: "{query}"

        Context (Course Details):
        {escaped_context}

        Task:
        From the provided context, extract and list the faculty names along with their associated courses. For each faculty, include the following format:
          - Faculty Name
          - Course Title
          - Semester Fee
          - Total Fee
          - Email
          - Contact Number

        If no faculties or courses are found, kindly inform the user by stating: "No faculty or course details were found in the provided context."
        """
    else:
        prompt = f"""
        Based on the user's query: "{query}", please find the most relevant courses and details from the information below.

        User's query: {query}

        Context (Course Details):
        {escaped_context}

        Task:
        From the provided context, identify and recommend the top courses that match the user's query. For each recommended course, include the following information:
          - Course Title
          - Faculty
          - Semester Fee
          - Total Fee
          - Email
          - Contact Number

        If no relevant courses are found based on the query, kindly inform the user by stating: "No relevant courses were found."

        If the context is insufficient or unclear, please return the message: "More information is needed to generate relevant course recommendations."
        """
    
    return prompt.strip()  # Strip leading/trailing whitespace for cleaner output





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
