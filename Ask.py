import generate_embeding
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import signal
import sys
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv() 

google_api_key = os.getenv("GOOGLE_API_KEY")

# Function to handle exit signal
def signal_handler(sig, frame):
    print('\nWelcome To Findbest Course')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


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
                 QUESTION:'{query}'
                 CONTEXT:'{context}'              
            ANSWER:     
    """).format(query=query, context=context)
    return prompt
    

    
    
    
    
    
# Function to get relevant context from the database
def 
get_relevent_context_from_db(query):
    context = ""

    # embeddings_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    embeddings_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Use the correct parameter name: embedding_function instead of embedding
    vector_db = Chroma(persist_directory="./findcourseDB", embedding_function=embeddings_function)
    
    # Perform similarity search
    search_results = vector_db.similarity_search(query, k=10)
    for result in search_results:
        context += result.page_content + "\n"
    return context

def generate_answer(prompt):
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    answer = model.generate_content(prompt)
    return answer.text

#Gemini_1.5_Flash

welcome_text = generate_answer("Welcome To Findbest Course")
print(welcome_text)

# Main loop to handle user queries
while True:
    print("---------------------------------------------------")
    print("What You Want?")
    query = input("query: ")
    context = get_relevent_context_from_db(query)
    prompt = generate_rag_prompt(query = query, context = context)
    answer = generate_answer(prompt = prompt)
    print(answer)
    
