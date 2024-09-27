import pdfplumber
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document  # Import Document schema
from generate_embeding import vector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv() 

# Extract Data from PDF: Separate topic and table data for each page
def extract_tables_from_pdf(pdf_path):
    tables_with_topics = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract topic and table data from each page
            topic = page.extract_text().split('\n')[0].strip()  # Assuming first line is the topic
            table = page.extract_table()
            if table:
                # Store table data along with its topic
                tables_with_topics.append((topic, table))
    return tables_with_topics

# Clean and process each table separately
def clean_and_convert_to_dataframe(topic, raw_table):
    # Convert to DataFrame and clean the data
    df = pd.DataFrame(raw_table[1:], columns=raw_table[0])  # Assuming first row as header
    df = df.applymap(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
    df.columns = ['Degree', 'Semester Fee', 'Total Programme Fee', 'Email', 'Contact No']
    return df

# Define the Dynamic Class
class DynamicClass:
    def __init__(self, topic, degree, semester_fee, total_fee, email, contact_no):
        self.topic = topic  # Faculty or topic
        self.degree = degree
        self.semester_fee = semester_fee
        self.total_fee = total_fee
        self.email = email
        self.contact_no = contact_no

    def __repr__(self):
        return (f"Faculty={self.topic}, Degree={self.degree}, Semester Fee={self.semester_fee}, "
                f"Total Fee={self.total_fee}, Email={self.email}, Contact No={self.contact_no}")

# Embed and vectorize the data
def embed_data(data, embeddings):
    return embeddings.embed_query(data)

# Path to the PDF file
pdf_path = 'Course.pdf'  # Change this to the path of your PDF file
tables_with_topics = extract_tables_from_pdf(pdf_path)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store all programs and their embeddings
all_programs = []
all_documents = []  # Store as a list of Document objects

for topic, raw_table in tables_with_topics:
    df_clean = clean_and_convert_to_dataframe(topic, raw_table)
    # Create DynamicClass objects for the current table
    for index, row in df_clean.iterrows():
        program = DynamicClass(
            topic,
            row['Degree'],
            row['Semester Fee'],
            row['Total Programme Fee'],
            row['Email'],
            row['Contact No']
        )
        # Add to the list
        all_programs.append(program)

        # Create a Document object with the program details
        doc_content = (f"Faculty: {program.topic}, Degree: {program.degree}, "
                       f"Semester Fee: {program.semester_fee}, Total Fee: {program.total_fee}, "
                       f"Email: {program.email}, Contact No: {program.contact_no}")
        doc = Document(page_content=doc_content, metadata={"topic": program.topic})
        all_documents.append(doc)
        

# Initialize Chroma vector store with the Document objects and embeddings
vectorstore = Chroma.from_documents(documents=all_documents, embedding=embeddings, persist_directory="findcourseDB")

# Assuming vectorstore is your Chroma vector store object
document_count = vectorstore._collection.count()

print(f"Total number of documents stored: {document_count}")

print("Vector store initialized and stored in 'vectorstore' directory.")



  
        


