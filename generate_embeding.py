# from langchain_community.vectorstores import Chroma 
# from langchain_community.embeddings import HuggingFaceEmbeddings 
# from langchain.text_splitter import RecursiveCharacterTextSplitter 
# from langchain_community.document_loaders import PyPDFLoader 

# loders = [PyPDFLoader('./Course.pdf')]

# docs = []


# for file in loders:
#     docs.extend(file.load())


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=100)
# docs = text_splitter.split_documents(docs)

# embeddings_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'}) 

# vectorstore = Chroma.from_documents(docs, embeddings_function, persist_directory="./chroma_db_nccn") 

# print(vectorstore._collection.count())

# generate_embeding.py

# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Define the function to load data and split documents
# def load_and_split_docs():
#     urls = [
#         'https://www.livemint.com/economy/budget-2024-key-highlights-live-updates-nirmala-sitharaman-infrastructure-defence-income-tax-modi-budget-23-july-11721654502862.html',
#         'https://cleartax.in/s/budget-2024-highlights',
#         'https://www.hindustantimes.com/budget',
#         'https://economictimes.indiatimes.com/news/economy/policy/budget-2024-highlights-india-nirmala-sitharaman-capex-fiscal-deficit-tax-slab-key-announcement-in-union-budget-2024-25/articleshow/111942707.cms?from=mdr'
#     ]
#     loader = UnstructuredURLLoader(urls=urls)
#     data = loader.load()

#     # Split the data into smaller chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
#     docs = text_splitter.split_documents(data)
    
#     return docs

# # Load and split documents
# docs = load_and_split_docs()


# from langchain_community.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings()

# vector = embeddings.embed_query("hello, world!")

# print(vector[:5])   