import os
from dotenv import load_dotenv

# Load the .env file for GOOGLE_API_KEY
load_dotenv()

# Optional: Check if API key is loaded correctly
api_key = os.getenv("GOOGLE_API_KEY")
print("Loaded GOOGLE_API_KEY:", api_key)

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


def initialize_rag():
    # ✅ Correct path to your PDF
    loader = PyPDFLoader("prospectus2025-2026.pdf")
    
    # Load the PDF as documents
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Set up Gemini Pro model
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    # Setup Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )

    return qa_chain
