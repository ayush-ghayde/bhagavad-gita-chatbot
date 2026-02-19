from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Step 1: Load raw PDF(s)
DATA_PATH = "data/" 

def load_pdf_files(data):
    print(" Step 1: Loading PDF files from folder...")
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} pages.")
    return documents

# Step 2: Create Chunks
def create_chunks(extracted_data):
    print(" Step 2: Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f" Created {len(text_chunks)} text chunks.")
    return text_chunks

# Step 3: Create Vector Embeddings 
def get_embedding_model():
    print(" Step 3: Initializing Embedding Model (Downloading if first time)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(" Embedding model ready.")
    return embedding_model

if __name__ == "__main__":
    # Check if data folder exists
    if not os.path.exists(DATA_PATH):
        print(f" Error: '{DATA_PATH}' folder does not exist. Please create it and add PDFs.")
    else:
        # Execute Pipeline
        documents = load_pdf_files(data=DATA_PATH)
        text_chunks = create_chunks(extracted_data=documents)
        embedding_model = get_embedding_model()

        # Step 4: Store embeddings in FAISS
        print(" Step 4: Generating Vectors and saving to FAISS (Wait for 1-2 mins)...")
        DB_FAISS_PATH = "vectorstore/db_faiss"
        
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        
        print(f" Success! Vector database saved at: {DB_FAISS_PATH}")