import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
FAISS_PATH = "faiss_index"

def get_embeddings():
    """Returns the HuggingFace embeddings model."""
    # Using a fast, local embedding model
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store():
    """Reads documents from the data directory and builds the FAISS index."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR} directory. Please add text files here.")
        return None

    documents = []
    
    # Load TXT files
    txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())
    
    # Load PDF files
    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())
    
    # Load DOCX files
    docx_loader = DirectoryLoader(DATA_DIR, glob="**/*.docx", loader_cls=Docx2txtLoader)
    documents.extend(docx_loader.load())

    if not documents:
        print("No documents found in the data directory.")
        return None

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # Generate embeddings and create FAISS index
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save the index locally
    vector_store.save_local(FAISS_PATH)
    print(f"Successfully built and saved FAISS index to {FAISS_PATH}")
    return vector_store

def load_vector_store():
    """Loads the FAISS index from disk."""
    if not os.path.exists(FAISS_PATH):
        return None
    
    embeddings = get_embeddings()
    # Allow dangerous deserialization since we created this file locally
    return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    build_vector_store()
