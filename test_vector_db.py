from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import chromadb

# Load PDF document and create doc splits
def load_doc(file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

def create_db(splits, collection_name):
    try:
        print("Creating vector database...")
        
        # Initialize embedding model
        try:
            embedding = HuggingFaceEmbeddings()
            print("Embedding model initialized.")
        except Exception as e:
            print(f"Failed to initialize embedding model: {e}")
            return None
        
        # Initialize ChromaDB client
        try:
            new_client = chromadb.EphemeralClient()
            print("ChromaDB client initialized.")
        except Exception as e:
            print(f"Failed to initialize ChromaDB client: {e}")
            return None
        
        # Convert splits to Document objects
        try:
            documents = [Document(page_content=split.page_content, metadata=split.metadata) for split in splits]
            print(f"Prepared {len(documents)} documents for vector database.")
        except Exception as e:
            print(f"Failed to convert splits to Document objects: {e}")
            return None
        
        # Create vector database
        try:
            print(f"Creating Chroma vector database with collection name: {collection_name}")
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=embedding,
                client=new_client,
                collection_name=collection_name
            )
            print(vectordb)
            print("Vector database created.")
            return vectordb
        except Exception as e:
            print(f"Failed to create vector database: {e}")
            return None
    
    except Exception as e:
        print(f"Unexpected error during vector database creation: {e}")
        return None

# Example usage
def main():
    file_path = "N:\\AI\\Lama3AI-master\\pdfFiles\\ppp.pdf"
    chunk_size = 600
    chunk_overlap = 40

    # Load and split document
    doc_splits = load_doc(file_path, chunk_size, chunk_overlap)
    
    # Create vector database
    collection_name = Path(file_path).stem
    vector_db = create_db(doc_splits, collection_name)
    if vector_db is not None:
        print("Vector database creation succeeded.")
    else:
        print("Vector database creation failed.")

if __name__ == "__main__":
    main()
