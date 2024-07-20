from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

# Load PDF document and create doc splits
def load_doc(file_path, chunk_size, chunk_overlap):
    print("Loading document...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)
    print(f"Created {len(doc_splits)} document splits.")
    return doc_splits

def main():
    file_path = "N:\\AI\\Lama3AI-master\\pdfFiles\\ppp.pdf"  # Example file path
    chunk_size = 600
    chunk_overlap = 40
    
    # Load and split document
    doc_splits = load_doc(file_path, chunk_size, chunk_overlap)
    for i, split in enumerate(doc_splits):
        print(f"Document Split {i+1}: {split.page_content[:200]}...")  # Print first 200 characters of each split

if __name__ == "__main__":
    main()
