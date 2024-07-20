import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import Document  # Import Document class

from pathlib import Path
import chromadb
from unidecode import unidecode

from transformers import AutoTokenizer
import transformers
import torch
import re

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

def create_db(splits, collection_name):
    print("Creating vector database...")

    try:
        # Initialize the embedding model
        embedding = HuggingFaceEmbeddings()
        print("Embedding model initialized.")
        
        # Initialize the ChromaDB client
        new_client = chromadb.EphemeralClient()
        print("ChromaDB client initialized.")

        # Convert splits to Document objects if they are not already
        documents = []
        for split in splits:
            try:
                doc = Document(page_content=split.page_content, metadata=split.metadata)
                documents.append(doc)
            except Exception as e:
                print(f"Error converting split to Document: {e}")
        print(f"Prepared {len(documents)} documents for vector database.")
        
        try:
            # vectordb = Chroma.from_documents(
            #     documents=documents,
            #     embedding=embedding,
            #     client=new_client,
            #     collection_name=collection_name,
            #     persist_directory="chroma_db"
            # )
            vectordb = Chroma(persist_directory="./db", embedding_function=embedding)
            print("Vector database created.")
            return vectordb
        except Exception as e:
            print(f"Exception occurred while creating Chroma document: {e}")
            return None
    except Exception as e:
        print(f"Error during vector database creation: {e}")
        return None

# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db):
    print("Initializing LLM chain...")
    if llm_model == "meta-llama/Llama-2-7b-chat-hf":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_k=top_k,
        )
    else:
        raise Exception("Unsupported LLM model for this setup")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    retriever = vector_db.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    print("LLM chain initialized.")
    return qa_chain

# Generate collection name for vector database
def create_collection_name(filepath):
    collection_name = Path(filepath).stem
    collection_name = collection_name.replace(" ", "-")
    collection_name = unidecode(collection_name)
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    collection_name = collection_name[:50]
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    return collection_name

# Initialize database
def initialize_database(file_path, chunk_size, chunk_overlap):
    print("Initializing database...")
    collection_name = create_collection_name(file_path)
    print(f"Collection name: {collection_name}")
    doc_splits = load_doc(file_path, chunk_size, chunk_overlap)
    vector_db = create_db(doc_splits, collection_name)
    print("Database initialized.")
    return vector_db, collection_name

# Initialize LLM
def initialize_LLM(llm_temperature, max_tokens, top_k, vector_db):
    llm_name = "meta-llama/Llama-2-7b-chat-hf"
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db)
    return qa_chain

# Format chat history
def format_chat_history(chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history

# Conversation
def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(history)
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip() if response_sources else "No source"
    response_source2 = response_sources[1].page_content.strip() if len(response_sources) > 1 else "No source"
    response_source3 = response_sources[2].page_content.strip() if len(response_sources) > 2 else "No source"
    response_source1_page = response_sources[0].metadata["page"] + 1 if response_sources else "N/A"
    response_source2_page = response_sources[1].metadata["page"] + 1 if len(response_sources) > 1 else "N/A"
    response_source3_page = response_sources[2].metadata["page"] + 1 if len(response_sources) > 2 else "N/A"
    new_history = history + [(message, response_answer)]
    return response_answer, new_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page

def main():
    # Upload PDF
    file_path = "N:\\AI\\Lama3AI-master\\pdfFiles\\ppp.pdf"  # Example file path
    chunk_size = 600
    chunk_overlap = 40

    # Process document
    vector_db, collection_name = initialize_database(file_path, chunk_size, chunk_overlap)
    
    if vector_db is None:
        print("Failed to create vector database. Exiting...")
        return
    
    print(f"Vector database created with collection name: {collection_name}")

    # Initialize QA chain
    llm_temperature = 0.7
    max_tokens = 1024
    top_k = 3
    qa_chain = initialize_LLM(llm_temperature, max_tokens, top_k, vector_db)
    
    if qa_chain is None:
        print("Failed to initialize LLM chain. Exiting...")
        return

    print("LLM chain initialized")

    # Chatbot interaction
    chat_history = []
    while True:
        user_message = input("You: ")
        if user_message.lower() in ['exit', 'quit']:
            break
        response_answer, chat_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page = conversation(qa_chain, user_message, chat_history)
        print(f"Assistant: {response_answer}")
        print(f"Source 1 (Page {response_source1_page}): {response_source1}")
        print(f"Source 2 (Page {response_source2_page}): {response_source2}")
        print(f"Source 3 (Page {response_source3_page}): {response_source3}")

if __name__ == "__main__":
    main()
