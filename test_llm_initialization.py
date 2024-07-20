from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from pathlib import Path

def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db):
    print("Initializing LLM chain...")
    llm = HuggingFaceEndpoint(
        repo_id=llm_model,
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_k=top_k,
    )
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

def main():
    # Assume we have vector_db already created
    # For testing purposes, you might need to pass an actual vector_db here
    vector_db = ...  # Load or pass the vector_db created in the previous step
    
    llm_model = "meta-llama/Llama-2-7b-chat-hf"
    llm_temperature = 0.7
    max_tokens = 1024
    top_k = 3
    
    qa_chain = initialize_llmchain(llm_model, llm_temperature, max_tokens, top_k, vector_db)
    if qa_chain is not None:
        print("LLM chain initialization succeeded.")
    else:
        print("LLM chain initialization failed.")

if __name__ == "__main__":
    main()
