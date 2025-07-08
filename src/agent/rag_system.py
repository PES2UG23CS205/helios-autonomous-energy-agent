import os
import streamlit as st
import traceback

# LangChain Community components
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS

# --- NEW: Import the Ollama LLM and HuggingFace Embeddings ---
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

# Core LangChain components
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAGSystem:
    # We will use the model we downloaded with 'ollama pull'
    def __init__(self, knowledge_base_dir="knowledge_base", model_name="llama3:8b"):
        
        # --- NO API KEYS NEEDED! We are initializing the local LLM ---
        try:
            self.llm = OllamaLLM(model=model_name)
            # A quick test to see if the Ollama server is running
            self.llm.invoke("Hi") 
            print("✅ Ollama LLM is running and connected.")
        except Exception as e:
            st.error("❌ Failed to connect to local Ollama server. Is it running?", icon="🔌")
            st.info("Please make sure you have installed Ollama and have run `ollama pull llama3:8b` in your terminal.")
            print(f"Ollama connection error: {e}")
            st.stop()
            
        self.knowledge_base_dir = knowledge_base_dir
        self.db = self._create_vector_db()
        retriever = self.db.as_retriever()
        
        # This prompt is designed to make the LLM reason about the context it's given
        template = """
        You are an expert assistant for the Helios Energy Management System. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer from the provided context, clearly state that the information is not in your knowledge base. Do not make up information.
        Provide a concise and helpful answer.

        Context: {context}

        Question: {question}

        Answer:"""
        prompt_template = PromptTemplate.from_template(template)

        self.qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

    @st.cache_resource(show_spinner="🔄 Indexing knowledge base...")
    def _create_vector_db(_self):
        try:
            loader = DirectoryLoader(_self.knowledge_base_dir, glob="**/*.txt", show_progress=True)
            documents = loader.load()
            if not documents:
                st.warning("⚠️ No documents found in `knowledge_base` directory. Agent will have limited knowledge.")
                # Return an empty FAISS index if no documents are found
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                # Create an index with a placeholder text to avoid errors
                placeholder_db = FAISS.from_texts(["No documents available."], embeddings)
                return placeholder_db

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150) # Increased overlap for better context
            texts = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(texts, embeddings)
            print("✅ Knowledge base indexed and cached.")
            return db
        except Exception as e:
            st.error(f"Fatal Error: Failed to create vector database from knowledge base. Error: {e}", icon="🚨")
            st.stop()

    def query(self, question: str):
        """Queries the RAG system and handles errors gracefully."""
        try:
            return self.qa_chain.invoke(question)
        except Exception:
            st.error("❌ A critical error occurred while querying the language model.")
            print("--- RAG SYSTEM CRITICAL TRACEBACK ---")
            traceback.print_exc()
            print("-----------------------------------")
            return "⚠️ I'm sorry, a system error prevented me from answering."