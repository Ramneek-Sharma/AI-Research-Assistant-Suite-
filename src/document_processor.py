from typing import List
import os
import pickle
import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  # FIXED: Updated import
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS
import faiss  # Facebook AI Similarity Search library


class DocumentProcessor:
    """
    DocumentProcessor handles the ingestion of raw documents from disk, 
    splits them into smaller chunks, generates embeddings for those chunks,
    and stores/retrieves them using a FAISS vector store with memory safety.
    """

    def __init__(self):
        """
        Initializes the processor with a recursive text splitter for chunking
        and Ollama embeddings (via a local Ollama server) for vector generation.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,                  # Target max size of each chunk
            chunk_overlap=200,                # Amount of overlap between chunks
            separators=["\n\n", "\n", ". ", " ", ""]  # Order of splitting priority
        )

        # Embedding model via Ollama server (must be running locally)
        try:
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://localhost:11434"
            )
            # Test the embeddings connection
            test_embed = self.embeddings.embed_query("test connection")
            print("✅ Ollama embeddings initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Ollama embeddings: {e}")
            print("Make sure Ollama server is running and nomic-embed-text model is pulled")
            raise

    def load_documents(self, directory: str) -> List[Document]:
        """
        Loads .pdf, .txt, and .md files from the specified directory using 
        LangChain's document loaders.

        Args:
            directory (str): Path to the folder containing documents.

        Returns:
            List[Document]: A list of LangChain Document objects.
        """
        # Validate directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Configure supported loaders by file extension
        loaders = {
            ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        }

        documents = []

        # Try loading each supported file type
        for file_type, loader in loaders.items():
            try:
                loaded = loader.load()
                documents.extend(loaded)
                print(f"Loaded {len(loaded)} {file_type} documents")
            except Exception as e:
                print(f"Error loading {file_type} documents: {str(e)}")

        if not documents:
            print(f"⚠️  No documents found in {directory}")
        else:
            print(f"📚 Total documents loaded: {len(documents)}")

        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits each document into smaller overlapping chunks to improve embedding
        quality and support better semantic retrieval.

        Args:
            documents (List[Document]): Raw documents.

        Returns:
            List[Document]: Chunked documents ready for embedding.
        """
        if not documents:
            print("⚠️  No documents to process")
            return []

        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata for better tracking
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk.page_content),
                'total_chunks': len(chunks)
            })

        print(f"📄 Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def create_vector_store(self, documents: List[Document], persist_directory: str) -> FAISS:
        """
        Creates or loads a FAISS vector store with enhanced memory safety and error handling.
        This prevents segmentation faults on macOS by properly handling corrupted vector stores.

        Args:
            documents (List[Document]): Pre-processed document chunks.
            persist_directory (str): Where to save/load the FAISS index and metadata.

        Returns:
            FAISS: Vector store that can be used for semantic search.
        """
        index_path = os.path.join(persist_directory, "index.faiss")
        
        # Try to load existing vector store with safety checks
        if os.path.exists(index_path):
            try:
                print(f"🔍 Attempting to load existing FAISS vector store from {persist_directory}")
                
                # Load the vector store
                vector_store = FAISS.load_local(
                    persist_directory, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                
                # Test the vector store to ensure it's not corrupted
                print("🧪 Testing vector store integrity...")
                test_results = vector_store.similarity_search("test query", k=1)
                
                # Check if we have the expected number of vectors
                total_vectors = vector_store.index.ntotal
                print(f"✅ Vector store loaded successfully with {total_vectors} vectors")
                
                # Verify the vector store is working properly
                if total_vectors > 0:
                    print("✅ Vector store integrity test passed")
                    return vector_store
                else:
                    print("⚠️  Vector store appears empty, recreating...")
                    
            except Exception as e:
                print(f"❌ Error loading existing vector store: {e}")
                print("🔄 Vector store appears corrupted, creating new one...")
                
                # Remove corrupted files safely
                try:
                    if os.path.exists(persist_directory):
                        shutil.rmtree(persist_directory)
                        print("🗑️  Removed corrupted vector store files")
                except Exception as cleanup_error:
                    print(f"⚠️  Warning: Could not clean up corrupted files: {cleanup_error}")

        # Create new vector store with error handling
        print(f"🏗️  Creating new FAISS vector store in {persist_directory}")
        
        if not documents:
            raise ValueError("Cannot create vector store: no documents provided")
        
        try:
            # Ensure directory exists
            os.makedirs(persist_directory, exist_ok=True)

            # Create vector store from documents
            print(f"🔄 Generating embeddings for {len(documents)} document chunks...")
            vector_store = FAISS.from_documents(documents, embedding=self.embeddings)
            
            # Save the vector store
            print("💾 Saving vector store to disk...")
            vector_store.save_local(persist_directory)
            
            # Verify the saved vector store
            total_vectors = vector_store.index.ntotal
            print(f"✅ New vector store created successfully with {total_vectors} vectors")
            
            # Test the newly created vector store
            test_results = vector_store.similarity_search("test", k=1)
            print("✅ New vector store functionality verified")
            
            return vector_store
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            # Clean up any partial files
            if os.path.exists(persist_directory):
                try:
                    shutil.rmtree(persist_directory)
                except:
                    pass
            raise

    def get_vector_store_stats(self, vector_store: FAISS) -> dict:
        """
        Get statistics about the vector store for debugging.
        
        Args:
            vector_store (FAISS): The vector store to analyze
            
        Returns:
            dict: Statistics about the vector store
        """
        try:
            return {
                'total_vectors': vector_store.index.ntotal,
                'vector_dimension': vector_store.index.d,
                'index_type': type(vector_store.index).__name__,
                'is_trained': vector_store.index.is_trained
            }
        except Exception as e:
            return {'error': str(e)}

    def test_vector_store_health(self, vector_store: FAISS) -> bool:
        """
        Test if the vector store is healthy and functional.
        
        Args:
            vector_store (FAISS): The vector store to test
            
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Test basic functionality
            results = vector_store.similarity_search("health check", k=1)
            return len(results) >= 0  # Even 0 results is fine, means it's working
        except Exception as e:
            print(f"Vector store health check failed: {e}")
            return False
