# Import necessary modules
from src.document_processor import DocumentProcessor
from src.voice_assistant_rag import VoiceAssistantRAG
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

# Fix for MKL libraries issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setup_knowledge_base():
    """
    Handle document upload and processing.
    """
    st.title("📚 Knowledge Base Setup")

    doc_processor = DocumentProcessor()

    uploaded_files = st.file_uploader(
        "Upload your documents", 
        accept_multiple_files=True, 
        type=["pdf", "txt", "md"],
        help="Upload PDF, TXT, or MD files to create your knowledge base"
    )

    if uploaded_files and st.button("🔄 Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            temp_dir = tempfile.mkdtemp()

            try:
                # Save uploaded files
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                # Process documents
                documents = doc_processor.load_documents(temp_dir)
                processed_docs = doc_processor.process_documents(documents)

                # Create vector store
                vector_store = doc_processor.create_vector_store(
                    processed_docs, "knowledge_base"
                )

                # Store in session state
                st.session_state.vector_store = vector_store
                st.session_state.total_chunks = len(processed_docs)
                st.session_state.total_docs = len(documents)

                st.success(f"✅ Successfully processed {len(documents)} documents into {len(processed_docs)} chunks!")
                
                # Show processing stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", len(documents))
                with col2:
                    st.metric("Text Chunks", len(processed_docs))
                with col3:
                    st.metric("Vector Store", "Ready ✅")

            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")

            finally:
                # Cleanup
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)

def main():
    """
    Main Streamlit application with voice input and text-only responses.
    """
    st.set_page_config(
        page_title="🎤 Voice RAG Assistant", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load environment variables (ElevenLabs API key not needed now)
    load_dotenv()

    # Sidebar navigation
    st.sidebar.title("🎤 Voice RAG Assistant")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "📍 Navigation", 
        ["📚 Setup Knowledge Base", "🎤 Voice Assistant"],
        help="Choose between setting up documents or using the voice assistant"
    )

    if page == "📚 Setup Knowledge Base":
        setup_knowledge_base()

    else:  # Voice Assistant
        if "vector_store" not in st.session_state:
            st.error("❌ Please setup knowledge base first!")
            st.info("👈 Go to 'Setup Knowledge Base' in the sidebar to upload your documents.")
            return

        st.title("🎤 Voice Assistant (Voice Input + Text Response)")
        st.markdown("*Ask questions using your voice and get text responses*")

        # Initialize assistant (no API key needed now)
        if "assistant" not in st.session_state:
            try:
                with st.spinner("🔧 Initializing Voice Assistant..."):
                    assistant = VoiceAssistantRAG()  # No API key needed
                    assistant.setup_vector_store(st.session_state.vector_store)
                    st.session_state.assistant = assistant
                st.success("✅ Voice Assistant ready!")
            except Exception as e:
                st.error(f"❌ Error initializing assistant: {e}")
                return

        assistant = st.session_state.assistant

        # Show knowledge base stats
        st.sidebar.markdown("### 📊 Knowledge Base")
        if "total_docs" in st.session_state:
            st.sidebar.metric("Documents", st.session_state.total_docs)
            st.sidebar.metric("Text Chunks", st.session_state.total_chunks)

        # Voice recording settings
        st.sidebar.markdown("### 🎤 Voice Settings")
        duration = st.sidebar.slider("Recording Duration (seconds)", 1, 10, 5)

        # Main interface
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 🎙️ Voice Input")
            
            if st.button("🎤 Start Recording", type="primary", use_container_width=True):
                try:
                    with st.spinner(f"🎙️ Recording for {duration} seconds..."):
                        audio_data = assistant.record_audio(duration)
                        st.session_state.audio_data = audio_data
                    st.success("✅ Recording completed!")
                except Exception as e:
                    st.error(f"❌ Recording error: {e}")

        with col2:
            st.markdown("#### 📝 Process Voice")
            
            if st.button("🔄 Convert Voice to Text & Get Answer", type="secondary", use_container_width=True):
                if "audio_data" not in st.session_state:
                    st.error("❌ Please record audio first!")
                    return

                try:
                    # Step 1: Voice to text
                    with st.spinner("📝 Converting voice to text..."):
                        query = assistant.transcribe_audio(st.session_state.audio_data)
                        
                    if query and query.strip():
                        st.markdown("### 🗣️ You said:")
                        st.info(f"'{query}'")
                        
                        # Step 2: Generate text response
                        with st.spinner("🤔 Generating response..."):
                            response = assistant.generate_response(query)
                            
                        st.markdown("### 🤖 AI Response:")
                        st.markdown(f"**{response}**")
                        
                        # Store in chat history
                        if "chat_history" not in st.session_state:
                            st.session_state.chat_history = []
                        
                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": response,
                            "timestamp": st.experimental_get_query_params().get("timestamp", [""])[0]
                        })
                        
                    else:
                        st.warning("⚠️ Could not understand the audio. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing voice: {e}")

        # Text input alternative
        st.markdown("---")
        st.markdown("#### ⌨️ Or Type Your Question")
        
        text_query = st.text_input(
            "Enter your question:", 
            placeholder="What would you like to know about your documents?",
            key="text_input"
        )
        
        if st.button("📤 Submit Text Question", use_container_width=True) and text_query:
            try:
                with st.spinner("🤔 Generating response..."):
                    response = assistant.generate_response(text_query)
                    
                st.markdown("### 🤖 AI Response:")
                st.markdown(f"**{response}**")
                
                # Store in chat history
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append({
                    "question": text_query,
                    "answer": response
                })
                
            except Exception as e:
                st.error(f"❌ Error: {e}")

        # Chat history
        if "chat_history" in st.session_state and st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 📜 Chat History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"💬 Q{len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."):
                    st.markdown(f"**❓ Question:** {chat['question']}")
                    st.markdown(f"**🤖 Answer:** {chat['answer']}")

        # Clear history button
        if st.sidebar.button("🗑️ Clear Chat History"):
            if "chat_history" in st.session_state:
                st.session_state.chat_history = []
            assistant.clear_conversation_history()
            st.rerun()

if __name__ == "__main__":
    main()
