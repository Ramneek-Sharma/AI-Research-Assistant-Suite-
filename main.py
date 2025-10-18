# # # Import necessary modules
# # from src.document_processor import DocumentProcessor
# # from src.voice_assistant_rag import VoiceAssistantRAG
# # import streamlit as st
# # import tempfile
# # import os
# # from dotenv import load_dotenv
# # import pandas as pd

# # # Fix for MKL libraries issue
# # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # def setup_knowledge_base():
# #     """
# #     Enhanced document upload and processing with analytics.
# #     """
# #     st.title("📚 Enhanced Knowledge Base Setup")

# #     doc_processor = DocumentProcessor()

# #     uploaded_files = st.file_uploader(
# #         "Upload your documents", 
# #         accept_multiple_files=True, 
# #         type=["pdf", "txt", "md"],
# #         help="Upload PDF, TXT, or MD files to create your intelligent knowledge base"
# #     )

# #     if uploaded_files and st.button("🔄 Process Documents", type="primary"):
# #         with st.spinner("Processing documents with AI intelligence..."):
# #             temp_dir = tempfile.mkdtemp()

# #             try:
# #                 # Save uploaded files
# #                 for file in uploaded_files:
# #                     file_path = os.path.join(temp_dir, file.name)
# #                     with open(file_path, "wb") as f:
# #                         f.write(file.getbuffer())

# #                 # Process documents
# #                 documents = doc_processor.load_documents(temp_dir)
# #                 processed_docs = doc_processor.process_documents(documents)

# #                 # Create vector store
# #                 vector_store = doc_processor.create_vector_store(
# #                     processed_docs, "knowledge_base"
# #                 )

# #                 # Store in session state
# #                 st.session_state.vector_store = vector_store
# #                 st.session_state.documents = documents
# #                 st.session_state.processed_docs = processed_docs
# #                 st.session_state.total_chunks = len(processed_docs)
# #                 st.session_state.total_docs = len(documents)

# #                 st.success(f"✅ Successfully processed {len(documents)} documents into {len(processed_docs)} chunks!")
                
# #                 # Enhanced stats display
# #                 col1, col2, col3, col4 = st.columns(4)
                
# #                 with col1:
# #                     st.metric("Documents", len(documents))
# #                 with col2:
# #                     st.metric("Text Chunks", len(processed_docs))
# #                 with col3:
# #                     st.metric("Vector Store", "Ready ✅")
# #                 with col4:
# #                     avg_chunk_size = sum(len(doc.page_content) for doc in processed_docs) / len(processed_docs)
# #                     st.metric("Avg Chunk Size", f"{avg_chunk_size:.0f}")

# #                 # Document analysis (basic version without external dependencies)
# #                 st.markdown("### 📊 Document Analysis")
                
# #                 # Basic document stats
# #                 doc_stats = []
# #                 for i, doc in enumerate(documents):
# #                     content = doc.page_content
# #                     doc_stats.append({
# #                         'filename': os.path.basename(doc.metadata.get('source', f'document_{i}')),
# #                         'word_count': len(content.split()),
# #                         'char_count': len(content),
# #                         'estimated_reading_time': len(content.split()) / 200  # ~200 words per minute
# #                     })
                
# #                 # Display document stats
# #                 if doc_stats:
# #                     stats_df = pd.DataFrame(doc_stats)
# #                     st.dataframe(stats_df, use_container_width=True)
                    
# #                     # Simple visualizations
# #                     col1, col2 = st.columns(2)
                    
# #                     with col1:
# #                         # Word count distribution
# #                         st.markdown("#### 📈 Word Count Distribution")
# #                         st.bar_chart(stats_df.set_index('filename')['word_count'])
                    
# #                     with col2:
# #                         # Reading time estimation
# #                         st.markdown("#### ⏱️ Estimated Reading Time (minutes)")
# #                         st.bar_chart(stats_df.set_index('filename')['estimated_reading_time'])

# #             except Exception as e:
# #                 st.error(f"❌ Error processing documents: {str(e)}")
# #                 st.info("💡 Make sure Ollama server is running with required models")

# #             finally:
# #                 # Cleanup
# #                 try:
# #                     for file in os.listdir(temp_dir):
# #                         os.remove(os.path.join(temp_dir, file))
# #                     os.rmdir(temp_dir)
# #                 except Exception as cleanup_error:
# #                     print(f"Warning: Could not clean up temp directory: {cleanup_error}")

# # def analytics_dashboard():
# #     """
# #     Analytics dashboard for document insights.
# #     """
# #     if "processed_docs" not in st.session_state:
# #         st.error("❌ Please setup knowledge base first!")
# #         st.info("👈 Go to 'Setup Knowledge Base' to upload and analyze your documents.")
# #         return
    
# #     st.title("📊 Document Analytics Dashboard")
    
# #     processed_docs = st.session_state.processed_docs
# #     documents = st.session_state.documents
    
# #     # Summary metrics
# #     col1, col2, col3, col4 = st.columns(4)
    
# #     with col1:
# #         st.metric("Total Documents", len(documents))
# #     with col2:
# #         st.metric("Text Chunks", len(processed_docs))
# #     with col3:
# #         total_words = sum(len(doc.page_content.split()) for doc in processed_docs)
# #         st.metric("Total Words", f"{total_words:,}")
# #     with col4:
# #         avg_words = total_words / len(processed_docs) if processed_docs else 0
# #         st.metric("Avg Words/Chunk", f"{avg_words:.0f}")
    
# #     # Document analysis
# #     st.markdown("### 📈 Detailed Document Analysis")
    
# #     # Create analysis data
# #     analysis_data = []
# #     for i, doc in enumerate(documents):
# #         content = doc.page_content
# #         analysis_data.append({
# #             'filename': os.path.basename(doc.metadata.get('source', f'document_{i}')),
# #             'word_count': len(content.split()),
# #             'char_count': len(content),
# #             'sentences': len([s for s in content.split('.') if s.strip()]),
# #             'reading_time_min': len(content.split()) / 200
# #         })
    
# #     if analysis_data:
# #         analysis_df = pd.DataFrame(analysis_data)
        
# #         # Interactive filtering
# #         col1, col2 = st.columns(2)
        
# #         with col1:
# #             # Filter by filename
# #             selected_files = st.multiselect(
# #                 "Select Documents",
# #                 options=analysis_df['filename'].tolist(),
# #                 default=analysis_df['filename'].tolist()[:5] if len(analysis_df) > 5 else analysis_df['filename'].tolist()
# #             )
        
# #         with col2:
# #             # FIXED: Safe word count filter
# #             min_word_count = int(analysis_df['word_count'].min())
# #             max_word_count = int(analysis_df['word_count'].max())
            
# #             # Handle case where min equals max (prevents slider error)
# #             if min_word_count == max_word_count:
# #                 st.info(f"All documents have {min_word_count} words")
# #                 word_filter = min_word_count
# #             else:
# #                 word_filter = st.slider(
# #                     "Minimum Word Count", 
# #                     min_word_count, 
# #                     max_word_count, 
# #                     min_word_count
# #                 )
        
# #         # Apply filters
# #         filtered_df = analysis_df[
# #             (analysis_df['filename'].isin(selected_files)) &
# #             (analysis_df['word_count'] >= (word_filter if min_word_count != max_word_count else min_word_count))
# #         ]
        
# #         # Display filtered data
# #         st.dataframe(filtered_df, use_container_width=True)
        
# #         # Visualizations
# #         if not filtered_df.empty:
# #             col1, col2 = st.columns(2)
            
# #             with col1:
# #                 st.markdown("#### 📊 Word Count by Document")
# #                 st.bar_chart(filtered_df.set_index('filename')['word_count'])
            
# #             with col2:
# #                 st.markdown("#### ⏱️ Reading Time Distribution")
# #                 st.bar_chart(filtered_df.set_index('filename')['reading_time_min'])

# # def main():
# #     """
# #     Enhanced main application with analytics and better error handling.
# #     """
# #     st.set_page_config(
# #         page_title="🧠 InsightForge AI Research Suite", 
# #         layout="wide",
# #         initial_sidebar_state="expanded"
# #     )

# #     # Load environment variables
# #     load_dotenv()

# #     # Enhanced sidebar
# #     st.sidebar.title("🧠 InsightForge AI Research Suite")
# #     st.sidebar.markdown("---")
    
# #     page = st.sidebar.radio(
# #         "📍 Navigation", 
# #         ["📚 Setup Knowledge Base", "🎤 Voice Assistant", "📊 Analytics Dashboard"],
# #         help="Navigate between document setup, voice interaction, and analytics"
# #     )

# #     # Show system status in sidebar
# #     st.sidebar.markdown("### 🔧 System Status")
    
# #     # Check Ollama status
# #     try:
# #         import requests
# #         response = requests.get("http://localhost:11434", timeout=2)
# #         st.sidebar.success("🟢 Ollama Server: Online")
# #     except:
# #         st.sidebar.error("🔴 Ollama Server: Offline")
# #         st.sidebar.info("Start with: `ollama serve`")

# #     if page == "📚 Setup Knowledge Base":
# #         setup_knowledge_base()

# #     elif page == "📊 Analytics Dashboard":
# #         analytics_dashboard()

# #     else:  # Voice Assistant
# #         if "vector_store" not in st.session_state:
# #             st.error("❌ Please setup knowledge base first!")
# #             st.info("👈 Go to 'Setup Knowledge Base' to upload your documents.")
# #             return

# #         st.title("🎤 Voice Assistant (Voice Input + Text Response)")
# #         st.markdown("*Ask questions using your voice and get intelligent text responses*")

# #         # Initialize assistant with better error handling
# #         if "assistant" not in st.session_state:
# #             try:
# #                 with st.spinner("🔧 Initializing Voice Assistant..."):
# #                     assistant = VoiceAssistantRAG()
# #                     assistant.setup_vector_store(st.session_state.vector_store)
# #                     st.session_state.assistant = assistant
# #                 st.success("✅ Voice Assistant ready!")
# #             except Exception as e:
# #                 st.error(f"❌ Error initializing assistant: {e}")
# #                 st.info("💡 Make sure Ollama server is running and models are downloaded")
# #                 return

# #         assistant = st.session_state.assistant

# #         # Enhanced sidebar with intelligence stats
# #         st.sidebar.markdown("### 📊 Knowledge Base")
# #         if "total_docs" in st.session_state:
# #             st.sidebar.metric("Documents", st.session_state.total_docs)
# #             st.sidebar.metric("Text Chunks", st.session_state.total_chunks)

# #         # Voice recording settings with enhanced duration
# #         st.sidebar.markdown("### 🎤 Voice Settings")
# #         duration = st.sidebar.slider("Recording Duration (seconds)", 1, 30, 15)
        
# #         # Add recording tips
# #         if duration > 20:
# #             st.sidebar.info("💡 Longer recordings work better for complex questions")
# #         elif duration < 5:
# #             st.sidebar.warning("⚠️ Very short recordings might miss parts of your question")

# #         # Main interface
# #         col1, col2 = st.columns([1, 1])

# #         with col1:
# #             st.markdown("#### 🎙️ Voice Input")
            
# #             if st.button("🎤 Start Recording", type="primary", use_container_width=True):
# #                 try:
# #                     with st.spinner(f"🎙️ Recording for {duration} seconds..."):
# #                         audio_data = assistant.record_audio(duration)
# #                         st.session_state.audio_data = audio_data
# #                     st.success("✅ Recording completed!")
# #                 except Exception as e:
# #                     st.error(f"❌ Recording error: {e}")
# #                     st.info("💡 Make sure your microphone is connected and working")

# #         with col2:
# #             st.markdown("#### 📝 Process Voice")
            
# #             if st.button("🔄 Convert Voice to Text & Get Answer", type="secondary", use_container_width=True):
# #                 if "audio_data" not in st.session_state:
# #                     st.error("❌ Please record audio first!")
# #                     return

# #                 try:
# #                     # Step 1: Voice to text
# #                     with st.spinner("📝 Converting voice to text..."):
# #                         query = assistant.transcribe_audio(st.session_state.audio_data)
                        
# #                     if query and query.strip():
# #                         st.markdown("### 🗣️ You said:")
# #                         st.info(f"'{query}'")
                        
# #                         # Step 2: Generate text response
# #                         with st.spinner("🤔 Generating response..."):
# #                             response = assistant.generate_response(query)
                            
# #                         st.markdown("### 🤖 AI Response:")
# #                         st.markdown(f"**{response}**")
                        
# #                         # Store in chat history
# #                         if "chat_history" not in st.session_state:
# #                             st.session_state.chat_history = []
                        
# #                         st.session_state.chat_history.append({
# #                             "question": query,
# #                             "answer": response,
# #                             "type": "voice"
# #                         })
                        
# #                     else:
# #                         st.warning("⚠️ Could not understand the audio. Please try again.")
                        
# #                 except Exception as e:
# #                     st.error(f"❌ Error processing voice: {e}")

# #         # Text input alternative
# #         st.markdown("---")
# #         st.markdown("#### ⌨️ Or Type Your Question")
        
# #         text_query = st.text_input(
# #             "Enter your question:", 
# #             placeholder="What would you like to know about your documents?",
# #             key="text_input"
# #         )
        
# #         if st.button("📤 Submit Text Question", use_container_width=True) and text_query:
# #             try:
# #                 with st.spinner("🤔 Generating response..."):
# #                     response = assistant.generate_response(text_query)
                    
# #                 st.markdown("### 🤖 AI Response:")
# #                 st.markdown(f"**{response}**")
                
# #                 # Store in chat history
# #                 if "chat_history" not in st.session_state:
# #                     st.session_state.chat_history = []
                
# #                 st.session_state.chat_history.append({
# #                     "question": text_query,
# #                     "answer": response,
# #                     "type": "text"
# #                 })
                
# #             except Exception as e:
# #                 st.error(f"❌ Error: {e}")

# #         # Enhanced chat history
# #         if "chat_history" in st.session_state and st.session_state.chat_history:
# #             st.markdown("---")
# #             st.markdown("### 📜 Chat History")
            
# #             # Add search functionality
# #             search_term = st.text_input("🔍 Search chat history:", placeholder="Search questions or answers...")
            
# #             filtered_history = st.session_state.chat_history
# #             if search_term:
# #                 filtered_history = [
# #                     chat for chat in st.session_state.chat_history 
# #                     if search_term.lower() in chat['question'].lower() or search_term.lower() in chat['answer'].lower()
# #                 ]
            
# #             for i, chat in enumerate(reversed(filtered_history)):
# #                 chat_type_icon = "🎤" if chat.get('type') == 'voice' else "⌨️"
# #                 with st.expander(f"{chat_type_icon} Q{len(filtered_history)-i}: {chat['question'][:50]}..."):
# #                     st.markdown(f"**❓ Question:** {chat['question']}")
# #                     st.markdown(f"**🤖 Answer:** {chat['answer']}")

# #         # Enhanced controls
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             if st.button("🗑️ Clear Chat History", use_container_width=True):
# #                 if "chat_history" in st.session_state:
# #                     st.session_state.chat_history = []
# #                 assistant.clear_conversation_history()
# #                 st.rerun()
        
# #         with col2:
# #             if st.button("💾 Download Chat History", use_container_width=True):
# #                 if "chat_history" in st.session_state:
# #                     import json
# #                     chat_data = json.dumps(st.session_state.chat_history, indent=2)
# #                     st.download_button(
# #                         "📥 Download JSON",
# #                         chat_data,
# #                         "chat_history.json",
# #                         "application/json"
# #                     )

# # if __name__ == "__main__":
# #     main()
# # Import necessary modules
# # Import necessary modules
# # Import necessary modules
# from src.document_processor import DocumentProcessor
# from src.voice_assistant_rag import VoiceAssistantRAG
# from src.document_intelligence import DocumentIntelligence  # NEW
# from src.knowledge_graph import KnowledgeGraphBuilder      # NEW
# import streamlit as st
# import tempfile
# import os
# from dotenv import load_dotenv
# import pandas as pd
# import plotly.express as px

# # Fix for MKL libraries issue
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# def setup_knowledge_base():
#     """
#     Enhanced document upload and processing with intelligence and knowledge graphs.
#     """
#     st.title("📚 Enhanced Knowledge Base Setup")

#     doc_processor = DocumentProcessor()
#     doc_intelligence = DocumentIntelligence()      # NEW
#     knowledge_graph = KnowledgeGraphBuilder()      # NEW

#     uploaded_files = st.file_uploader(
#         "Upload your documents", 
#         accept_multiple_files=True, 
#         type=["pdf", "txt", "md"],
#         help="Upload PDF, TXT, or MD files to create your intelligent knowledge base"
#     )

#     if uploaded_files and st.button("🔄 Process Documents", type="primary"):
#         with st.spinner("Processing documents with AI intelligence..."):
#             temp_dir = tempfile.mkdtemp()

#             try:
#                 # 🔧 CRITICAL FIX: Clear old vector store and session state FIRST
#                 import shutil
#                 if os.path.exists("knowledge_base"):
#                     shutil.rmtree("knowledge_base")
#                     st.info("🗑️ Cleared previous knowledge base to ensure fresh processing")

#                 # Clear related session state
#                 session_keys_to_clear = ["vector_store", "assistant", "chat_history"]
#                 for key in session_keys_to_clear:
#                     if key in st.session_state:
#                         del st.session_state[key]
                        
#                 st.info("🔄 Starting fresh document processing...")

#                 # Save uploaded files
#                 for file in uploaded_files:
#                     file_path = os.path.join(temp_dir, file.name)
#                     with open(file_path, "wb") as f:
#                         f.write(file.getbuffer())

#                 # Process documents
#                 documents = doc_processor.load_documents(temp_dir)
#                 processed_docs = doc_processor.process_documents(documents)

#                 # NEW: Analyze document intelligence
#                 with st.spinner("🧠 Analyzing document intelligence..."):
#                     intelligence_results = doc_intelligence.categorize_documents(documents)
#                     st.session_state.intelligence_results = intelligence_results

#                 # NEW: Build knowledge graph
#                 with st.spinner("🕸️ Building knowledge graph..."):
#                     graph = knowledge_graph.build_document_relationships(documents)
#                     st.session_state.knowledge_graph = knowledge_graph

#                 # Create fresh vector store
#                 vector_store = doc_processor.create_vector_store(
#                     processed_docs, "knowledge_base"
#                 )

#                 # Store in session state
#                 st.session_state.vector_store = vector_store
#                 st.session_state.documents = documents
#                 st.session_state.processed_docs = processed_docs

#                 st.success(f"✅ Successfully processed {len(documents)} documents into {len(processed_docs)} chunks!")
                
#                 # Enhanced stats display
#                 col1, col2, col3, col4 = st.columns(4)
                
#                 with col1:
#                     st.metric("Documents", len(documents))
#                 with col2:
#                     st.metric("Text Chunks", len(processed_docs))
#                 with col3:
#                     st.metric("Categories", len(intelligence_results['categories']))
#                 with col4:
#                     st.metric("Graph Nodes", len(graph.nodes()))

#                 # NEW: Display document analysis
#                 st.markdown("### 📊 Document Intelligence Analysis")
                
#                 # Category distribution
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.markdown("#### 📂 Document Categories")
#                     categories_df = pd.DataFrame(
#                         list(intelligence_results['categories'].items()),
#                         columns=['Category', 'Count']
#                     )
#                     if not categories_df.empty:
#                         fig_pie = px.pie(categories_df, values='Count', names='Category', 
#                                        title="Document Distribution by Category")
#                         st.plotly_chart(fig_pie, use_container_width=True)
#                     else:
#                         st.info("No categories detected")
                
#                 with col2:
#                     st.markdown("#### 📈 Document Metrics")
#                     metrics_data = intelligence_results['summary_stats']
                    
#                     st.metric("Average Words per Document", 
#                              f"{metrics_data['avg_word_count']:.0f}")
#                     st.metric("Most Common Category", 
#                              metrics_data['most_common_category'].title())

#                 # NEW: Document analysis table
#                 st.markdown("#### 📋 Individual Document Analysis")
#                 analysis_df = pd.DataFrame(intelligence_results['document_analysis'])
#                 if not analysis_df.empty:
#                     analysis_df['filename'] = analysis_df['filename'].apply(lambda x: os.path.basename(x))
                    
#                     st.dataframe(
#                         analysis_df[['filename', 'category', 'word_count', 'readability_score', 'key_topics']],
#                         use_container_width=True
#                     )

#                 # NEW: Knowledge Graph Visualization
#                 st.markdown("### 🕸️ Document Relationship Graph")
#                 fig = knowledge_graph.generate_interactive_visualization()
#                 if fig:
#                     st.plotly_chart(fig, use_container_width=True)
#                     st.info("💡 Hover over nodes to see document details. Connected documents share similar content.")
#                 else:
#                     st.warning("No significant relationships found between documents. Try uploading more related documents.")

#             except Exception as e:
#                 st.error(f"❌ Error processing documents: {str(e)}")
#                 st.info("💡 Make sure Ollama server is running with required models")

#             finally:
#                 # Cleanup
#                 try:
#                     for file in os.listdir(temp_dir):
#                         os.remove(os.path.join(temp_dir, file))
#                     os.rmdir(temp_dir)
#                 except Exception as cleanup_error:
#                     print(f"Warning: Could not clean up temp directory: {cleanup_error}")

# def analytics_dashboard():
#     """
#     Analytics dashboard for document insights with intelligence features.
#     """
#     if "intelligence_results" not in st.session_state:
#         st.error("❌ Please setup knowledge base first!")
#         st.info("👈 Go to 'Setup Knowledge Base' to upload and analyze your documents.")
#         return
        
#     st.title("📊 Document Analytics Dashboard")
    
#     intelligence_results = st.session_state.intelligence_results
    
#     # Enhanced analytics display
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.metric("Total Documents", intelligence_results['summary_stats']['total_documents'])
#     with col2:
#         st.metric("Average Words", f"{intelligence_results['summary_stats']['avg_word_count']:.0f}")
#     with col3:
#         st.metric("Categories Detected", len(intelligence_results['categories']))
    
#     # Detailed analysis
#     st.markdown("### 📈 Detailed Document Analysis")
    
#     analysis_df = pd.DataFrame(intelligence_results['document_analysis'])
    
#     if not analysis_df.empty:
#         # Interactive filtering
#         col1, col2 = st.columns(2)
#         with col1:
#             selected_categories = st.multiselect(
#                 "Filter by Category",
#                 options=analysis_df['category'].unique(),
#                 default=analysis_df['category'].unique()
#             )
        
#         with col2:
#             # FIXED: Safe word count filter
#             min_word_count = int(analysis_df['word_count'].min())
#             max_word_count = int(analysis_df['word_count'].max())
            
#             if min_word_count == max_word_count:
#                 st.info(f"All documents have {min_word_count} words")
#                 word_filter = min_word_count
#             else:
#                 word_filter = st.slider(
#                     "Minimum Word Count", 
#                     min_word_count, 
#                     max_word_count, 
#                     min_word_count
#                 )
        
#         # Filter data
#         filtered_df = analysis_df[
#             (analysis_df['category'].isin(selected_categories)) &
#             (analysis_df['word_count'] >= word_filter)
#         ]
        
#         # Display filtered results
#         st.dataframe(filtered_df, use_container_width=True)
        
#         # Visualizations
#         if not filtered_df.empty:
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 # Word count distribution
#                 fig_hist = px.histogram(filtered_df, x='word_count', nbins=20,
#                                       title="Word Count Distribution")
#                 st.plotly_chart(fig_hist, use_container_width=True)
            
#             with col2:
#                 # Readability vs Word count
#                 fig_scatter = px.scatter(filtered_df, x='word_count', y='readability_score',
#                                        color='category', title="Readability vs Document Length")
#                 st.plotly_chart(fig_scatter, use_container_width=True)

#     # NEW: Knowledge Graph in Analytics
#     if "knowledge_graph" in st.session_state:
#         st.markdown("### 🕸️ Knowledge Graph Analysis")
#         knowledge_graph = st.session_state.knowledge_graph
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Graph Nodes", len(knowledge_graph.graph.nodes()))
#         with col2:
#             st.metric("Graph Edges", len(knowledge_graph.graph.edges()))
        
#         # Show the knowledge graph
#         fig = knowledge_graph.generate_interactive_visualization()
#         if fig:
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("Knowledge graph will appear here after uploading documents")

# def main():
#     """
#     Enhanced main application with document intelligence and knowledge graphs.
#     """
#     st.set_page_config(
#         page_title="🧠 InsightForge AI Research Suite", 
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )

#     # Load environment variables
#     load_dotenv()

#     # Enhanced sidebar
#     st.sidebar.title("🧠 InsightForge AI Research Suite")
#     st.sidebar.markdown("---")
    
#     page = st.sidebar.radio(
#         "📍 Navigation", 
#         ["📚 Setup Knowledge Base", "🎤 Voice Assistant", "📊 Analytics Dashboard"],
#         help="Navigate between document setup, voice interaction, and analytics"
#     )

#     # Show system status in sidebar
#     st.sidebar.markdown("### 🔧 System Status")
    
#     # Check Ollama status
#     try:
#         import requests
#         response = requests.get("http://localhost:11434", timeout=2)
#         st.sidebar.success("🟢 Ollama Server: Online")
#     except:
#         st.sidebar.error("🔴 Ollama Server: Offline")
#         st.sidebar.info("Start with: `ollama serve`")

#     if page == "📚 Setup Knowledge Base":
#         setup_knowledge_base()

#     elif page == "📊 Analytics Dashboard":
#         analytics_dashboard()

#     else:  # Voice Assistant - COMPLETE IMPLEMENTATION
#         if "vector_store" not in st.session_state:
#             st.error("❌ Please setup knowledge base first!")
#             st.info("👈 Go to 'Setup Knowledge Base' to upload your documents.")
#             return

#         st.title("🎤 Voice Assistant (Voice Input + Text Response)")
#         st.markdown("*Ask questions using your voice and get intelligent text responses*")

#         # Initialize assistant with better error handling
#         if "assistant" not in st.session_state:
#             try:
#                 with st.spinner("🔧 Initializing Voice Assistant..."):
#                     assistant = VoiceAssistantRAG()
#                     assistant.setup_vector_store(st.session_state.vector_store)
#                     st.session_state.assistant = assistant
#                 st.success("✅ Voice Assistant ready!")
#             except Exception as e:
#                 st.error(f"❌ Error initializing assistant: {e}")
#                 st.info("💡 Make sure Ollama server is running and models are downloaded")
#                 return

#         assistant = st.session_state.assistant

#         # Enhanced sidebar with intelligence stats
#         if "intelligence_results" in st.session_state:
#             st.sidebar.markdown("### 📊 Knowledge Base Intelligence")
#             results = st.session_state.intelligence_results
#             st.sidebar.metric("Documents", results['summary_stats']['total_documents'])
#             st.sidebar.metric("Categories", len(results['categories']))
#             st.sidebar.metric("Avg Words", f"{results['summary_stats']['avg_word_count']:.0f}")

#         # Voice recording settings with enhanced duration
#         st.sidebar.markdown("### 🎤 Voice Settings")
#         duration = st.sidebar.slider("Recording Duration (seconds)", 1, 30, 15)
        
#         # Add recording tips
#         if duration > 20:
#             st.sidebar.info("💡 Longer recordings work better for complex questions")
#         elif duration < 5:
#             st.sidebar.warning("⚠️ Very short recordings might miss parts of your question")

#         # Main interface
#         col1, col2 = st.columns([1, 1])

#         with col1:
#             st.markdown("#### 🎙️ Voice Input")
            
#             if st.button("🎤 Start Recording", type="primary", use_container_width=True):
#                 try:
#                     with st.spinner(f"🎙️ Recording for {duration} seconds..."):
#                         audio_data = assistant.record_audio(duration)
#                         st.session_state.audio_data = audio_data
#                     st.success("✅ Recording completed!")
#                 except Exception as e:
#                     st.error(f"❌ Recording error: {e}")
#                     st.info("💡 Make sure your microphone is connected and working")

#         with col2:
#             st.markdown("#### 📝 Process Voice")
            
#             if st.button("🔄 Convert Voice to Text & Get Answer", type="secondary", use_container_width=True):
#                 if "audio_data" not in st.session_state:
#                     st.error("❌ Please record audio first!")
#                     return

#                 try:
#                     # Step 1: Voice to text
#                     with st.spinner("📝 Converting voice to text..."):
#                         query = assistant.transcribe_audio(st.session_state.audio_data)
                        
#                     if query and query.strip():
#                         st.markdown("### 🗣️ You said:")
#                         st.info(f"'{query}'")
                        
#                         # Step 2: Generate text response with chat history
#                         with st.spinner("🤔 Generating response..."):
#                             # 🔧 FIXED: Pass chat_history parameter
#                             response = assistant.generate_response(
#                                 query, 
#                                 chat_history=st.session_state.get("chat_history", [])
#                             )
                            
#                         st.markdown("### 🤖 AI Response:")
#                         st.markdown(f"**{response}**")
                        
#                         # Store in chat history
#                         if "chat_history" not in st.session_state:
#                             st.session_state.chat_history = []
                        
#                         st.session_state.chat_history.append({
#                             "question": query,
#                             "answer": response,
#                             "type": "voice"
#                         })
                        
#                     else:
#                         st.warning("⚠️ Could not understand the audio. Please try again.")
                        
#                 except Exception as e:
#                     st.error(f"❌ Error processing voice: {e}")

#         # Text input alternative
#         st.markdown("---")
#         st.markdown("#### ⌨️ Or Type Your Question")
        
#         text_query = st.text_input(
#             "Enter your question:", 
#             placeholder="What would you like to know about your documents?",
#             key="text_input"
#         )
        
#         if st.button("📤 Submit Text Question", use_container_width=True) and text_query:
#             try:
#                 with st.spinner("🤔 Generating response..."):
#                     # 🔧 FIXED: Pass chat_history parameter
#                     response = assistant.generate_response(
#                         text_query, 
#                         chat_history=st.session_state.get("chat_history", [])
#                     )
                    
#                 st.markdown("### 🤖 AI Response:")
#                 st.markdown(f"**{response}**")
                
#                 # Store in chat history
#                 if "chat_history" not in st.session_state:
#                     st.session_state.chat_history = []
                
#                 st.session_state.chat_history.append({
#                     "question": text_query,
#                     "answer": response,
#                     "type": "text"
#                 })
                
#             except Exception as e:
#                 st.error(f"❌ Error: {e}")

#         # Enhanced chat history
#         if "chat_history" in st.session_state and st.session_state.chat_history:
#             st.markdown("---")
#             st.markdown("### 📜 Chat History")
            
#             # Add search functionality
#             search_term = st.text_input("🔍 Search chat history:", placeholder="Search questions or answers...")
            
#             filtered_history = st.session_state.chat_history
#             if search_term:
#                 filtered_history = [
#                     chat for chat in st.session_state.chat_history 
#                     if search_term.lower() in chat['question'].lower() or search_term.lower() in chat['answer'].lower()
#                 ]
            
#             for i, chat in enumerate(reversed(filtered_history)):
#                 chat_type_icon = "🎤" if chat.get('type') == 'voice' else "⌨️"
#                 with st.expander(f"{chat_type_icon} Q{len(filtered_history)-i}: {chat['question'][:50]}..."):
#                     st.markdown(f"**❓ Question:** {chat['question']}")
#                     st.markdown(f"**🤖 Answer:** {chat['answer']}")

#         # Enhanced controls
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("🗑️ Clear Chat History", use_container_width=True):
#                 if "chat_history" in st.session_state:
#                     st.session_state.chat_history = []
#                 assistant.clear_conversation_history()
#                 st.rerun()
        
#         with col2:
#             if st.button("💾 Download Chat History", use_container_width=True):
#                 if "chat_history" in st.session_state:
#                     import json
#                     chat_data = json.dumps(st.session_state.chat_history, indent=2)
#                     st.download_button(
#                         "📥 Download JSON",
#                         chat_data,
#                         "chat_history.json",
#                         "application/json"
#                     )

# if __name__ == "__main__":
#     main()

# Import necessary modules
from src.document_processor import DocumentProcessor
from src.voice_assistant_rag import VoiceAssistantRAG
from src.document_intelligence import DocumentIntelligence
from src.knowledge_graph import KnowledgeGraphBuilder
from src.smart_study_buddy import SmartStudyBuddy
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import random
import time
from datetime import datetime

# Fix for MKL libraries issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setup_knowledge_base():
    """
    Enhanced document upload and processing with intelligence and knowledge graphs.
    """
    st.title("📚 Enhanced Knowledge Base Setup")

    doc_processor = DocumentProcessor()
    doc_intelligence = DocumentIntelligence()
    knowledge_graph = KnowledgeGraphBuilder()

    uploaded_files = st.file_uploader(
        "Upload your documents", 
        accept_multiple_files=True, 
        type=["pdf", "txt", "md"],
        help="Upload PDF, TXT, or MD files to create your intelligent knowledge base"
    )

    if uploaded_files and st.button("🔄 Process Documents", type="primary"):
        with st.spinner("Processing documents with AI intelligence..."):
            temp_dir = tempfile.mkdtemp()

            try:
                # 🔧 CRITICAL FIX: Clear old vector store and session state FIRST
                import shutil
                if os.path.exists("knowledge_base"):
                    shutil.rmtree("knowledge_base")
                    st.info("🗑️ Cleared previous knowledge base to ensure fresh processing")

                # Clear related session state
                session_keys_to_clear = ["vector_store", "assistant", "chat_history", "study_questions", "study_notes", "quiz_history"]
                for key in session_keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                        
                st.info("🔄 Starting fresh document processing...")

                # Save uploaded files
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                # Process documents
                documents = doc_processor.load_documents(temp_dir)
                processed_docs = doc_processor.process_documents(documents)

                # NEW: Analyze document intelligence
                with st.spinner("🧠 Analyzing document intelligence..."):
                    intelligence_results = doc_intelligence.categorize_documents(documents)
                    st.session_state.intelligence_results = intelligence_results

                # NEW: Build knowledge graph
                with st.spinner("🕸️ Building knowledge graph..."):
                    graph = knowledge_graph.build_document_relationships(documents)
                    st.session_state.knowledge_graph = knowledge_graph

                # Create fresh vector store
                vector_store = doc_processor.create_vector_store(
                    processed_docs, "knowledge_base"
                )

                # Store in session state
                st.session_state.vector_store = vector_store
                st.session_state.documents = documents
                st.session_state.processed_docs = processed_docs

                st.success(f"✅ Successfully processed {len(documents)} documents into {len(processed_docs)} chunks!")
                
                # Enhanced stats display
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Documents", len(documents))
                with col2:
                    st.metric("Text Chunks", len(processed_docs))
                with col3:
                    st.metric("Categories", len(intelligence_results['categories']))
                with col4:
                    st.metric("Graph Nodes", len(graph.nodes()))

                # NEW: Display document analysis
                st.markdown("### 📊 Document Intelligence Analysis")
                
                # Category distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📂 Document Categories")
                    categories_df = pd.DataFrame(
                        list(intelligence_results['categories'].items()),
                        columns=['Category', 'Count']
                    )
                    if not categories_df.empty:
                        fig_pie = px.pie(categories_df, values='Count', names='Category', 
                                       title="Document Distribution by Category")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info("No categories detected")
                
                with col2:
                    st.markdown("#### 📈 Document Metrics")
                    metrics_data = intelligence_results['summary_stats']
                    
                    st.metric("Average Words per Document", 
                             f"{metrics_data['avg_word_count']:.0f}")
                    st.metric("Most Common Category", 
                             metrics_data['most_common_category'].title())

                # NEW: Document analysis table
                st.markdown("#### 📋 Individual Document Analysis")
                analysis_df = pd.DataFrame(intelligence_results['document_analysis'])
                if not analysis_df.empty:
                    analysis_df['filename'] = analysis_df['filename'].apply(lambda x: os.path.basename(x))
                    
                    st.dataframe(
                        analysis_df[['filename', 'category', 'word_count', 'readability_score', 'key_topics']],
                        use_container_width=True
                    )

                # NEW: Knowledge Graph Visualization
                st.markdown("### 🕸️ Document Relationship Graph")
                fig = knowledge_graph.generate_interactive_visualization()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("💡 Hover over nodes to see document details. Connected documents share similar content.")
                else:
                    st.warning("No significant relationships found between documents. Try uploading more related documents.")

            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
                st.info("💡 Make sure Ollama server is running with required models")

            finally:
                # Cleanup
                try:
                    for file in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, file))
                    os.rmdir(temp_dir)
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temp directory: {cleanup_error}")

def analytics_dashboard():
    """
    Analytics dashboard for document insights with intelligence features.
    """
    if "intelligence_results" not in st.session_state:
        st.error("❌ Please setup knowledge base first!")
        st.info("👈 Go to 'Setup Knowledge Base' to upload and analyze your documents.")
        return
        
    st.title("📊 Document Analytics Dashboard")
    
    intelligence_results = st.session_state.intelligence_results
    
    # Enhanced analytics display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", intelligence_results['summary_stats']['total_documents'])
    with col2:
        st.metric("Average Words", f"{intelligence_results['summary_stats']['avg_word_count']:.0f}")
    with col3:
        st.metric("Categories Detected", len(intelligence_results['categories']))
    
    # Detailed analysis
    st.markdown("### 📈 Detailed Document Analysis")
    
    analysis_df = pd.DataFrame(intelligence_results['document_analysis'])
    
    if not analysis_df.empty:
        # Interactive filtering
        col1, col2 = st.columns(2)
        with col1:
            selected_categories = st.multiselect(
                "Filter by Category",
                options=analysis_df['category'].unique(),
                default=analysis_df['category'].unique()
            )
        
        with col2:
            # FIXED: Safe word count filter
            min_word_count = int(analysis_df['word_count'].min())
            max_word_count = int(analysis_df['word_count'].max())
            
            if min_word_count == max_word_count:
                st.info(f"All documents have {min_word_count} words")
                word_filter = min_word_count
            else:
                word_filter = st.slider(
                    "Minimum Word Count", 
                    min_word_count, 
                    max_word_count, 
                    min_word_count
                )
        
        # Filter data
        filtered_df = analysis_df[
            (analysis_df['category'].isin(selected_categories)) &
            (analysis_df['word_count'] >= word_filter)
        ]
        
        # Display filtered results
        st.dataframe(filtered_df, use_container_width=True)
        
        # Visualizations
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Word count distribution
                fig_hist = px.histogram(filtered_df, x='word_count', nbins=20,
                                      title="Word Count Distribution")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Readability vs Word count
                fig_scatter = px.scatter(filtered_df, x='word_count', y='readability_score',
                                       color='category', title="Readability vs Document Length")
                st.plotly_chart(fig_scatter, use_container_width=True)

    # NEW: Knowledge Graph in Analytics
    if "knowledge_graph" in st.session_state:
        st.markdown("### 🕸️ Knowledge Graph Analysis")
        knowledge_graph = st.session_state.knowledge_graph
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Graph Nodes", len(knowledge_graph.graph.nodes()))
        with col2:
            st.metric("Graph Edges", len(knowledge_graph.graph.edges()))
        
        # Show the knowledge graph
        fig = knowledge_graph.generate_interactive_visualization()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Knowledge graph will appear here after uploading documents")

def smart_study_buddy_section():
    """Smart Study Buddy main section"""
    if "vector_store" not in st.session_state:
        st.error("❌ Please setup knowledge base first!")
        st.info("👈 Go to 'Setup Knowledge Base' to upload your documents.")
        return
        
    st.title("🧠 Smart Study Buddy")
    st.markdown("*Your AI-powered learning companion that creates questions and tests your knowledge*")
    
    # Initialize Smart Study Buddy
    if "study_buddy" not in st.session_state:
        st.session_state.study_buddy = SmartStudyBuddy()
    
    # Sub-navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 Dashboard", 
        "❓ Generate Questions", 
        "📝 Study Notes", 
        "🎯 Quiz Mode"
    ])
    
    with tab1:
        study_dashboard()
        
    with tab2:
        question_generator()
        
    with tab3:
        study_notes_section()
        
    with tab4:
        quiz_mode()

def study_dashboard():
    """Study overview dashboard"""
    st.markdown("### 📊 Your Study Overview")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        questions_count = len(st.session_state.get("study_questions", []))
        st.metric("Generated Questions", questions_count)
        
    with col2:
        notes_count = len(st.session_state.get("study_notes", []))
        st.metric("Study Notes", notes_count)
        
    with col3:
        quiz_sessions = len(st.session_state.get("quiz_history", []))
        st.metric("Quiz Sessions", quiz_sessions)
        
    with col4:
        if "quiz_history" in st.session_state and st.session_state.quiz_history:
            avg_score = sum(q["score"] for q in st.session_state.quiz_history) / len(st.session_state.quiz_history)
            st.metric("Average Score", f"{avg_score:.1f}%")
        else:
            st.metric("Average Score", "No data")
    
    # Show available documents
    if "documents" in st.session_state:
        st.markdown("### 📚 Available Documents")
        for i, doc in enumerate(st.session_state.documents):
            doc_name = os.path.basename(doc.metadata.get('source', f'Document {i}'))
            st.write(f"📄 {doc_name}")
    
    # Recent quiz sessions
    if "quiz_history" in st.session_state and st.session_state.quiz_history:
        st.markdown("### 📈 Recent Quiz Sessions")
        for session in reversed(st.session_state.quiz_history[-3:]):
            with st.expander(f"Quiz on {session['date']} - Score: {session['score']:.1f}%"):
                st.write(f"**Questions Answered:** {session['total_questions']}")
                st.write(f"**Correct Answers:** {session['correct_answers']}")
                st.write(f"**Topics:** {', '.join(session['topics'])}")
                st.write(f"**Duration:** {session['duration']:.0f} seconds")

def question_generator():
    """Generate study questions interface"""
    st.markdown("### ❓ AI Question Generator")
    
    if "documents" not in st.session_state:
        st.warning("No documents found. Please upload documents first.")
        return
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        num_questions = st.slider("Number of questions", 1, 15, 5)
    with col2:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    
    if st.button("🔄 Generate Questions", type="primary"):
        with st.spinner("🧠 AI is generating questions..."):
            try:
                questions = st.session_state.study_buddy.generate_study_questions(
                    st.session_state.documents, num_questions, difficulty
                )
                
                if questions:
                    # Store questions
                    if "study_questions" not in st.session_state:
                        st.session_state.study_questions = []
                    st.session_state.study_questions.extend(questions)
                    
                    st.success(f"✅ Generated {len(questions)} new questions!")
                    
                    # Preview questions
                    st.markdown("### 📝 Generated Questions Preview")
                    for i, q in enumerate(questions[:3]):
                        with st.expander(f"Question {i+1}: {q['question'][:50]}..."):
                            st.write(f"**Q:** {q['question']}")
                            for j, option in enumerate(q['options']):
                                marker = "✅" if j == q['correct'] else "⭕"
                                st.write(f"{marker} {chr(65+j)}: {option}")
                            st.write(f"**Topic:** {q['topic']}")
                            st.write(f"**Difficulty:** {q['difficulty']}")
                else:
                    st.warning("No questions generated. Try again.")
                    
            except Exception as e:
                st.error(f"Error generating questions: {str(e)}")
    
    # Show existing questions
    if "study_questions" in st.session_state and st.session_state.study_questions:
        st.markdown("### 📚 Your Question Bank")
        st.info(f"You have {len(st.session_state.study_questions)} questions ready for quizzes!")
        
        if st.button("🗑️ Clear All Questions"):
            del st.session_state.study_questions
            st.rerun()

def study_notes_section():
    """Study notes interface"""
    st.markdown("### 📝 Study Notes Generator")
    
    if "documents" not in st.session_state:
        st.warning("No documents found. Please upload documents first.")
        return
    
    if st.button("📝 Generate Study Notes", type="primary"):
        with st.spinner("📚 Creating study notes..."):
            try:
                notes = st.session_state.study_buddy.create_study_notes(
                    st.session_state.documents
                )
                
                if notes:
                    st.session_state.study_notes = notes
                    st.success(f"✅ Generated notes for {len(notes)} documents!")
                    
                    # Display notes
                    for note in notes:
                        with st.expander(f"📄 {note['document']} - {note['topic']}"):
                            st.markdown(f"**Summary:** {note['summary']}")
                            st.markdown("**Key Points:**")
                            for point in note['key_points']:
                                st.write(f"• {point}")
                            
                            st.markdown("**Flashcards:**")
                            for card in note['flashcards']:
                                st.info(f"**Q:** {card['front']}\n**A:** {card['back']}")
                else:
                    st.warning("No notes generated. Try again.")
                    
            except Exception as e:
                st.error(f"Error generating notes: {str(e)}")
    
    # Show existing notes
    if "study_notes" in st.session_state and st.session_state.study_notes:
        st.markdown("### 📚 Your Study Notes")
        for note in st.session_state.study_notes:
            with st.expander(f"📄 {note['document']} - {note['topic']}"):
                st.markdown(f"**Summary:** {note['summary']}")
                st.markdown("**Key Points:**")
                for point in note['key_points']:
                    st.write(f"• {point}")

def quiz_mode():
    """Interactive quiz interface"""
    st.markdown("### 🎯 Quiz Mode")
    
    if "study_questions" not in st.session_state or not st.session_state.study_questions:
        st.warning("⚠️ No questions available! Please generate questions first.")
        st.info("👈 Go to 'Generate Questions' tab to create quiz questions.")
        return
    
    # Quiz settings
    col1, col2 = st.columns(2)
    with col1:
        available_questions = len(st.session_state.study_questions)
        num_quiz_questions = st.slider(
            "Questions in quiz", 
            1, 
            min(10, available_questions), 
            min(5, available_questions)
        )
    
    with col2:
        st.metric("Available Questions", available_questions)
    
    # Start quiz
    if st.button("🚀 Start Quiz", type="primary"):
        questions = random.sample(st.session_state.study_questions, num_quiz_questions)
        st.session_state.current_quiz = {
            "questions": questions,
            "current_question": 0,
            "answers": [],
            "score": 0,
            "start_time": time.time()
        }
        st.session_state.quiz_active = True
        st.rerun()
    
    # Active quiz
    if st.session_state.get("quiz_active", False):
        quiz = st.session_state.current_quiz
        current_idx = quiz["current_question"]
        
        if current_idx < len(quiz["questions"]):
            question = quiz["questions"][current_idx]
            
            st.markdown(f"### Question {current_idx + 1} of {len(quiz['questions'])}")
            st.markdown(f"**Topic:** {question['topic']}")
            st.markdown(f"**{question['question']}**")
            
            # Answer options
            user_answer = st.radio(
                "Choose your answer:",
                options=range(len(question["options"])),
                format_func=lambda x: f"{chr(65+x)}: {question['options'][x]}",
                key=f"quiz_q_{current_idx}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Submit Answer", type="primary"):
                    is_correct = user_answer == question["correct"]
                    
                    quiz["answers"].append({
                        "question_id": question["id"],
                        "user_answer": user_answer,
                        "correct_answer": question["correct"],
                        "is_correct": is_correct
                    })
                    
                    if is_correct:
                        quiz["score"] += 1
                        st.success("✅ Correct! Well done!")
                    else:
                        correct_option = question["options"][question["correct"]]
                        st.error(f"❌ Wrong! Correct answer: {chr(65+question['correct'])}: {correct_option}")
                    
                    quiz["current_question"] += 1
                    time.sleep(2)
                    st.rerun()
            
            with col2:
                if st.button("⏭️ Skip Question"):
                    quiz["current_question"] += 1
                    st.rerun()
        
        else:
            # Quiz completed
            final_score = (quiz["score"] / len(quiz["questions"])) * 100
            duration = time.time() - quiz["start_time"]
            
            st.markdown("## 🎉 Quiz Completed!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Your Score", f"{final_score:.1f}%")
            with col2:
                st.metric("Correct Answers", f"{quiz['score']}/{len(quiz['questions'])}")
            with col3:
                st.metric("Time Taken", f"{duration:.0f} seconds")
            
            # Performance feedback
            if final_score >= 80:
                st.success("🌟 Excellent performance! You've mastered this material!")
            elif final_score >= 60:
                st.info("👍 Good job! A bit more practice will make you perfect!")
            else:
                st.warning("📚 Keep studying! Review the notes and try again.")
            
            # Save to history
            quiz_session = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "score": final_score,
                "total_questions": len(quiz["questions"]),
                "correct_answers": quiz["score"],
                "topics": list(set(q["topic"] for q in quiz["questions"])),
                "duration": duration
            }
            
            if "quiz_history" not in st.session_state:
                st.session_state.quiz_history = []
            st.session_state.quiz_history.append(quiz_session)
            
            # Reset quiz
            del st.session_state.current_quiz
            del st.session_state.quiz_active
            
            if st.button("🔄 Take Another Quiz"):
                st.rerun()

def main():
    """
    Enhanced main application with document intelligence, knowledge graphs, and Smart Study Buddy.
    """
    st.set_page_config(
        page_title="🧠 InsightForge AI Research Suite", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load environment variables
    load_dotenv()

    # Enhanced sidebar
    st.sidebar.title("🧠 InsightForge AI Research Suite")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "📍 Navigation", 
        [
            "📚 Setup Knowledge Base", 
            "🎤 Voice Assistant", 
            "📊 Analytics Dashboard",
            "🧠 Smart Study Buddy"
        ],
        help="Navigate between document setup, voice interaction, analytics, and smart learning"
    )

    # Show system status in sidebar
    st.sidebar.markdown("### 🔧 System Status")
    
    # Check Ollama status
    try:
        import requests
        response = requests.get("http://localhost:11434", timeout=2)
        st.sidebar.success("🟢 Ollama Server: Online")
    except:
        st.sidebar.error("🔴 Ollama Server: Offline")
        st.sidebar.info("Start with: `ollama serve`")

    if page == "📚 Setup Knowledge Base":
        setup_knowledge_base()

    elif page == "📊 Analytics Dashboard":
        analytics_dashboard()

    elif page == "🧠 Smart Study Buddy":
        smart_study_buddy_section()

    else:  # Voice Assistant
        if "vector_store" not in st.session_state:
            st.error("❌ Please setup knowledge base first!")
            st.info("👈 Go to 'Setup Knowledge Base' to upload your documents.")
            return

        st.title("🎤 Voice Assistant (Voice Input + Text Response)")
        st.markdown("*Ask questions using your voice and get intelligent text responses*")

        # Initialize assistant with better error handling
        if "assistant" not in st.session_state:
            try:
                with st.spinner("🔧 Initializing Voice Assistant..."):
                    assistant = VoiceAssistantRAG()
                    assistant.setup_vector_store(st.session_state.vector_store)
                    st.session_state.assistant = assistant
                st.success("✅ Voice Assistant ready!")
            except Exception as e:
                st.error(f"❌ Error initializing assistant: {e}")
                st.info("💡 Make sure Ollama server is running and models are downloaded")
                return

        assistant = st.session_state.assistant

        # Enhanced sidebar with intelligence stats
        if "intelligence_results" in st.session_state:
            st.sidebar.markdown("### 📊 Knowledge Base Intelligence")
            results = st.session_state.intelligence_results
            st.sidebar.metric("Documents", results['summary_stats']['total_documents'])
            st.sidebar.metric("Categories", len(results['categories']))
            st.sidebar.metric("Avg Words", f"{results['summary_stats']['avg_word_count']:.0f}")

        # Voice recording settings with enhanced duration
        st.sidebar.markdown("### 🎤 Voice Settings")
        duration = st.sidebar.slider("Recording Duration (seconds)", 1, 30, 15)
        
        # Add recording tips
        if duration > 20:
            st.sidebar.info("💡 Longer recordings work better for complex questions")
        elif duration < 5:
            st.sidebar.warning("⚠️ Very short recordings might miss parts of your question")

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
                    st.info("💡 Make sure your microphone is connected and working")

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
                        
                        # Step 2: Generate text response with chat history
                        with st.spinner("🤔 Generating response..."):
                            response = assistant.generate_response(
                                query, 
                                chat_history=st.session_state.get("chat_history", [])
                            )
                            
                        st.markdown("### 🤖 AI Response:")
                        st.markdown(f"**{response}**")
                        
                        # Store in chat history
                        if "chat_history" not in st.session_state:
                            st.session_state.chat_history = []
                        
                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": response,
                            "type": "voice"
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
                    response = assistant.generate_response(
                        text_query, 
                        chat_history=st.session_state.get("chat_history", [])
                    )
                    
                st.markdown("### 🤖 AI Response:")
                st.markdown(f"**{response}**")
                
                # Store in chat history
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append({
                    "question": text_query,
                    "answer": response,
                    "type": "text"
                })
                
            except Exception as e:
                st.error(f"❌ Error: {e}")

        # Enhanced chat history
        if "chat_history" in st.session_state and st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 📜 Chat History")
            
            # Add search functionality
            search_term = st.text_input("🔍 Search chat history:", placeholder="Search questions or answers...")
            
            filtered_history = st.session_state.chat_history
            if search_term:
                filtered_history = [
                    chat for chat in st.session_state.chat_history 
                    if search_term.lower() in chat['question'].lower() or search_term.lower() in chat['answer'].lower()
                ]
            
            for i, chat in enumerate(reversed(filtered_history)):
                chat_type_icon = "🎤" if chat.get('type') == 'voice' else "⌨️"
                with st.expander(f"{chat_type_icon} Q{len(filtered_history)-i}: {chat['question'][:50]}..."):
                    st.markdown(f"**❓ Question:** {chat['question']}")
                    st.markdown(f"**🤖 Answer:** {chat['answer']}")

        # Enhanced controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                if "chat_history" in st.session_state:
                    st.session_state.chat_history = []
                assistant.clear_conversation_history()
                st.rerun()
        
        with col2:
            if st.button("💾 Download Chat History", use_container_width=True):
                if "chat_history" in st.session_state:
                    import json
                    chat_data = json.dumps(st.session_state.chat_history, indent=2)
                    st.download_button(
                        "📥 Download JSON",
                        chat_data,
                        "chat_history.json",
                        "application/json"
                    )

if __name__ == "__main__":
    main()
