import tempfile
import os
import soundfile as sf
import sounddevice as sd
# Use faster-whisper for voice-to-text only
from faster_whisper import WhisperModel

# Updated LangChain imports (memory removed)
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama, OllamaEmbeddings

class VoiceAssistantRAG:
    """
    VoiceAssistantRAG with voice input (Whisper) and text-only responses.
    Voice generation (TTS) disabled for now.
    """

    def __init__(self, elevenlabs_api_key=None):
        """
        Initialize with Whisper for voice input only.
        TTS functionality disabled.
        """
        try:
            print("🎤 Initializing Whisper model for voice input...")
            # Whisper for speech-to-text (voice input)
            self.whisper_model = WhisperModel(
                "base",
                device="cpu",
                compute_type="int8"
            )
            print("✅ Whisper model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            raise

        try:
            print("🤖 Initializing Ollama LLM...")
            self.llm = ChatOllama(model="llama3.2", temperature=0)
            print("✅ Ollama LLM initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Ollama LLM: {e}")
            raise

        try:
            print("🔢 Initializing Ollama embeddings...")
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://localhost:11434"
            )
            print("✅ Ollama embeddings initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Ollama embeddings: {e}")
            raise

        self.vector_store = None
        self.qa_chain = None
        self.sample_rate = 44100
        
        # Voice generation disabled for now
        print("⚠️ Voice generation (TTS) disabled - text responses only")
        self.voice_generator = None

        print("🎉 VoiceAssistantRAG initialization complete (Voice Input + Text Output)")

    def setup_vector_store(self, vector_store):
        """
        Initialize vector store without deprecated memory configuration.
        """
        try:
            print("🔧 Setting up vector store for RAG chain...")
            
            if vector_store is None:
                raise ValueError("Vector store cannot be None")

            print("🧪 Testing vector store health...")
            test_results = vector_store.similarity_search("test", k=1)
            print(f"✅ Vector store health check passed (found {len(test_results)} results)")

            self.vector_store = vector_store

            # 🔧 FIXED: Remove deprecated memory parameter
            print("⛓️ Creating conversational retrieval chain...")
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                verbose=True,
                return_source_documents=True
                # ❌ REMOVED: memory=memory (deprecated in LangChain)
                # ❌ REMOVED: output_key="answer" (not needed without memory)
            )

            print("✅ Vector store and QA chain setup complete")

        except Exception as e:
            print(f"❌ Error setting up vector store: {e}")
            raise

    def record_audio(self, duration=5):
        """
        Record audio for voice input.
        """
        try:
            print(f"🎙️ Recording audio for {duration} seconds...")
            
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float64'
            )
            sd.wait()
            
            print("✅ Audio recording completed")
            return recording
            
        except Exception as e:
            print(f"❌ Error during audio recording: {e}")
            raise

    def transcribe_audio(self, audio_array):
        """
        Convert voice to text using Whisper.
        """
        temp_path = None
        try:
            print("📝 Converting voice to text...")
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_path = temp_audio.name
            
            sf.write(temp_path, audio_array, self.sample_rate)
            
            print("🔄 Transcribing audio with Whisper...")
            segments, info = self.whisper_model.transcribe(
                temp_path, 
                beam_size=5,
                language="en"
            )
            
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            
            print(f"✅ Voice-to-text completed: '{transcribed_text[:50]}...'")
            print(f"📊 Language: {info.language}, Confidence: {info.language_probability:.2f}")
            
            return transcribed_text

        except Exception as e:
            print(f"❌ Error during voice-to-text conversion: {e}")
            return "Sorry, I couldn't understand the audio."
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as cleanup_error:
                    print(f"⚠️ Warning: Failed to delete temp audio file: {cleanup_error}")

    def generate_response(self, query, chat_history=None):
        """
        🔧 FIXED: Generate text response using RAG pipeline with chat_history parameter.
        """
        try:
            print(f"🤔 Processing query: '{query[:50]}...'")
            
            if self.qa_chain is None:
                return "Error: Vector store not initialized. Please setup knowledge base first."

            if not query or not query.strip():
                return "Please provide a valid question."

            # 🔧 FIX: Ensure chat_history is provided to avoid "Missing input keys" error
            if chat_history is None:
                chat_history = []

            # Convert Streamlit chat history format to LangChain format if needed
            langchain_history = []
            if chat_history:
                for chat in chat_history:
                    if isinstance(chat, dict) and "question" in chat and "answer" in chat:
                        # Convert to (human_message, ai_message) tuple format
                        langchain_history.append((chat["question"], chat["answer"]))

            print("🔄 Generating response with RAG chain...")
            response = self.qa_chain.invoke({
                "question": query,
                "chat_history": langchain_history  # 🔧 FIXED: Always include chat_history
            })
            
            answer = response.get("answer", "I couldn't generate a proper response.")
            sources = response.get("source_documents", [])
            
            print(f"✅ Text response generated successfully with {len(sources)} source documents")
            
            return answer

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ {error_msg}")
            return f"I apologize, but I encountered an error: {str(e)}"

    # Voice generation methods disabled
    def text_to_speech(self, text: str, voice_name: str = None) -> str:
        """
        Voice generation disabled - returns None.
        """
        print("📝 Voice generation disabled - displaying text response only")
        return None

    def get_conversation_history(self):
        """Get conversation history - now managed by session state."""
        try:
            # Since we removed LangChain memory, conversation history is now 
            # managed by Streamlit session state in main.py
            print("📜 Conversation history managed by session state")
            return []
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []

    def clear_conversation_history(self):
        """Clear conversation history - now managed by session state."""
        try:
            # Since we removed LangChain memory, this is now handled in main.py
            print("🗑️ Conversation history clearing handled by session state")
        except Exception as e:
            print(f"Error clearing conversation history: {e}")