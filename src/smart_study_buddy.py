import streamlit as st
import random
import time
from datetime import datetime
import os

class SmartStudyBuddy:
    """Smart Study Buddy - AI-powered learning companion"""
    
    def __init__(self):
        self.question_types = ["multiple_choice", "true_false", "short_answer"]
        self.difficulties = ["Easy", "Medium", "Hard"]
    
    def generate_study_questions(self, documents, num_questions=5, difficulty="Medium"):
        """Generate study questions from documents using Llama"""
        questions = []
        
        for i in range(num_questions):
            if not documents:
                continue
                
            # Select random document
            doc = documents[i % len(documents)]
            doc_content = doc.page_content[:1500]  # First 1500 chars
            
            # Create question generation prompt
            prompt = f"""
            Based on this document content, create a study question:
            
            Content: {doc_content}
            
            Generate a {difficulty.lower()} difficulty multiple choice question with 4 options.
            Format your response exactly as:
            QUESTION: [your question here]
            A: [option A]
            B: [option B] 
            C: [option C]
            D: [option D]
            CORRECT: [A/B/C/D]
            TOPIC: [main topic from content]
            """
            
            try:
                # Use existing assistant if available
                if "assistant" in st.session_state:
                    response = st.session_state.assistant.generate_response(
                        prompt, chat_history=[]
                    )
                    question = self.parse_question_response(response, i+1)
                    if question:
                        questions.append(question)
                else:
                    # Fallback sample question for testing
                    sample_question = {
                        "id": i+1,
                        "question": f"Sample question {i+1} from document content",
                        "options": ["Option A", "Option B", "Option C", "Option D"],
                        "correct": 0,
                        "type": "multiple_choice",
                        "difficulty": difficulty,
                        "topic": "Sample Topic",
                        "document": os.path.basename(doc.metadata.get('source', f'Document {i}'))
                    }
                    questions.append(sample_question)
                    
            except Exception as e:
                st.error(f"Error generating question {i+1}: {str(e)}")
                continue
        
        return questions
    
    def parse_question_response(self, response, question_id):
        """Parse AI response into structured question format"""
        try:
            lines = response.strip().split('\n')
            question = ""
            options = []
            correct = ""
            topic = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("QUESTION:"):
                    question = line.replace("QUESTION:", "").strip()
                elif line.startswith(("A:", "B:", "C:", "D:")):
                    options.append(line[2:].strip())
                elif line.startswith("CORRECT:"):
                    correct = line.replace("CORRECT:", "").strip()
                elif line.startswith("TOPIC:"):
                    topic = line.replace("TOPIC:", "").strip()
            
            if question and len(options) == 4 and correct:
                correct_index = ord(correct.upper()) - ord('A') if correct.upper() in 'ABCD' else 0
                
                return {
                    "id": question_id,
                    "question": question,
                    "options": options,
                    "correct": correct_index,
                    "type": "multiple_choice",
                    "difficulty": "Medium",
                    "topic": topic or "General",
                    "document": "Document"
                }
        except Exception as e:
            print(f"Error parsing question: {e}")
            return None
        
        return None
    
    def create_study_notes(self, documents):
        """Generate detailed study notes with summary, key topics, and flashcards"""
        notes = []
        
        for i, doc in enumerate(documents[:3]):  # Limit to first 3 documents
            doc_content = doc.page_content[:2500]  # Increased content length
            doc_name = os.path.basename(doc.metadata.get('source', f'Document {i}'))
            
            # Create comprehensive prompt for study notes
            prompt = f"""
            Create comprehensive study notes from the following content. 
            
            Content:
            {doc_content}
            
            Please provide:
            1. SUMMARY: Write a 2-3 sentence summary of the main ideas
            2. KEY_TOPICS: List 5 important topics/concepts (separate with |)
            3. FLASHCARDS: Create 3 question-answer pairs for memorization
            
            Format your response exactly as:
            SUMMARY: [your summary here]
            KEY_TOPICS: Topic1 | Topic2 | Topic3 | Topic4 | Topic5
            FLASHCARDS:
            Q1: [question 1]
            A1: [answer 1]
            Q2: [question 2]
            A2: [answer 2]
            Q3: [question 3]
            A3: [answer 3]
            MAIN_TOPIC: [primary topic of document]
            """
            
            try:
                if "assistant" in st.session_state:
                    response = st.session_state.assistant.generate_response(
                        prompt, chat_history=[]
                    )
                    note = self.parse_notes_response(response, doc_name)
                    if note:
                        notes.append(note)
                    else:
                        # If parsing fails, create a basic note with actual content
                        notes.append(self.create_fallback_note(doc_content, doc_name))
                else:
                    # Create fallback note when assistant not available
                    notes.append(self.create_fallback_note(doc_content, doc_name))
                    
            except Exception as e:
                st.error(f"Error creating notes for {doc_name}: {str(e)}")
                # Create fallback note on error
                notes.append(self.create_fallback_note(doc_content, doc_name))
        
        return notes
    
    def create_fallback_note(self, doc_content, doc_name):
        """Create a fallback note when AI generation fails"""
        # Extract first few sentences for summary
        sentences = doc_content.split('.')[:3]
        summary = '. '.join(sentences) + '.' if sentences else "Content from uploaded document."
        
        # Extract some words as key topics
        words = doc_content.split()[:50]  # First 50 words
        key_topics = [word.capitalize() for word in words[::10] if len(word) > 3][:5]  # Every 10th word
        if not key_topics:
            key_topics = ["Document Content", "Key Concepts", "Main Ideas", "Important Points", "Study Material"]
        
        return {
            "document": doc_name,
            "topic": "General Study Material",
            "summary": summary[:200] + "..." if len(summary) > 200 else summary,
            "key_points": key_topics,
            "flashcards": [
                {"front": f"What is the main topic of {doc_name}?", "back": "The main concepts covered in this document"},
                {"front": "What should I focus on when studying this material?", "back": "Key points and important concepts"},
                {"front": f"How can I summarize {doc_name}?", "back": summary[:100] + "..."}
            ]
        }
    
    def parse_notes_response(self, response, doc_name):
        """Parse AI response into structured notes format"""
        try:
            lines = response.strip().split('\n')
            summary = ""
            key_points = []
            topic = ""
            flashcards = []
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if line.startswith("SUMMARY:"):
                    summary = line.replace("SUMMARY:", "").strip()
                
                elif line.startswith("KEY_TOPICS:"):
                    topics_text = line.replace("KEY_TOPICS:", "").strip()
                    key_points = [topic.strip() for topic in topics_text.split('|') if topic.strip()]
                
                elif line.startswith("MAIN_TOPIC:"):
                    topic = line.replace("MAIN_TOPIC:", "").strip()
                
                elif line.startswith("FLASHCARDS:"):
                    # Parse flashcards
                    i += 1
                    current_question = ""
                    current_answer = ""
                    
                    while i < len(lines):
                        line = lines[i].strip()
                        if not line:
                            i += 1
                            continue
                            
                        if line.startswith("Q"):
                            current_question = line.split(":", 1)[1].strip() if ":" in line else line
                        elif line.startswith("A"):
                            current_answer = line.split(":", 1)[1].strip() if ":" in line else line
                            if current_question and current_answer:
                                flashcards.append({
                                    "front": current_question,
                                    "back": current_answer
                                })
                                current_question = ""
                                current_answer = ""
                        i += 1
                    break
                
                i += 1
            
            # Validate parsed data and provide defaults
            if not summary:
                summary = "Study material covering important concepts and topics."
            
            if not key_points:
                key_points = ["Key Concept 1", "Key Concept 2", "Key Concept 3", "Important Topics", "Study Focus"]
            
            if not flashcards:
                flashcards = [
                    {"front": f"What is covered in {doc_name}?", "back": "Important concepts and study material"},
                    {"front": "What should I remember from this document?", "back": "Key points and main ideas"},
                    {"front": "How should I approach studying this?", "back": "Focus on main concepts and practice questions"}
                ]
            
            return {
                "document": doc_name,
                "topic": topic or "Study Material",
                "summary": summary,
                "key_points": key_points,
                "flashcards": flashcards
            }
            
        except Exception as e:
            print(f"Error parsing notes response: {e}")
            return None
