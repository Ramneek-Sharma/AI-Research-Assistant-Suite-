import os
from typing import List, Dict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from langchain_core.documents import Document

class DocumentIntelligence:
    """
    Advanced document analysis with categorization, tagging, and insights.
    """
    
    def __init__(self):
        self.categories = {
            'research': ['research', 'study', 'analysis', 'methodology', 'experiment', 'paper', 'model', 'transformer'],
            'business': ['revenue', 'profit', 'market', 'strategy', 'financial', 'business', 'company'],
            'technical': ['algorithm', 'implementation', 'code', 'system', 'architecture', 'programming'],
            'legal': ['contract', 'agreement', 'compliance', 'regulation', 'law', 'legal'],
            'medical': ['patient', 'treatment', 'diagnosis', 'clinical', 'medical', 'health']
        }
        
    def categorize_documents(self, documents: List[Document]) -> Dict:
        """
        Automatically categorize documents by content.
        """
        results = {
            'categories': {},
            'document_analysis': [],
            'summary_stats': {}
        }
        
        for i, doc in enumerate(documents):
            content = doc.page_content.lower()
            scores = {}
            
            # Calculate category scores
            for category, keywords in self.categories.items():
                score = sum(content.count(keyword) for keyword in keywords)
                scores[category] = score
            
            # Determine primary category
            primary_category = max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
            
            # Extract metadata
            word_count = len(content.split())
            readability = self._calculate_readability(content)
            
            doc_analysis = {
                'doc_index': i,
                'filename': doc.metadata.get('source', f'document_{i}'),
                'category': primary_category,
                'confidence': scores[primary_category] / max(1, word_count) * 1000,
                'word_count': word_count,
                'readability_score': readability,
                'key_topics': self._extract_key_topics(content)
            }
            
            results['document_analysis'].append(doc_analysis)
            
            # Update category counts
            if primary_category not in results['categories']:
                results['categories'][primary_category] = 0
            results['categories'][primary_category] += 1
        
        # Calculate summary statistics
        results['summary_stats'] = {
            'total_documents': len(documents),
            'avg_word_count': np.mean([doc['word_count'] for doc in results['document_analysis']]),
            'most_common_category': max(results['categories'], key=results['categories'].get) if results['categories'] else 'unknown',
            'category_distribution': results['categories']
        }
        
        return results
    
    def _calculate_readability(self, text: str) -> float:
        """Simple readability score."""
        sentences = text.split('.')
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0
            
        avg_sentence_length = len(words) / len(sentences)
        score = max(0, min(100, 206.835 - (1.015 * avg_sentence_length)))
        return round(score, 2)
    
    def _extract_key_topics(self, text: str, max_topics: int = 3) -> List[str]:
        """Extract key topics using frequency analysis."""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those'}
        
        words = [word.lower().strip('.,!?;:"()[]') for word in text.split()]
        words = [word for word in words if len(word) > 3 and word not in stop_words]
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_topics]
        return [word for word, freq in top_words if freq > 1]
