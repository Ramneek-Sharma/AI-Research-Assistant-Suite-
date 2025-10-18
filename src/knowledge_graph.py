import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_core.documents import Document
import os

class KnowledgeGraphBuilder:
    """
    Build and visualize knowledge graphs from document collections.
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.document_similarities = None
        
    def build_document_relationships(self, documents: List[Document], similarity_threshold: float = 0.3):
        """
        Build relationships between documents based on content similarity.
        """
        # Extract text content
        texts = [doc.page_content for doc in documents]
        filenames = [doc.metadata.get('source', f'doc_{i}') for i, doc in enumerate(documents)]
        
        # Calculate TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        self.document_similarities = similarity_matrix
        
        # Build graph
        self.graph.clear()
        
        # Add nodes (documents)
        for i, filename in enumerate(filenames):
            node_name = os.path.basename(filename)
            self.graph.add_node(
                node_name,
                doc_index=i,
                word_count=len(texts[i].split()),
                type='document'
            )
        
        # Add edges (relationships)
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i][j]
                
                if similarity > similarity_threshold:
                    node_i = os.path.basename(filenames[i])
                    node_j = os.path.basename(filenames[j])
                    
                    self.graph.add_edge(
                        node_i, 
                        node_j, 
                        weight=similarity,
                        relationship_strength=self._categorize_similarity(similarity)
                    )
        
        return self.graph
    
    def _categorize_similarity(self, similarity: float) -> str:
        """Categorize similarity strength."""
        if similarity > 0.7:
            return 'strong'
        elif similarity > 0.5:
            return 'moderate'
        else:
            return 'weak'
    
    def generate_interactive_visualization(self) -> go.Figure:
        """
        Create an interactive Plotly visualization of the knowledge graph.
        """
        if len(self.graph.nodes()) == 0:
            return None
            
        # Calculate layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_size = []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = self.graph.nodes[node]
            word_count = node_data.get('word_count', 0)
            
            node_text.append(node)
            node_info.append(f"Document: {node}<br>Words: {word_count}<br>Connections: {len(list(self.graph.neighbors(node)))}")
            node_size.append(max(10, min(50, word_count / 100)))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_info,
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_size,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )
        
        # FIXED: Create figure with correct Plotly syntax for newer versions
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text='Document Knowledge Graph',
                    font=dict(size=16)  # ✅ Fixed: Use dict format instead of titlefont_size
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[dict(
                    text="Hover over nodes to see document details and connections",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="#888", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
    
    def get_document_clusters(self, n_clusters: int = 3) -> Dict:
        """
        Identify document clusters based on similarity.
        """
        if self.document_similarities is None:
            return {}
            
        # Use the similarity matrix to identify clusters
        from sklearn.cluster import AgglomerativeClustering
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, 
            metric='precomputed',
            linkage='average'
        )
        
        # Convert similarity to distance
        distance_matrix = 1 - self.document_similarities
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        return clusters
    
    def get_similarity_stats(self) -> Dict:
        """
        Get statistics about document similarities.
        """
        if self.document_similarities is None:
            return {}
        
        # Get upper triangular part (excluding diagonal)
        upper_tri = np.triu(self.document_similarities, k=1)
        similarities = upper_tri[upper_tri > 0]
        
        if len(similarities) == 0:
            return {
                'avg_similarity': 0,
                'max_similarity': 0,
                'min_similarity': 0,
                'total_connections': 0
            }
        
        return {
            'avg_similarity': float(np.mean(similarities)),
            'max_similarity': float(np.max(similarities)),
            'min_similarity': float(np.min(similarities)),
            'total_connections': len(similarities)
        }
    
    def find_most_connected_documents(self, top_n: int = 5) -> List[Dict]:
        """
        Find the most connected documents in the graph.
        """
        if len(self.graph.nodes()) == 0:
            return []
        
        # Calculate node degrees (number of connections)
        node_degrees = dict(self.graph.degree())
        
        # Sort by degree (most connected first)
        sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for node, degree in sorted_nodes[:top_n]:
            node_data = self.graph.nodes[node]
            result.append({
                'document': node,
                'connections': degree,
                'word_count': node_data.get('word_count', 0)
            })
        
        return result
    
    def get_graph_metrics(self) -> Dict:
        """
        Calculate various graph metrics for analysis.
        """
        if len(self.graph.nodes()) == 0:
            return {}
        
        try:
            # Basic metrics
            num_nodes = len(self.graph.nodes())
            num_edges = len(self.graph.edges())
            
            # Density (how connected the graph is)
            density = nx.density(self.graph) if num_nodes > 1 else 0
            
            # Average clustering coefficient
            clustering_coeff = nx.average_clustering(self.graph) if num_nodes > 2 else 0
            
            # Number of connected components
            num_components = nx.number_connected_components(self.graph)
            
            return {
                'nodes': num_nodes,
                'edges': num_edges,
                'density': round(density, 3),
                'avg_clustering': round(clustering_coeff, 3),
                'connected_components': num_components,
                'is_connected': nx.is_connected(self.graph) if num_nodes > 1 else True
            }
        
        except Exception as e:
            print(f"Warning: Error calculating graph metrics: {e}")
            return {
                'nodes': len(self.graph.nodes()),
                'edges': len(self.graph.edges()),
                'error': str(e)
            }
