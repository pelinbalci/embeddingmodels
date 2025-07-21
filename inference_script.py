"""
ColTrast Inference and Retrieval System
Demonstrates how to use the fine-tuned embedding model for query-based retrieval
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class ColTrastInference:
    """
    Inference class for the trained ColTrast model
    """
    
    def __init__(self, model_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the trained model
        from coltrast_training import ColTrastModel  # Import from training script
        self.model = ColTrastModel(model_name)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Storage for indexed chunks
        self.chunks = []
        self.chunk_embeddings = None
        self.faiss_index = None
        
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding vector
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get embedding (pooled representation)
            pooled_emb, _ = self.model.encode_text(input_ids, attention_mask)
            
            # Convert to numpy
            embedding = pooled_emb.cpu().numpy()
            
        return embedding[0]  # Return single embedding
    
    def encode_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Encode multiple texts in batches
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                # Move to device
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                # Get embeddings
                batch_embeddings, _ = self.model.encode_text(input_ids, attention_mask)
                
                # Add to list
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def index_documents(self, dataset_path: str):
        """
        Index all documents from the dataset for retrieval
        """
        print("Loading and indexing documents...")
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create chunks from all projects
        from coltrast_training import SemanticChunker
        chunker = SemanticChunker()
        
        all_chunks = []
        chunk_metadata = []
        
        for project in data['projects']:
            project_id = project['project_id']
            readme_content = project['readme_content']
            
            # Create semantic chunks
            chunks = chunker.chunk_text(readme_content)
            
            for chunk in chunks:
                all_chunks.append(chunk['text'])
                chunk_metadata.append({
                    'project_id': project_id,
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text']
                })
        
        # Store chunks and metadata
        self.chunks = chunk_metadata
        
        # Encode all chunks
        print(f"Encoding {len(all_chunks)} chunks...")
        chunk_texts = [chunk['text'] for chunk in chunk_metadata]
        self.chunk_embeddings = self.encode_batch(chunk_texts)
        
        # Create FAISS index for fast similarity search
        dimension = self.chunk_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.chunk_embeddings / np.linalg.norm(
            self.chunk_embeddings, axis=1, keepdims=True
        )
        
        self.faiss_index.add(normalized_embeddings.astype('float32'))
        
        print(f"Indexed {len(all_chunks)} chunks successfully!")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for most relevant chunks given a query
        """
        if self.faiss_index is None:
            raise ValueError("Documents not indexed. Call index_documents() first.")
        
        # Encode query
        query_embedding = self.encode_text(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype('float32'), top_k
        )
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):  # Valid index
                chunk_info = self.chunks[idx].copy()
                chunk_info['similarity_score'] = float(score)
                chunk_info['rank'] = i + 1
                results.append(chunk_info)
        
        return results
    
    def answer_question(self, question: str, top_k: int = 3) -> Dict:
        """
        Answer a question by retrieving relevant context and providing structured response
        """
        # Search for relevant chunks
        relevant_chunks = self.search(question, top_k)
        
        # Combine context from top chunks
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"[Project: {chunk['project_id']}] {chunk['text']}")
        
        combined_context = "\n\n".join(context_parts)
        
        return {
            'question': question,
            'relevant_chunks': relevant_chunks,
            'combined_context': combined_context,
            'top_project': relevant_chunks[0]['project_id'] if relevant_chunks else None,
            'confidence': relevant_chunks[0]['similarity_score'] if relevant_chunks else 0.0
        }
    
    def save_index(self, index_path: str):
        """Save the FAISS index and metadata"""
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, f"{index_path}.faiss")
            
            with open(f"{index_path}_metadata.pkl", 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'embeddings': self.chunk_embeddings
                }, f)
            
            print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str):
        """Load a pre-built FAISS index and metadata"""
        self.faiss_index = faiss.read_index(f"{index_path}.faiss")
        
        with open(f"{index_path}_metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_embeddings = data['embeddings']
        
        print(f"Index loaded from {index_path}")

class AIProjectsQA:
    """
    Question-Answering system for AI projects using ColTrast
    """
    
    def __init__(self, model_path: str, dataset_path: str):
        self.retriever = ColTrastInference(model_path)
        self.retriever.index_documents(dataset_path)
        
        # Common question templates for AI projects
        self.question_templates = {
            'project_name': ['What is the project name?', 'What is this project called?'],
            'objective': ['What is the objective?', 'What is the goal?', 'What is the purpose?'],
            'model': ['Which AI model is used?', 'What model is used?', 'What architecture is used?'],
            'data_source': ['What is the data source?', 'Where does the data come from?'],
            'data_size': ['What is the data size?', 'How much data is used?'],
            'lessons': ['What are the lessons learned?', 'What insights were gained?'],
            'performance': ['What are the performance metrics?', 'How well does it perform?'],
            'framework': ['What framework is used?', 'What technology stack is used?']
        }
    
    def ask_question(self, question: str, verbose: bool = True) -> Dict:
        """Ask a question about AI projects"""
        result = self.retriever.answer_question(question)
        
        if verbose:
            print(f"\n🤖 Question: {question}")
            print(f"🎯 Top Project: {result['top_project']}")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            print(f"\n📝 Answer Context:")
            print("-" * 50)
            
            for i, chunk in enumerate(result['relevant_chunks'][:2]):  # Show top 2
                print(f"\n[Rank {chunk['rank']}] Project: {chunk['project_id']}")
                print(f"Similarity: {chunk['similarity_score']:.3f}")
                print(f"Content: {chunk['text'][:200]}...")
        
        return result
    
    def explore_project(self, project_keywords: str) -> Dict:
        """Explore a specific project by keywords"""
        results = {}
        
        print(f"\n🔍 Exploring projects related to: '{project_keywords}'")
        print("=" * 60)
        
        for category, questions in self.question_templates.items():
            query = f"{project_keywords} {questions[0]}"
            result = self.retriever.answer_question(query, top_k=1)
            
            if result['relevant_chunks']:
                results[category] = {
                    'answer': result['relevant_chunks'][0]['text'][:150] + "...",
                    'confidence': result['confidence'],
                    'project': result['top_project']
                }
                
                print(f"\n📌 {category.upper()}:")
                print(f"   {results[category]['answer']}")
                print(f"   (Confidence: {results[category]['confidence']:.3f})")
        
        return results
    
    def compare_projects(self, query: str, top_k: int = 3) -> Dict:
        """Compare multiple projects based on a query"""
        results = self.retriever.search(query, top_k=top_k)
        
        print(f"\n🔄 Comparing projects for: '{query}'")
        print("=" * 60)
        
        project_results = {}
        for result in results:
            project_id = result['project_id']
            if project_id not in project_results:
                project_results[project_id] = []
            project_results[project_id].append(result)
        
        for project_id, chunks in project_results.items():
            best_chunk = max(chunks, key=lambda x: x['similarity_score'])
            print(f"\n🏷️  {project_id.upper()}:")
            print(f"   Relevance: {best_chunk['similarity_score']:.3f}")
            print(f"   Context: {best_chunk['text'][:200]}...")
        
        return project_results

# Example usage and demonstration
def demo():
    """Demonstration of the ColTrast inference system"""
    
    print("🚀 ColTrast AI Projects Q&A System Demo")
    print("=" * 50)
    
    try:
        # Initialize the QA system
        qa_system = AIProjectsQA('coltrast_model.pth', 'ai_projects_dataset.json')
        
        # Example questions
        sample_questions = [
            "What AI model is used for medical diagnosis?",
            "What is the data size for sentiment analysis?",
            "What are the lessons learned from trading bot?",
            "Which project uses reinforcement learning?",
            "What framework is used for computer vision?",
        ]
        
        print("\n📋 Answering Sample Questions:")
        print("-" * 30)
        
        for question in sample_questions:
            qa_system.ask_question(question, verbose=True)
            print("\n" + "="*60 + "\n")
        
        # Explore specific project
        qa_system.explore_project("medical diagnosis")
        
        # Compare projects
        qa_system.compare_projects("deep learning model")
        
        # Save the index for future use
        qa_system.retriever.save_index('ai_projects_index')
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have run the training script first to create the model!")

if __name__ == "__main__":
    demo()
