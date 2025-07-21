"""
OpenAI Embeddings Data Processor for AI Projects
Handles Q&A pairs, README documents, and different question types (boolean, categorical, etc.)
"""

import json
import openai
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import time
import re
from dataclasses import dataclass
from enum import Enum
import pickle
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionType(Enum):
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical" 
    NUMERICAL = "numerical"
    TEXT = "text"
    LIST = "list"

@dataclass
class ProcessedQA:
    """Structured representation of a processed Q&A pair"""
    question: str
    answer: str
    question_type: QuestionType
    project_id: str
    embedding_text: str  # Optimized text for embedding
    metadata: Dict[str, Any]

class OpenAIEmbeddingProcessor:
    """
    Processor for creating optimized embeddings using OpenAI models
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        """
        Initialize with OpenAI API key and model
        
        Note: text-embedding-3-large is the current best model
        (text-ada-embedding-large is deprecated)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
        # Model specifications
        self.model_specs = {
            "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
            "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
            "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191}
        }
        
        self.max_tokens = self.model_specs.get(model, {}).get("max_tokens", 8191)
        self.embedding_dim = self.model_specs.get(model, {}).get("dimensions", 1536)
        
        # Rate limiting
        self.requests_per_minute = 3000  # Adjust based on your tier
        self.request_delay = 60 / self.requests_per_minute
        
        logger.info(f"Initialized OpenAI processor with model: {model}")
        logger.info(f"Embedding dimensions: {self.embedding_dim}")
    
    def detect_question_type(self, question: str, answer: str) -> QuestionType:
        """
        Automatically detect the type of question based on content
        """
        question_lower = question.lower()
        answer_lower = answer.lower().strip()
        
        # Boolean questions
        boolean_patterns = [
            r'\bis\s+(it|there|this)',
            r'\bdoes\s+',
            r'\bcan\s+',
            r'\bwill\s+',
            r'\bhas\s+',
            r'\bare\s+',
            r'\bdo\s+you',
        ]
        
        boolean_answers = ['yes', 'no', 'true', 'false', 'enabled', 'disabled']
        
        if (any(re.search(pattern, question_lower) for pattern in boolean_patterns) or 
            answer_lower in boolean_answers):
            return QuestionType.BOOLEAN
        
        # Numerical questions
        if (any(word in question_lower for word in ['how many', 'what size', 'how much', 'count']) or
            re.match(r'^\d+(\.\d+)?\s*(gb|mb|tb|%|percent|hours?|days?|months?|years?)?$', answer_lower)):
            return QuestionType.NUMERICAL
        
        # List questions (answers with commas, bullets, or "and")
        if (',' in answer or '\n-' in answer or '\n•' in answer or 
            len(answer.split(' and ')) > 2):
            return QuestionType.LIST
        
        # Categorical (short, specific answers)
        if len(answer.split()) <= 5 and not any(char in answer for char in '.!?'):
            return QuestionType.CATEGORICAL
        
        # Default to text
        return QuestionType.TEXT
    
    def create_embedding_text(self, question: str, answer: str, 
                            question_type: QuestionType, context: str = "") -> str:
        """
        Create optimized text for embedding based on question type
        """
        # Base components
        components = []
        
        # Question processing
        clean_question = question.strip().rstrip('?')
        components.append(f"Q: {clean_question}")
        
        # Answer processing based on type
        if question_type == QuestionType.BOOLEAN:
            # For boolean, include explicit yes/no context
            bool_answer = answer.lower()
            if bool_answer in ['yes', 'true', '1', 'enabled']:
                components.append(f"A: Yes, {answer}")
            elif bool_answer in ['no', 'false', '0', 'disabled']:
                components.append(f"A: No, {answer}")
            else:
                components.append(f"A: {answer}")
        
        elif question_type == QuestionType.CATEGORICAL:
            # For categorical, add category context
            components.append(f"A: {answer}")
            # Add semantic context for better retrieval
            if "model" in clean_question.lower():
                components.append("Model/Architecture information")
            elif "data" in clean_question.lower():
                components.append("Dataset/Data information")
            elif "time" in clean_question.lower():
                components.append("Time/Duration information")
        
        elif question_type == QuestionType.NUMERICAL:
            # For numerical, include units and context
            components.append(f"A: {answer}")
            # Extract and emphasize numerical value
            numbers = re.findall(r'\d+(?:\.\d+)?', answer)
            if numbers:
                components.append(f"Numerical value: {numbers[0]}")
        
        elif question_type == QuestionType.LIST:
            # For lists, structure better
            components.append(f"A: {answer}")
            # Extract list items
            items = re.split(r'[,\n•-]\s*', answer)
            if len(items) > 1:
                components.append(f"Items: {', '.join(item.strip() for item in items if item.strip())}")
        
        else:  # TEXT
            components.append(f"A: {answer}")
        
        # Add context if provided
        if context:
            components.append(f"Context: {context}")
        
        # Combine with appropriate separators
        embedding_text = " | ".join(components)
        
        # Ensure we don't exceed token limits (rough estimation: 1 token ≈ 4 characters)
        max_chars = self.max_tokens * 3  # Conservative estimate
        if len(embedding_text) > max_chars:
            embedding_text = embedding_text[:max_chars] + "..."
        
        return embedding_text
    
    def process_qa_pairs(self, qa_data: List[Dict]) -> List[ProcessedQA]:
        """
        Process Q&A pairs into optimized format for embedding
        """
        processed_qas = []
        
        logger.info(f"Processing {len(qa_data)} Q&A pairs...")
        
        for qa in tqdm(qa_data, desc="Processing Q&A pairs"):
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            project_id = qa.get('project_id', '')
            context = qa.get('context', '')
            
            # Detect question type
            q_type = self.detect_question_type(question, answer)
            
            # Create optimized embedding text
            embedding_text = self.create_embedding_text(question, answer, q_type, context)
            
            # Create processed Q&A
            processed_qa = ProcessedQA(
                question=question,
                answer=answer,
                question_type=q_type,
                project_id=project_id,
                embedding_text=embedding_text,
                metadata={
                    'qa_id': qa.get('qa_id', ''),
                    'context': context,
                    'question_length': len(question),
                    'answer_length': len(answer),
                    'embedding_text_length': len(embedding_text)
                }
            )
            
            processed_qas.append(processed_qa)
        
        # Log statistics
        type_counts = {}
        for qa in processed_qas:
            type_counts[qa.question_type.value] = type_counts.get(qa.question_type.value, 0) + 1
        
        logger.info("Question type distribution:")
        for q_type, count in type_counts.items():
            logger.info(f"  {q_type}: {count}")
        
        return processed_qas
    
    def chunk_document(self, text: str, chunk_size: int = 1500, overlap: int = 200) -> List[Dict]:
        """
        Chunk long documents with overlap for better context preservation
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            para_length = len(paragraph)
            
            # If adding this paragraph would exceed chunk size
            if current_length + para_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'length': current_length
                })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
                current_chunk = overlap_text + "\n\n" + paragraph
                current_length = len(current_chunk)
                chunk_id += 1
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_length = len(current_chunk)
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'length': current_length
            })
        
        return chunks
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process README documents into chunks optimized for embedding
        """
        processed_docs = []
        
        logger.info(f"Processing {len(documents)} documents...")
        
        for doc in tqdm(documents, desc="Processing documents"):
            project_id = doc.get('project_id', '')
            content = doc.get('readme_content', '')
            
            # Create chunks
            chunks = self.chunk_document(content)
            
            for chunk in chunks:
                # Add project context to chunk
                chunk_text = f"Project: {project_id}\n\n{chunk['text']}"
                
                processed_doc = {
                    'project_id': project_id,
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk_text,
                    'original_length': len(content),
                    'chunk_length': len(chunk_text),
                    'type': 'document_chunk'
                }
                
                processed_docs.append(processed_doc)
        
        logger.info(f"Created {len(processed_docs)} document chunks")
        return processed_docs
    
    def get_embedding(self, text: str, retry_count: int = 3) -> Optional[np.ndarray]:
        """
        Get embedding from OpenAI API with retry logic
        """
        for attempt in range(retry_count):
            try:
                # Rate limiting
                time.sleep(self.request_delay)
                
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                
                embedding = np.array(response.data[0].embedding)
                return embedding
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to get embedding after {retry_count} attempts")
                    return None
        
        return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Embed multiple texts in batches
        """
        all_embeddings = []
        
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                embedding = self.get_embedding(text)
                if embedding is not None:
                    batch_embeddings.append(embedding)
                else:
                    # Use zero vector for failed embeddings
                    batch_embeddings.append(np.zeros(self.embedding_dim))
            
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def create_embeddings_dataset(self, dataset_path: str) -> Dict:
        """
        Create complete embeddings dataset from AI projects data
        """
        logger.info("Creating embeddings dataset...")
        
        # Load original dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process Q&A pairs
        all_qa_pairs = []
        for project in data['projects']:
            for qa in project['qa_pairs']:
                qa['project_id'] = project['project_id']
                all_qa_pairs.append(qa)
        
        processed_qas = self.process_qa_pairs(all_qa_pairs)
        
        # Process documents
        documents = [{'project_id': p['project_id'], 'readme_content': p['readme_content']} 
                    for p in data['projects']]
        processed_docs = self.process_documents(documents)
        
        # Prepare texts for embedding
        qa_texts = [qa.embedding_text for qa in processed_qas]
        doc_texts = [doc['text'] for doc in processed_docs]
        
        # Create embeddings
        logger.info("Creating Q&A embeddings...")
        qa_embeddings = self.embed_batch(qa_texts)
        
        logger.info("Creating document embeddings...")
        doc_embeddings = self.embed_batch(doc_texts)
        
        # Combine everything
        embeddings_dataset = {
            'qa_data': [
                {
                    'question': qa.question,
                    'answer': qa.answer,
                    'question_type': qa.question_type.value,
                    'project_id': qa.project_id,
                    'embedding_text': qa.embedding_text,
                    'metadata': qa.metadata,
                    'embedding': qa_embeddings[i].tolist()
                }
                for i, qa in enumerate(processed_qas)
            ],
            'document_data': [
                {
                    **doc,
                    'embedding': doc_embeddings[i].tolist()
                }
                for i, doc in enumerate(processed_docs)
            ],
            'metadata': {
                'model': self.model,
                'embedding_dimension': self.embedding_dim,
                'total_qa_pairs': len(processed_qas),
                'total_document_chunks': len(processed_docs),
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        return embeddings_dataset

class OpenAIRetriever:
    """
    Retrieval system using OpenAI embeddings
    """
    
    def __init__(self, embeddings_dataset: Dict, api_key: str):
        self.dataset = embeddings_dataset
        self.client = openai.OpenAI(api_key=api_key)
        self.model = embeddings_dataset['metadata']['model']
        
        # Create FAISS indices
        self.setup_indices()
    
    def setup_indices(self):
        """Setup FAISS indices for fast similarity search"""
        
        # Q&A index
        qa_embeddings = np.array([item['embedding'] for item in self.dataset['qa_data']])
        self.qa_index = faiss.IndexFlatIP(qa_embeddings.shape[1])
        
        # Normalize for cosine similarity
        qa_embeddings_norm = qa_embeddings / np.linalg.norm(qa_embeddings, axis=1, keepdims=True)
        self.qa_index.add(qa_embeddings_norm.astype('float32'))
        
        # Document index
        doc_embeddings = np.array([item['embedding'] for item in self.dataset['document_data']])
        self.doc_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        
        doc_embeddings_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        self.doc_index.add(doc_embeddings_norm.astype('float32'))
        
        logger.info(f"Indices created: {len(self.dataset['qa_data'])} Q&As, {len(self.dataset['document_data'])} docs")
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query"""
        response = self.client.embeddings.create(
            input=query,
            model=self.model
        )
        embedding = np.array(response.data[0].embedding)
        return embedding / np.linalg.norm(embedding)  # Normalize
    
    def search_qa(self, query: str, top_k: int = 5, question_type: str = None) -> List[Dict]:
        """Search Q&A pairs"""
        query_embedding = self.embed_query(query)
        
        # Search in FAISS index
        scores, indices = self.qa_index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            min(top_k * 2, len(self.dataset['qa_data']))  # Get more results for filtering
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.dataset['qa_data']):
                qa_item = self.dataset['qa_data'][idx].copy()
                qa_item['similarity_score'] = float(score)
                
                # Filter by question type if specified
                if question_type is None or qa_item['question_type'] == question_type:
                    results.append(qa_item)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def search_documents(self, query: str, top_k: int = 5, project_id: str = None) -> List[Dict]:
        """Search document chunks"""
        query_embedding = self.embed_query(query)
        
        scores, indices = self.doc_index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            min(top_k * 2, len(self.dataset['document_data']))
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.dataset['document_data']):
                doc_item = self.dataset['document_data'][idx].copy()
                doc_item['similarity_score'] = float(score)
                
                # Filter by project if specified
                if project_id is None or doc_item['project_id'] == project_id:
                    results.append(doc_item)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5) -> Dict:
        """Combine Q&A and document search"""
        qa_results = self.search_qa(query, top_k//2 + 1)
        doc_results = self.search_documents(query, top_k//2 + 1)
        
        return {
            'query': query,
            'qa_results': qa_results,
            'document_results': doc_results,
            'combined_results': sorted(
                qa_results + doc_results,
                key=lambda x: x['similarity_score'],
                reverse=True
            )[:top_k]
        }
    
    def answer_question(self, question: str) -> Dict:
        """Provide structured answer to a question"""
        # First search Q&A pairs
        qa_results = self.search_qa(question, top_k=3)
        
        if qa_results and qa_results[0]['similarity_score'] > 0.8:
            # High confidence Q&A match
            best_qa = qa_results[0]
            return {
                'question': question,
                'answer': best_qa['answer'],
                'confidence': best_qa['similarity_score'],
                'source': 'qa_pair',
                'project_id': best_qa['project_id'],
                'question_type': best_qa['question_type'],
                'supporting_results': qa_results
            }
        else:
            # Search in documents for context
            doc_results = self.search_documents(question, top_k=3)
            
            if doc_results:
                # Extract relevant context from documents
                context = doc_results[0]['text'][:500] + "..."
                return {
                    'question': question,
                    'answer': f"Based on the documentation: {context}",
                    'confidence': doc_results[0]['similarity_score'],
                    'source': 'document',
                    'project_id': doc_results[0]['project_id'],
                    'question_type': 'inferred_from_docs',
                    'supporting_results': doc_results
                }
            else:
                return {
                    'question': question,
                    'answer': "No relevant information found.",
                    'confidence': 0.0,
                    'source': 'none',
                    'project_id': None,
                    'question_type': 'unknown',
                    'supporting_results': []
                }

# Example usage and demo
def create_sample_data_with_varied_questions():
    """Create sample data with boolean, categorical, and other question types"""
    
    sample_dataset = {
        "projects": [
            {
                "project_id": "medical_ai",
                "readme_content": """
# Medical AI Classification System

## Overview
This project implements a deep learning system for medical image classification.

## Technical Details
- **Model**: ResNet-50 with transfer learning
- **Framework**: PyTorch
- **Data Collection Period**: 6 months
- **GPU Acceleration**: Yes, using CUDA
- **Real-time Processing**: Enabled
- **Accuracy**: 96.2%
- **Dataset Size**: 50,000 images
- **Preprocessing Steps**: Normalization, augmentation, resizing
                """,
                "qa_pairs": [
                    {
                        "question": "Is GPU acceleration enabled?",
                        "answer": "Yes",
                        "context": "Hardware acceleration for training"
                    },
                    {
                        "question": "What is the time period for data collection?", 
                        "answer": "6 months",
                        "context": "Data collection duration"
                    },
                    {
                        "question": "What model architecture is used?",
                        "answer": "ResNet-50",
                        "context": "Deep learning model"
                    },
                    {
                        "question": "What preprocessing steps are applied?",
                        "answer": "Normalization, augmentation, resizing",
                        "context": "Data preprocessing pipeline"
                    },
                    {
                        "question": "Does the system support real-time processing?",
                        "answer": "Yes, enabled",
                        "context": "Real-time capabilities"
                    }
                ]
            },
            {
                "project_id": "nlp_sentiment",
                "readme_content": """
# NLP Sentiment Analysis

## Configuration
- **Model Type**: Transformer-based (BERT)
- **Training Duration**: 2 weeks
- **Distributed Training**: No
- **Batch Size**: 32
- **Learning Rate**: 2e-5
- **Accuracy**: 89.3%
                """,
                "qa_pairs": [
                    {
                        "question": "Is distributed training used?",
                        "answer": "No",
                        "context": "Training configuration"
                    },
                    {
                        "question": "What is the batch size?",
                        "answer": "32",
                        "context": "Training hyperparameters"
                    },
                    {
                        "question": "What is the learning rate?",
                        "answer": "2e-5",
                        "context": "Optimization parameters"
                    }
                ]
            }
        ]
    }
    
    return sample_dataset

def main():
    """Main demo function"""
    
    # Configuration
    OPENAI_API_KEY = "your-api-key-here"  # Replace with your actual API key
    
    if OPENAI_API_KEY == "your-api-key-here":
        logger.error("Please set your OpenAI API key!")
        return
    
    # Create sample data
    sample_data = create_sample_data_with_varied_questions()
    
    # Save sample data
    with open('sample_ai_projects.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # Initialize processor
    processor = OpenAIEmbeddingProcessor(OPENAI_API_KEY)
    
    # Create embeddings dataset
    embeddings_dataset = processor.create_embeddings_dataset('sample_ai_projects.json')
    
    # Save embeddings dataset
    with open('openai_embeddings_dataset.json', 'w') as f:
        json.dump(embeddings_dataset, f, indent=2)
    
    logger.info("Embeddings dataset created successfully!")
    
    # Initialize retriever
    retriever = OpenAIRetriever(embeddings_dataset, OPENAI_API_KEY)
    
    # Demo queries
    demo_queries = [
        "Is GPU acceleration used?",  # Boolean
        "What is the batch size?",    # Numerical  
        "What model is used?",        # Categorical
        "What preprocessing steps are applied?",  # List
        "How long was the training?", # Time period
    ]
    
    print("\n🤖 OpenAI Embeddings Demo")
    print("=" * 50)
    
    for query in demo_queries:
        print(f"\n❓ Query: {query}")
        result = retriever.answer_question(query)
        print(f"✅ Answer: {result['answer']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print(f"🏷️  Type: {result['question_type']}")
        print(f"📁 Project: {result['project_id']}")

if __name__ == "__main__":
    main()
