"""
ColTrast Training Model for AI Projects Dataset
Based on the HiPerRAG paper's ColTrast algorithm combining contrastive learning and late-interaction techniques
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Tuple, Optional
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Semantic chunking based on HiPerRAG approach
    Divides content into coherent segments using sentence similarity
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.7):
        self.encoder = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
    
    def chunk_text(self, text: str, max_chunk_size: int = 512) -> List[Dict]:
        """
        Chunk text into semantically coherent segments
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return [{"text": text, "chunk_id": 0}]
        
        # Encode sentences
        sentence_embeddings = self.encoder.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_chunk_embedding = sentence_embeddings[0:1]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            chunk_mean_embedding = np.mean(current_chunk_embedding, axis=0, keepdims=True)
            similarity = cosine_similarity(chunk_mean_embedding, sentence_embeddings[i:i+1])[0][0]
            
            # Check if we should add to current chunk or start new one
            current_text_length = sum(len(s) for s in current_chunk)
            
            if similarity > self.similarity_threshold and current_text_length < max_chunk_size:
                current_chunk.append(sentences[i])
                current_chunk_embedding = np.vstack([current_chunk_embedding, sentence_embeddings[i:i+1]])
            else:
                # Finalize current chunk
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": len(chunks),
                    "sentence_count": len(current_chunk)
                })
                
                # Start new chunk
                current_chunk = [sentences[i]]
                current_chunk_embedding = sentence_embeddings[i:i+1]
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                "text": chunk_text,
                "chunk_id": len(chunks),
                "sentence_count": len(current_chunk)
            })
        
        return chunks

class ColTrastDataset(Dataset):
    """
    Dataset for ColTrast training combining Q&A pairs and semantic chunks
    """
    
    def __init__(self, dataset_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunker = SemanticChunker()
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Process data
        self.processed_data = self._process_dataset()
        
    def _process_dataset(self) -> List[Dict]:
        """Process the dataset to create training pairs"""
        processed = []
        
        for project in self.data['projects']:
            project_id = project['project_id']
            readme_content = project['readme_content']
            qa_pairs = project['qa_pairs']
            
            # Create semantic chunks from README
            chunks = self.chunker.chunk_text(readme_content)
            
            # Create training examples from Q&A pairs
            for qa in qa_pairs:
                question = qa['question']
                answer = qa['answer']
                context = qa.get('context', '')
                
                # Find best matching chunk for this Q&A pair
                best_chunk = self._find_best_chunk(question + ' ' + answer, chunks)
                
                # Create positive pair (question, relevant chunk)
                processed.append({
                    'question': question,
                    'positive_chunk': best_chunk['text'],
                    'project_id': project_id,
                    'qa_id': qa.get('qa_id', ''),
                    'is_positive': True
                })
                
                # Create negative pairs (question, irrelevant chunks from other projects)
                negative_chunks = self._get_negative_chunks(project_id, chunks)
                for neg_chunk in negative_chunks[:2]:  # Limit negatives
                    processed.append({
                        'question': question,
                        'positive_chunk': neg_chunk['text'],
                        'project_id': project_id,
                        'qa_id': qa.get('qa_id', ''),
                        'is_positive': False
                    })
        
        return processed
    
    def _find_best_chunk(self, query_text: str, chunks: List[Dict]) -> Dict:
        """Find the most relevant chunk for a query"""
        if not chunks:
            return {"text": "", "chunk_id": 0}
        
        # For simplicity, use the first chunk that contains relevant keywords
        # In practice, you'd use semantic similarity
        query_lower = query_text.lower()
        for chunk in chunks:
            if any(word in chunk['text'].lower() for word in query_lower.split()[:3]):
                return chunk
        
        return chunks[0]  # Return first chunk as fallback
    
    def _get_negative_chunks(self, current_project_id: str, current_chunks: List[Dict]) -> List[Dict]:
        """Get negative chunks from other projects"""
        negative_chunks = []
        
        for project in self.data['projects']:
            if project['project_id'] != current_project_id:
                project_chunks = self.chunker.chunk_text(project['readme_content'])
                negative_chunks.extend(project_chunks[:1])  # Take one chunk per other project
        
        return negative_chunks
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        # Tokenize question and chunk
        question_encoding = self.tokenizer(
            item['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        chunk_encoding = self.tokenizer(
            item['positive_chunk'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'question_input_ids': question_encoding['input_ids'].squeeze(),
            'question_attention_mask': question_encoding['attention_mask'].squeeze(),
            'chunk_input_ids': chunk_encoding['input_ids'].squeeze(),
            'chunk_attention_mask': chunk_encoding['attention_mask'].squeeze(),
            'is_positive': torch.tensor(item['is_positive'], dtype=torch.float),
            'project_id': item['project_id']
        }

class ColTrastModel(nn.Module):
    """
    ColTrast model implementing contrastive learning with late interaction
    Based on HiPerRAG's approach
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", hidden_dim: int = 384):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.temperature = 0.07  # Temperature parameter for contrastive loss
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text using the transformer model"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        pooled_embedding = torch.sum(token_embeddings * attention_mask_expanded, 1) / torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        
        # Apply projection head
        projected_embedding = self.projection_head(pooled_embedding)
        
        return F.normalize(projected_embedding, p=2, dim=1), token_embeddings
    
    def late_interaction_similarity(self, question_tokens, chunk_tokens, q_mask, c_mask):
        """
        Compute late interaction similarity (ColBERT-style)
        """
        # Normalize token embeddings
        question_tokens = F.normalize(question_tokens, p=2, dim=-1)
        chunk_tokens = F.normalize(chunk_tokens, p=2, dim=-1)
        
        # Compute token-level similarities
        similarities = torch.matmul(question_tokens, chunk_tokens.transpose(-1, -2))
        
        # Apply masks
        q_mask_expanded = q_mask.unsqueeze(-1).expand_as(similarities)
        c_mask_expanded = c_mask.unsqueeze(1).expand_as(similarities)
        mask = q_mask_expanded & c_mask_expanded
        
        # Max pooling over chunk tokens for each question token
        similarities = similarities.masked_fill(~mask, -1e9)
        max_similarities = similarities.max(dim=-1)[0]
        
        # Sum over question tokens (with masking)
        max_similarities = max_similarities.masked_fill(~q_mask, 0)
        late_interaction_score = max_similarities.sum(dim=-1)
        
        return late_interaction_score
    
    def forward(self, question_input_ids, question_attention_mask, 
                chunk_input_ids, chunk_attention_mask):
        
        # Encode question and chunk
        question_emb, question_tokens = self.encode_text(question_input_ids, question_attention_mask)
        chunk_emb, chunk_tokens = self.encode_text(chunk_input_ids, chunk_attention_mask)
        
        # Contrastive similarity (pooled embeddings)
        contrastive_sim = torch.matmul(question_emb, chunk_emb.T) / self.temperature
        
        # Late interaction similarity (token-level)
        late_interaction_sim = self.late_interaction_similarity(
            question_tokens, chunk_tokens, 
            question_attention_mask, chunk_attention_mask
        )
        
        return contrastive_sim, late_interaction_sim

def contrastive_loss(similarities, labels, temperature=0.07):
    """
    Compute contrastive loss
    """
    batch_size = similarities.size(0)
    
    # Create labels for contrastive learning
    # Positive pairs are on the diagonal
    contrastive_labels = torch.arange(batch_size).to(similarities.device)
    
    # Compute loss
    loss = F.cross_entropy(similarities, contrastive_labels)
    
    return loss

def late_interaction_loss(li_scores, is_positive):
    """
    Compute late interaction loss
    """
    # Binary classification loss for late interaction scores
    targets = is_positive
    loss = F.binary_cross_entropy_with_logits(li_scores, targets)
    
    return loss

class ColTrastTrainer:
    """
    Trainer for ColTrast model
    """
    
    def __init__(self, model, train_loader, val_loader, device, learning_rate=2e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * 10  # Assuming 10 epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )
        
        # Loss weights (as in HiPerRAG)
        self.contrastive_weight = 0.5
        self.late_interaction_weight = 0.5
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_contrastive_loss = 0
        total_li_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            contrastive_sim, li_scores = self.model(
                batch['question_input_ids'],
                batch['question_attention_mask'],
                batch['chunk_input_ids'],
                batch['chunk_attention_mask']
            )
            
            # Compute losses
            c_loss = contrastive_loss(contrastive_sim, batch['is_positive'])
            li_loss = late_interaction_loss(li_scores, batch['is_positive'])
            
            # Combined loss (ColTrast approach)
            loss = (self.contrastive_weight * c_loss + 
                   self.late_interaction_weight * li_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_contrastive_loss += c_loss.item()
            total_li_loss += li_loss.item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'C_Loss': f'{c_loss.item():.4f}',
                'LI_Loss': f'{li_loss.item():.4f}'
            })
        
        return {
            'total_loss': total_loss / len(self.train_loader),
            'contrastive_loss': total_contrastive_loss / len(self.train_loader),
            'late_interaction_loss': total_li_loss / len(self.train_loader)
        }
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                contrastive_sim, li_scores = self.model(
                    batch['question_input_ids'],
                    batch['question_attention_mask'],
                    batch['chunk_input_ids'],
                    batch['chunk_attention_mask']
                )
                
                c_loss = contrastive_loss(contrastive_sim, batch['is_positive'])
                li_loss = late_interaction_loss(li_scores, batch['is_positive'])
                loss = self.contrastive_weight * c_loss + self.late_interaction_weight * li_loss
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs=10):
        """Full training loop"""
        train_losses = []
        val_losses = []
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            train_loss = train_metrics['total_loss']
            
            # Validate
            val_loss = self.validate()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Contrastive Loss: {train_metrics['contrastive_loss']:.4f}, "
                       f"Late Interaction Loss: {train_metrics['late_interaction_loss']:.4f}")
        
        return train_losses, val_losses

# Main training script
def main():
    """Main training function"""
    
    # Configuration
    config = {
        'dataset_path': 'ai_projects_dataset.json',
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'batch_size': 8,
        'learning_rate': 2e-5,
        'num_epochs': 10,
        'max_length': 512,
        'val_split': 0.2
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = ColTrastModel(config['model_name'])
    
    # Create datasets
    full_dataset = ColTrastDataset(config['dataset_path'], tokenizer, config['max_length'])
    
    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(config['val_split'] * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )
    
    logger.info(f"Dataset size: {dataset_size}")
    logger.info(f"Train size: {train_size}, Val size: {val_size}")
    
    # Initialize trainer
    trainer = ColTrastTrainer(model, train_loader, val_loader, device, config['learning_rate'])
    
    # Train model
    train_losses, val_losses = trainer.train(config['num_epochs'])
    
    # Save model
    torch.save(model.state_dict(), 'coltrast_model.pth')
    logger.info("Model saved as 'coltrast_model.pth'")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ColTrast Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.show()
    
    return model, trainer

if __name__ == "__main__":
    # First, run the dataset creation script
    exec(open('sample_dataset.py').read())
    
    # Then run the training
    trained_model, trainer = main()
