# ColTrast AI Projects Embedding System

This system implements the ColTrast approach from the HiPerRAG paper to fine-tune embedding models for AI project question-answering and retrieval.

## Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
torch>=1.9.0
transformers>=4.21.0
sentence-transformers>=2.2.0
scikit-learn>=1.0.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
faiss-cpu>=1.7.0
datasets>=2.0.0
accelerate>=0.20.0
```

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **For GPU support (recommended):**
```bash
# Replace faiss-cpu with faiss-gpu if you have CUDA
pip uninstall faiss-cpu
pip install faiss-gpu
```

## Usage Guide

### Step 1: Create Sample Dataset

Run the dataset creation script:

```python
# This creates ai_projects_dataset.json with sample AI projects
exec(open('sample_dataset.py').read())
```

The dataset includes:
- 3 sample AI projects (Medical Diagnosis, Sentiment Analysis, Trading Bot)
- README-style documentation for each project
- Q&A pairs covering common questions about AI projects
- Semantic chunking of project descriptions

### Step 2: Train the ColTrast Model

```python
# Train the embedding model using ColTrast approach
python coltrast_training.py
```

This will:
- Create semantic chunks from project descriptions
- Generate positive and negative training pairs
- Fine-tune a sentence transformer model using contrastive learning + late interaction
- Save the trained model as `coltrast_model.pth`
- Generate training curves plot

**Training Details:**
- **Base Model:** sentence-transformers/all-MiniLM-L6-v2
- **Training Approach:** ColTrast (Contrastive + Late Interaction)
- **Batch Size:** 8 (adjustable based on GPU memory)
- **Learning Rate:** 2e-5
- **Epochs:** 10

### Step 3: Use the Trained Model for Inference

```python
# Run the inference demo
python inference_script.py
```

This demonstrates:
- Question answering about AI projects
- Semantic search and retrieval
- Project comparison and exploration

## Key Features

### 1. Semantic Chunking
- Automatically splits project documentation into coherent segments
- Uses sentence similarity to maintain context
- Optimizes chunk size for embedding model input limits


Each README is segmented into semantic chunks using sentence embeddings:

Embeddings are created with SentenceTransformer (e.g. "all-MiniLM-L6-v2").

Sentences are merged into chunks as long as their cosine similarity stays above a threshold (0.7).
For each Q&A pair:

- Positive pair: A chunk from the same project most relevant to the Q&A (found via keyword overlap).
- Negative pairs: Chunks from other projects considered irrelevant.

### 2. ColTrast Training
- **Contrastive Learning:** Learns to distinguish between relevant and irrelevant content
- **Late Interaction:** Fine-grained token-level comparison for better semantic matching
- **Combined Loss:** Balances both approaches for optimal performance

In the ColTrast model, the training goal is to bring a question closer to its correct answer chunk in embedding space and push it away from unrelated chunks.

**Contrastive Loss (Global similarity)**

It uses the pooled embedding (vector summary of the full question/chunk).

Think of it as asking:
"Are these two texts globally similar?"

| Question                  | Chunk                               | Label |
| ------------------------- | ----------------------------------- | ----- |
| "What is the model used?" | "We use a transformer-based model"  | ✅ +1  |
| "What is the model used?" | "The dataset was collected in 2022" | ❌ 0   |


→ You take their pooled embeddings and use cosine similarity, then apply cross-entropy loss to encourage positives to be close and negatives to be far.

**Late Interaction Loss (Token-level match)**

Inspired by ColBERT: Instead of looking at the sentence as a whole, it compares each token in the question to each token in the chunk.

It asks:
"Does the question have token-level alignment with the chunk?"

How it works:

- Compute similarity between each question token and all chunk tokens.
- Keep the max similarity for each question token.
- Sum them → get a final similarity score.
- Use binary cross-entropy loss:
  - Positive pairs → score should be high
  - Negative pairs → score should be low


Contrastive loss helps learn general semantic similarity between question/chunk pairs.

Late interaction loss helps match fine-grained details—like specific words or technical terms.


### 3. Fast Retrieval
- FAISS indexing for efficient similarity search
- Normalized embeddings for cosine similarity
- Batch processing for large-scale inference

Normalization:

| Without normalization                                           | With normalization (unit vectors)                       |
| --------------------------------------------------------------- | ------------------------------------------------------- |
| Vectors may have large variance in scale → affects dot products | Cosine similarity becomes stable and bounded (−1 to +1) |
| Embedding similarity may be biased by magnitude                 | Focus is on **direction**, not length                   |


| Location              | Type of normalization   | Why?                                        |
| --------------------- | ----------------------- | ------------------------------------------- |
| After projection head | L2 (sentence embedding) | Contrastive loss uses cosine similarity     |
| On token embeddings   | L2 (per token)          | Late interaction similarity (ColBERT-style) |


## Example Queries

The system can answer questions like:

- **"What AI model is used for medical diagnosis?"**
  - Retrieves information about ResNet-50 from the medical imaging project

- **"What is the data size for sentiment analysis?"**
  - Returns details about the 500K+ review dataset

- **"What are the lessons learned from the trading bot?"**
  - Provides insights about market volatility and risk management

- **"Which project uses reinforcement learning?"**
  - Identifies the autonomous trading bot project

## Customization for Your Data

### 1. Prepare Your Dataset

Create a JSON file with this structure:

```json
{
  "projects": [
    {
      "project_id": "your_project_1",
      "readme_content": "# Your Project\n\nProject description...",
      "qa_pairs": [
        {
          "question": "What is the project name?",
          "answer": "Your Project Name",
          "context": "Brief context"
        }
      ]
    }
  ]
}
```

### 2. Modify the Training Script

Update the dataset processing in `ColTrastDataset` class:

```python
# Adjust the chunking parameters
self.chunker = SemanticChunker(
    similarity_threshold=0.7,  # Adjust based on your content
    max_chunk_size=512        # Adjust based on your model
)

# Modify question-answer matching logic
def _find_best_chunk(self, query_text, chunks):
    # Implement your domain-specific matching logic
    pass
```

### 3. Adjust Training Parameters

Modify the config in `main()` function:

```python
config = {
    'dataset_path': 'your_dataset.json',
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',  # Or another model
    'batch_size': 16,        # Increase if you have more GPU memory
    'learning_rate': 1e-5,   # Adjust based on your data size
    'num_epochs': 20,        # More epochs for larger datasets
    'max_length': 512,
    'val_split': 0.2
}
```

## Advanced Usage

### 1. Using Different Base Models

```python
# For better performance, try larger models:
model_name = "sentence-transformers/all-mpnet-base-v2"
model_name = "microsoft/DialoGPT-medium"
model_name = "facebook/dpr-question_encoder-single-nq-base"
```

### 2. Multi-GPU Training

```python
# Use DataParallel or DistributedDataParallel
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### 3. Hyperparameter Tuning

```python
# Experiment with loss weights
trainer.contrastive_weight = 0.3
trainer.late_interaction_weight = 0.7

# Adjust temperature parameter
model.temperature = 0.05  # Lower = harder negative mining
```

## Performance Tips

1. **Data Quality:** Ensure your Q&A pairs are high-quality and representative
2. **Chunking Strategy:** Adjust similarity thresholds based on your content structure
3. **Negative Sampling:** Include diverse negative examples for better contrastive learning
4. **Validation:** Use project-wise splits to avoid data leakage
5. **Model Size:** Balance between accuracy and inference speed based on your needs

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory:**
   - Reduce batch_size
   - Use gradient accumulation
   - Try model sharding

2. **Poor Retrieval Quality:**
   - Increase training epochs
   - Improve negative sampling strategy
   - Adjust similarity thresholds

3. **Slow Inference:**
   - Use FAISS GPU index
   - Implement model quantization
   - Cache embeddings

## Scaling to Production

For production deployment:

1. **Model Optimization:**
   - Convert to ONNX format
   - Use TensorRT for GPU acceleration
   - Implement model quantization

2. **Index Management:**
   - Use persistent FAISS indices
   - Implement incremental updates
   - Consider distributed search (e.g., Elasticsearch)

3. **API Development:**
   - Create REST API endpoints
   - Implement batch processing
   - Add monitoring and logging

This system provides a solid foundation for building AI project knowledge retrieval systems similar to the approach described in the HiPerRAG paper, adapted for your specific use case.

## Clarification: 

**1. Relationship between readme and q&a**

⚠️ Problem:
At first there is a problem in best chunk selection and negative pair creation: 

The system tries to find a chunk in the README that "matches" the Q&A pair using simple keyword overlap (not semantic search).

If it can’t find a good match, it just returns the first chunk as a fallback (see _find_best_chunk() in ColTrastDataset).

If the README doesn’t mention the answer at all, the positive pair might be misleading (question paired with unrelated chunk).

✅FIX: 

🔹 For Positive Chunks:
You now check semantic similarity between the Q+A and README chunks.

If no good match is found (e.g., similarity < 0.6), you use the answer text itself as the chunk.

✅ This guarantees that:

If README is relevant, it's used.

If README is irrelevant, we avoid misleading the model and use a Q&A-only pair.

🔹 For Negative Chunks:
Instead of using random chunks from other projects, you now:

Compute semantic similarity to the Q+A

Pick the most similar (but wrong) chunks as hard negatives

✅ This forces the model to:

Distinguish true relevance from close but misleading content.

Avoid learning from "too easy" or unrelated negatives.

After the fix, the training loss is better: 

![image](training_loss/trainingloss_comparison.png)

Left plot (before fix): 

- Validation loss steadily decreases, but the gap between train/val remains.
- Signs of underfitting or low-quality contrast in examples.

Right plot (after fix):

- Train loss drops faster and lower.
- Validation loss follows closely at first, then diverges slightly near the end. (model might start overfitting)

🔹Early stopping or regularization after epoch 7 to avoid overfitting.


**2. Are embeddings computed twice for the whole dataset?**

| Stage                           | What is embedded                    | Why                                              |
| ------------------------------- | ----------------------------------- | ------------------------------------------------ |
| 🔹 Chunking (semantic grouping) | Sentence-level embeddings           | To decide where to split the README into chunks  |
| 🔹 Training                     | Token-level embeddings from encoder | To compute loss (contrastive + late interaction) |


- Chunking stage uses a fast SentenceTransformer (MiniLM) to create embeddings for similarity-based chunking.
- Training stage uses a full encoder with a projection head to learn task-specific embeddings.

They serve different roles and are not redundant.


**3. Where do we store embeddings for inference?**

in inference_script.py → index_documents():

    [self.chunk_embeddings = self.encode_batch(chunk_texts)
    
    # Add to FAISS index
    self.faiss_index.add(...)
    self.chunks = chunk_metadata  # Store associated metadata]()

If you want to persist them:

    save_index(path)  # saves .faiss + .pkl

That way, you don’t need to re-encode the dataset every time you load the model.

