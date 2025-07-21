"""
Simple usage example for your 50 AI projects dataset with OpenAI embeddings
"""

import json
import openai
from openai_embedding_processor import OpenAIEmbeddingProcessor, OpenAIRetriever

def create_your_dataset_format():
    """
    Example of how to format your 50 AI projects dataset
    """
    
    # This is the format you should use for your actual data
    your_dataset = {
        "projects": [
            {
                "project_id": "project_001_image_classification",
                "readme_content": """
# Computer Vision for Medical Diagnosis

## Project Overview
This project implements a CNN-based system for classifying medical images.

## Technical Specifications
- **Model Architecture**: ResNet-50 with transfer learning
- **Framework**: PyTorch 1.9
- **Training Duration**: 3 weeks
- **GPU Requirements**: NVIDIA V100 or better
- **Real-time Processing**: Yes
- **Deployment**: Docker containerized
- **Data Security**: HIPAA compliant

## Dataset Information
- **Data Source**: Hospital imaging database
- **Collection Period**: 18 months
- **Image Resolution**: 1024x1024 pixels
- **Total Images**: 75,000 images
- **Data Split**: 60% train, 20% validation, 20% test

## Performance Metrics
- **Accuracy**: 94.7%
- **Precision**: 93.2%
- **Recall**: 95.1%
- **F1-Score**: 94.1%
                """,
                "qa_pairs": [
                    # Boolean questions
                    {
                        "question": "Is real-time processing supported?",
                        "answer": "Yes",
                        "context": "System capabilities"
                    },
                    {
                        "question": "Is the system HIPAA compliant?",
                        "answer": "Yes",
                        "context": "Data security and compliance"
                    },
                    {
                        "question": "Does the model use transfer learning?",
                        "answer": "Yes",
                        "context": "Model training approach"
                    },
                    
                    # Categorical/Dropdown questions
                    {
                        "question": "What is the time period for data collection?",
                        "answer": "18 months",
                        "context": "Data collection duration"
                    },
                    {
                        "question": "What framework is used?",
                        "answer": "PyTorch",
                        "context": "Development framework"
                    },
                    {
                        "question": "What model architecture is implemented?",
                        "answer": "ResNet-50",
                        "context": "Deep learning architecture"
                    },
                    {
                        "question": "What is the deployment method?",
                        "answer": "Docker containerized",
                        "context": "Deployment strategy"
                    },
                    
                    # Numerical questions
                    {
                        "question": "What is the model accuracy?",
                        "answer": "94.7%",
                        "context": "Performance metrics"
                    },
                    {
                        "question": "How many images are in the dataset?",
                        "answer": "75,000 images",
                        "context": "Dataset size"
                    },
                    {
                        "question": "What is the image resolution?",
                        "answer": "1024x1024 pixels",
                        "context": "Image specifications"
                    },
                    
                    # List/Multiple value questions
                    {
                        "question": "What are the performance metrics?",
                        "answer": "Accuracy: 94.7%, Precision: 93.2%, Recall: 95.1%, F1-Score: 94.1%",
                        "context": "Complete performance evaluation"
                    }
                ]
            },
            
            # Add more projects following the same pattern...
            {
                "project_id": "project_002_nlp_chatbot",
                "readme_content": """
# Conversational AI Chatbot

## Overview
Enterprise chatbot using transformer-based language models.

## Technical Details
- **Model**: GPT-based architecture with fine-tuning
- **Context Window**: 4096 tokens
- **Response Time**: < 2 seconds
- **Multi-language Support**: Yes (English, Spanish, French)
- **Integration**: REST API and WebSocket
                """,
                "qa_pairs": [
                    {
                        "question": "Does it support multiple languages?",
                        "answer": "Yes",
                        "context": "Language capabilities"
                    },
                    {
                        "question": "What is the context window size?",
                        "answer": "4096 tokens",
                        "context": "Model specifications"
                    },
                    {
                        "question": "What is the average response time?",
                        "answer": "< 2 seconds",
                        "context": "Performance metrics"
                    }
                ]
            }
        ]
    }
    
    return your_dataset

def process_your_50_projects(api_key: str, dataset_path: str):
    """
    Process your actual 50 AI projects dataset
    """
    
    print("🚀 Processing your 50 AI projects with OpenAI embeddings...")
    
    # Initialize the processor with your API key
    processor = OpenAIEmbeddingProcessor(
        api_key=api_key,
        model="text-embedding-3-large"  # Best performance
    )
    
    # Create embeddings dataset
    print("📊 Creating embeddings (this may take a few minutes for 50 projects)...")
    embeddings_dataset = processor.create_embeddings_dataset(dataset_path)
    
    # Save the embeddings dataset
    output_file = "your_ai_projects_embeddings.json"
    with open(output_file, 'w') as f:
        json.dump(embeddings_dataset, f, indent=2)
    
    print(f"✅ Embeddings saved to {output_file}")
    
    # Display statistics
    metadata = embeddings_dataset['metadata']
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total Q&A pairs: {metadata['total_qa_pairs']}")
    print(f"   Total document chunks: {metadata['total_document_chunks']}")
    print(f"   Embedding model: {metadata['model']}")
    print(f"   Embedding dimensions: {metadata['embedding_dimension']}")
    
    return embeddings_dataset

def setup_retrieval_system(embeddings_dataset: dict, api_key: str):
    """
    Set up the retrieval system for querying your projects
    """
    
    print("\n🔍 Setting up retrieval system...")
    
    # Initialize retriever
    retriever = OpenAIRetriever(embeddings_dataset, api_key)
    
    print("✅ Retrieval system ready!")
    
    return retriever

def demo_queries_for_your_projects(retriever):
    """
    Demo queries specific to your AI projects use case
    """
    
    # Your typical question patterns
    demo_queries = [
        # Boolean questions
        "Is real-time processing supported?",
        "Does the model use GPU acceleration?", 
        "Is the system cloud-deployed?",
        "Does it support multi-language input?",
        
        # Time period questions (your specific example)
        "What is the time period for data collection?",
        "How long was the training duration?",
        "What is the data collection timeframe?",
        
        # Categorical/Dropdown style
        "What framework is used?",
        "What model architecture is implemented?",
        "What is the deployment method?",
        "What programming language is used?",
        
        # Numerical questions  
        "What is the model accuracy?",
        "How many parameters does the model have?",
        "What is the dataset size?",
        "What is the batch size used?",
        
        # Performance questions
        "What are the performance metrics?",
        "What optimization techniques were used?",
        "What are the system requirements?",
    ]
    
    print("\n🤖 Demo: Querying your AI projects...")
    print("=" * 60)
    
    for query in demo_queries:
        print(f"\n❓ Query: {query}")
        
        # Search for answer
        result = retriever.answer_question(query)
        
        print(f"✅ Answer: {result['answer']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print(f"🏷️  Question Type: {result['question_type']}")
        print(f"📁 Source Project: {result['project_id']}")
        
        # Show if it's from Q&A pair or document
        if result['source'] == 'qa_pair':
            print("📋 Source: Direct Q&A match")
        else:
            print("📄 Source: Document context")
        
        print("-" * 50)

def search_by_project(retriever, project_keywords: str):
    """
    Search for information about a specific project
    """
    
    print(f"\n🔍 Searching for project: '{project_keywords}'")
    print("=" * 40)
    
    # Search documents related to the project
    doc_results = retriever.search_documents(project_keywords, top_k=3)
    
    if doc_results:
        for i, result in enumerate(doc_results, 1):
            print(f"\n📄 Result {i} (Score: {result['similarity_score']:.3f})")
            print(f"Project: {result['project_id']}")
            print(f"Content: {result['text'][:200]}...")
    else:
        print("No matching projects found.")

def filter_by_question_type(retriever, question_type: str):
    """
    Find all questions of a specific type
    """
    
    print(f"\n🏷️  Finding all {question_type} questions...")
    
    # Get all Q&A data
    qa_data = retriever.dataset['qa_data']
    
    # Filter by question type
    filtered_questions = [
        qa for qa in qa_data 
        if qa['question_type'] == question_type
    ]
    
    print(f"Found {len(filtered_questions)} {question_type} questions:")
    
    for qa in filtered_questions[:5]:  # Show first 5
        print(f"  Q: {qa['question']}")
        print(f"  A: {qa['answer']}")
        print(f"  Project: {qa['project_id']}")
        print()

def main():
    """
    Main function to process and query your AI projects
    """
    
    # 🔑 Set your OpenAI API key here
    API_KEY = "your-openai-api-key-here"
    
    if API_KEY == "your-openai-api-key-here":
        print("❌ Please set your OpenAI API key in the script!")
        print("Get your key from: https://platform.openai.com/api-keys")
        return
    
    # Step 1: Create sample dataset (replace with your actual data loading)
    print("📝 Creating sample dataset...")
    sample_data = create_your_dataset_format()
    
    # Save sample data
    with open('your_ai_projects.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # Step 2: Process your dataset with OpenAI embeddings
    embeddings_dataset = process_your_50_projects(API_KEY, 'your_ai_projects.json')
    
    # Step 3: Set up retrieval system
    retriever = setup_retrieval_system(embeddings_dataset, API_KEY)
    
    # Step 4: Demo various queries
    demo_queries_for_your_projects(retriever)
    
    # Step 5: Advanced search examples
    search_by_project(retriever, "medical diagnosis computer vision")
    
    # Step 6: Filter by question types
    filter_by_question_type(retriever, "boolean")
    filter_by_question_type(retriever, "categorical")
    
    print("\n🎉 Demo completed!")
    print("\nNext steps:")
    print("1. Replace sample data with your actual 50 projects")
    print("2. Adjust question types and processing as needed")
    print("3. Build your web interface or API on top of this")

# Cost estimation function
def estimate_costs(num_projects: int, avg_qa_pairs_per_project: int, avg_readme_length: int):
    """
    Estimate OpenAI API costs for your dataset
    """
    
    print(f"\n💰 Cost Estimation for {num_projects} projects:")
    print("-" * 40)
    
    # Rough token estimation (1 token ≈ 4 characters)
    avg_qa_tokens = 100  # Average for Q&A pair with optimization
    avg_doc_tokens = avg_readme_length // 4  # README converted to tokens
    
    total_qa_tokens = num_projects * avg_qa_pairs_per_project * avg_qa_tokens
    total_doc_tokens = num_projects * avg_doc_tokens
    total_tokens = total_qa_tokens + total_doc_tokens
    
    # Current pricing for text-embedding-3-large: $0.00013 per 1K tokens
    cost_per_1k_tokens = 0.00013
    total_cost = (total_tokens / 1000) * cost_per_1k_tokens
    
    print(f"Estimated tokens:")
    print(f"  Q&A pairs: {total_qa_tokens:,} tokens")
    print(f"  Documents: {total_doc_tokens:,} tokens") 
    print(f"  Total: {total_tokens:,} tokens")
    print(f"\nEstimated cost: ${total_cost:.2f}")
    print(f"Per project: ${total_cost/num_projects:.3f}")

if __name__ == "__main__":
    # First, estimate costs for your dataset
    estimate_costs(
        num_projects=50,
        avg_qa_pairs_per_project=10,  # Adjust based on your data
        avg_readme_length=2000        # Adjust based on your README sizes
    )
    
    # Then run the main demo
    main()
