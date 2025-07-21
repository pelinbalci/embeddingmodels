# Sample AI Projects Dataset Creation

import json
import os
from typing import List, Dict

# Sample README files for 3 AI projects
sample_readmes = {
    "project_1": """
# Image Classification for Medical Diagnosis

## Project Overview
This project develops a deep learning model for automated medical image classification to assist radiologists in diagnosing chest X-rays.

## Objective
The main objective is to create an AI system that can accurately classify chest X-ray images into normal, pneumonia, and COVID-19 cases with at least 95% accuracy.

## AI Model Used
- **Primary Model**: ResNet-50 with transfer learning
- **Framework**: PyTorch
- **Architecture**: Convolutional Neural Network (CNN)
- **Pre-trained weights**: ImageNet

## Data Source
- **Dataset**: ChestX-ray14 dataset from NIH
- **Additional data**: COVID-19 Radiography Database
- **Data format**: DICOM and PNG images
- **Resolution**: 1024x1024 pixels

## Data Size
- **Training set**: 45,000 images
- **Validation set**: 8,000 images  
- **Test set**: 7,000 images
- **Total size**: ~15 GB

## Technical Implementation
- Data preprocessing with normalization and augmentation
- Transfer learning from ImageNet weights
- Custom classification head with 3 output classes
- Training for 50 epochs with early stopping

## Performance Metrics
- **Accuracy**: 96.2%
- **Precision**: 95.8%
- **Recall**: 96.1%
- **F1-Score**: 95.9%

## Lessons Learned
1. **Data Quality**: High-quality, diverse training data is crucial for medical AI applications
2. **Transfer Learning**: Pre-trained models significantly reduce training time and improve performance
3. **Class Imbalance**: Proper handling of imbalanced medical datasets is essential
4. **Validation Strategy**: Stratified cross-validation ensures robust model evaluation
5. **Interpretability**: Model interpretability is critical for medical applications

## Future Work
- Implement GradCAM for model interpretability
- Expand to additional chest conditions
- Deploy as web application for clinical use
""",

    "project_2": """
# Natural Language Processing for Customer Sentiment Analysis

## Project Overview
This project implements a transformer-based sentiment analysis system for processing customer reviews and feedback across multiple e-commerce platforms.

## Objective
Build an automated sentiment classification system that can process customer reviews in real-time and categorize them into positive, negative, and neutral sentiments with high accuracy.

## AI Model Used
- **Primary Model**: BERT (Bidirectional Encoder Representations from Transformers)
- **Specific variant**: DistilBERT for faster inference
- **Framework**: Hugging Face Transformers, PyTorch
- **Architecture**: Transformer-based encoder

## Data Source
- **Primary**: Amazon Product Reviews dataset
- **Secondary**: Yelp Reviews dataset
- **Additional**: Twitter sentiment data
- **Format**: JSON and CSV files
- **Languages**: English (primary), Spanish (secondary)

## Data Size
- **Training samples**: 500,000 reviews
- **Validation samples**: 75,000 reviews
- **Test samples**: 100,000 reviews
- **Total storage**: ~2.5 GB text data

## Technical Implementation
- Text preprocessing with tokenization and normalization
- Fine-tuning DistilBERT on domain-specific data
- Custom classification head for 3-class sentiment
- Batch processing with dynamic padding

## Performance Metrics
- **Accuracy**: 89.3%
- **Macro F1-Score**: 88.7%
- **Inference time**: 45ms per review
- **Throughput**: 1,200 reviews/second

## Lessons Learned
1. **Domain Adaptation**: Fine-tuning on domain-specific data significantly improves performance
2. **Model Efficiency**: DistilBERT provides good balance between accuracy and speed
3. **Text Preprocessing**: Proper handling of informal text and emojis is crucial
4. **Multilingual Challenges**: Language-specific preprocessing improves cross-lingual performance
5. **Real-time Processing**: Optimization for inference speed is essential for production deployment

## Future Work
- Implement multi-label emotion classification
- Add support for additional languages
- Deploy as microservice API
- Integrate with business intelligence dashboards
""",

    "project_3": """
# Reinforcement Learning for Autonomous Trading Bot

## Project Overview
This project develops an autonomous trading bot using deep reinforcement learning to make profitable trading decisions in cryptocurrency markets.

## Objective
Create an AI agent that can autonomously trade cryptocurrencies by learning optimal trading strategies through interaction with historical and simulated market environments.

## AI Model Used
- **Primary Model**: Deep Q-Network (DQN) with experience replay
- **Enhancement**: Double DQN with Dueling Network architecture
- **Framework**: TensorFlow and Stable-Baselines3
- **Architecture**: Deep Neural Network with LSTM layers for sequential data

## Data Source
- **Primary**: Binance API for real-time and historical data
- **Secondary**: CoinGecko API for market indicators
- **Features**: OHLCV data, technical indicators, market sentiment
- **Timeframe**: 1-minute to 1-hour intervals
- **Assets**: Bitcoin, Ethereum, and 10 major altcoins

## Data Size
- **Historical data**: 3 years of minute-level data
- **Training episodes**: 100,000 simulated trading sessions
- **Feature dimensions**: 50+ technical and market indicators
- **Storage requirement**: ~8 GB

## Technical Implementation
- Custom trading environment using OpenAI Gym
- State space: normalized price data and technical indicators
- Action space: buy, sell, hold with position sizing
- Reward function: risk-adjusted returns with transaction costs
- Training with experience replay buffer of 100,000 transitions

## Performance Metrics
- **Sharpe Ratio**: 1.85
- **Maximum Drawdown**: 12.3%
- **Win Rate**: 62.4%
- **Annual Return**: 34.7% (backtesting)
- **Training time**: 48 hours on GPU

## Lessons Learned
1. **Market Dynamics**: Cryptocurrency markets are highly volatile and require robust risk management
2. **Feature Engineering**: Technical indicators and market sentiment features are crucial
3. **Overfitting Risk**: Extensive validation on out-of-sample data prevents overfitting to historical patterns
4. **Transaction Costs**: Realistic modeling of fees and slippage significantly impacts performance
5. **Continuous Learning**: Markets evolve, requiring periodic model retraining

## Future Work
- Implement multi-agent system for portfolio management
- Add sentiment analysis from social media and news
- Develop risk management modules
- Test with real money in paper trading environment
"""
}

# Generate Q&A pairs for each project
def generate_qa_pairs(project_id: str, readme_content: str) -> List[Dict]:
    """Generate Q&A pairs based on README content"""
    
    qa_pairs = []
    
    # Extract project name from README title
    lines = readme_content.strip().split('\n')
    project_name = lines[1] if len(lines) > 1 else f"Project {project_id}"
    
    # Common questions and their answers based on README structure
    if "Image Classification for Medical Diagnosis" in readme_content:
        qa_pairs = [
            {
                "question": "What is the project name?",
                "answer": "Image Classification for Medical Diagnosis",
                "context": "Medical AI project for chest X-ray analysis"
            },
            {
                "question": "What is the objective?",
                "answer": "The main objective is to create an AI system that can accurately classify chest X-ray images into normal, pneumonia, and COVID-19 cases with at least 95% accuracy.",
                "context": "Medical diagnosis automation using deep learning"
            },
            {
                "question": "Which AI model is used?",
                "answer": "ResNet-50 with transfer learning using PyTorch framework. The architecture is a Convolutional Neural Network (CNN) with pre-trained ImageNet weights.",
                "context": "Deep learning model for image classification"
            },
            {
                "question": "What is the data source?",
                "answer": "ChestX-ray14 dataset from NIH and COVID-19 Radiography Database. Data format includes DICOM and PNG images with 1024x1024 pixel resolution.",
                "context": "Medical imaging datasets for training"
            },
            {
                "question": "What is the data size?",
                "answer": "Training set: 45,000 images, Validation set: 8,000 images, Test set: 7,000 images. Total size is approximately 15 GB.",
                "context": "Large-scale medical image dataset"
            },
            {
                "question": "What are the lessons learned?",
                "answer": "Key lessons include: 1) High-quality, diverse training data is crucial for medical AI, 2) Transfer learning significantly reduces training time, 3) Proper handling of imbalanced datasets is essential, 4) Stratified cross-validation ensures robust evaluation, 5) Model interpretability is critical for medical applications.",
                "context": "Insights from medical AI development"
            }
        ]
    
    elif "Natural Language Processing for Customer Sentiment Analysis" in readme_content:
        qa_pairs = [
            {
                "question": "What is the project name?",
                "answer": "Natural Language Processing for Customer Sentiment Analysis",
                "context": "NLP project for analyzing customer feedback"
            },
            {
                "question": "What is the objective?",
                "answer": "Build an automated sentiment classification system that can process customer reviews in real-time and categorize them into positive, negative, and neutral sentiments with high accuracy.",
                "context": "Real-time sentiment analysis automation"
            },
            {
                "question": "Which AI model is used?",
                "answer": "BERT (Bidirectional Encoder Representations from Transformers), specifically DistilBERT for faster inference. Uses Hugging Face Transformers and PyTorch framework with transformer-based encoder architecture.",
                "context": "Transformer model for text classification"
            },
            {
                "question": "What is the data source?",
                "answer": "Primary: Amazon Product Reviews dataset, Secondary: Yelp Reviews dataset, Additional: Twitter sentiment data. Format includes JSON and CSV files in English and Spanish languages.",
                "context": "Multi-platform customer review data"
            },
            {
                "question": "What is the data size?",
                "answer": "Training samples: 500,000 reviews, Validation samples: 75,000 reviews, Test samples: 100,000 reviews. Total storage: approximately 2.5 GB of text data.",
                "context": "Large-scale text dataset for sentiment analysis"
            },
            {
                "question": "What are the lessons learned?",
                "answer": "Key insights: 1) Fine-tuning on domain-specific data significantly improves performance, 2) DistilBERT provides good balance between accuracy and speed, 3) Proper handling of informal text and emojis is crucial, 4) Language-specific preprocessing improves cross-lingual performance, 5) Optimization for inference speed is essential for production deployment.",
                "context": "NLP development best practices"
            }
        ]
    
    elif "Reinforcement Learning for Autonomous Trading Bot" in readme_content:
        qa_pairs = [
            {
                "question": "What is the project name?",
                "answer": "Reinforcement Learning for Autonomous Trading Bot",
                "context": "AI trading system using reinforcement learning"
            },
            {
                "question": "What is the objective?",
                "answer": "Create an AI agent that can autonomously trade cryptocurrencies by learning optimal trading strategies through interaction with historical and simulated market environments.",
                "context": "Autonomous cryptocurrency trading"
            },
            {
                "question": "Which AI model is used?",
                "answer": "Deep Q-Network (DQN) with experience replay, enhanced with Double DQN and Dueling Network architecture. Uses TensorFlow and Stable-Baselines3 framework with Deep Neural Network including LSTM layers for sequential data.",
                "context": "Reinforcement learning for financial markets"
            },
            {
                "question": "What is the data source?",
                "answer": "Primary: Binance API for real-time and historical data, Secondary: CoinGecko API for market indicators. Features include OHLCV data, technical indicators, and market sentiment for Bitcoin, Ethereum, and 10 major altcoins.",
                "context": "Cryptocurrency market data"
            },
            {
                "question": "What is the data size?",
                "answer": "Historical data: 3 years of minute-level data, Training episodes: 100,000 simulated trading sessions, Feature dimensions: 50+ technical and market indicators, Storage requirement: approximately 8 GB.",
                "context": "High-frequency financial data"
            },
            {
                "question": "What are the lessons learned?",
                "answer": "Key learnings: 1) Cryptocurrency markets are highly volatile requiring robust risk management, 2) Technical indicators and market sentiment features are crucial, 3) Extensive validation on out-of-sample data prevents overfitting, 4) Realistic modeling of fees and slippage significantly impacts performance, 5) Markets evolve requiring periodic model retraining.",
                "context": "Financial AI development insights"
            }
        ]
    
    # Add project_id and chunk information to each QA pair
    for i, qa in enumerate(qa_pairs):
        qa['project_id'] = project_id
        qa['chunk_id'] = f"{project_id}_chunk_{i}"
        qa['qa_id'] = f"{project_id}_qa_{i}"
    
    return qa_pairs

# Create the complete dataset
def create_dataset():
    """Create the complete dataset with projects, readmes, and QA pairs"""
    
    dataset = {
        "projects": [],
        "metadata": {
            "total_projects": len(sample_readmes),
            "total_qa_pairs": 0,
            "creation_date": "2025-01-21",
            "description": "Sample AI projects dataset for embedding model fine-tuning"
        }
    }
    
    total_qa_pairs = 0
    
    for project_id, readme_content in sample_readmes.items():
        qa_pairs = generate_qa_pairs(project_id, readme_content)
        total_qa_pairs += len(qa_pairs)
        
        project_data = {
            "project_id": project_id,
            "readme_content": readme_content,
            "qa_pairs": qa_pairs,
            "stats": {
                "qa_count": len(qa_pairs),
                "readme_length": len(readme_content),
                "readme_lines": len(readme_content.split('\n'))
            }
        }
        
        dataset["projects"].append(project_data)
    
    dataset["metadata"]["total_qa_pairs"] = total_qa_pairs
    
    return dataset

# Generate and save the dataset
if __name__ == "__main__":
    dataset = create_dataset()
    
    # Save as JSON
    with open("ai_projects_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print("Dataset created successfully!")
    print(f"Total projects: {dataset['metadata']['total_projects']}")
    print(f"Total Q&A pairs: {dataset['metadata']['total_qa_pairs']}")
    
    # Print sample data
    print("\n--- Sample Project ---")
    sample_project = dataset['projects'][0]
    print(f"Project ID: {sample_project['project_id']}")
    print(f"Q&A pairs count: {sample_project['stats']['qa_count']}")
    print("\n--- Sample Q&A Pair ---")
    sample_qa = sample_project['qa_pairs'][0]
    print(f"Question: {sample_qa['question']}")
    print(f"Answer: {sample_qa['answer'][:100]}...")
