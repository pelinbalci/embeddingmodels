"""
Benchmark script to compare ColTrast vs OpenAI embeddings on your AI projects data
"""

import json
import time
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import openai

class EmbeddingBenchmark:
    """
    Comprehensive benchmark comparing different embedding approaches
    """
    
    def __init__(self, dataset_path: str, openai_api_key: str = None):
        self.dataset_path = dataset_path
        self.openai_api_key = openai_api_key
        
        # Load test dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Initialize models
        self.coltrast_model = None
        self.openai_client = None
        
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Benchmark results storage
        self.results = {}
    
    def setup_coltrast(self, model_path: str):
        """Setup ColTrast model for benchmarking"""
        try:
            from inference_script import ColTrastInference
            self.coltrast_model = ColTrastInference(model_path)
            self.coltrast_model.index_documents(self.dataset_path)
            print("✅ ColTrast model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading ColTrast: {e}")
    
    def get_openai_embedding(self, text: str, model: str = "text-embedding-3-large") -> np.ndarray:
        """Get OpenAI embedding for text"""
        if not self.openai_client:
            raise ValueError("OpenAI API key not provided")
        
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=model
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error getting OpenAI embedding: {e}")
            return np.zeros(1536)  # Default embedding size
    
    def create_test_cases(self) -> List[Dict]:
        """Create test cases from the dataset"""
        test_cases = []
        
        for project in self.data['projects']:
            project_id = project['project_id']
            
            for qa in project['qa_pairs']:
                test_case = {
                    'question': qa['question'],
                    'correct_answer': qa['answer'],
                    'correct_project': project_id,
                    'context': qa.get('context', ''),
                    'qa_id': qa.get('qa_id', '')
                }
                test_cases.append(test_case)
        
        return test_cases
    
    def evaluate_retrieval_accuracy(self, test_cases: List[Dict], model_name: str, 
                                  top_k: int = 5) -> Dict:
        """Evaluate retrieval accuracy for a model"""
        
        correct_retrievals = 0
        total_cases = len(test_cases)
        retrieval_ranks = []
        similarities = []
        response_times = []
        
        print(f"\n🔍 Testing {model_name} on {total_cases} test cases...")
        
        for i, test_case in enumerate(test_cases):
            question = test_case['question']
            correct_project = test_case['correct_project']
            
            start_time = time.time()
            
            try:
                if model_name == "ColTrast":
                    # Use ColTrast model
                    if self.coltrast_model:
                        results = self.coltrast_model.search(question, top_k=top_k)
                        retrieved_projects = [r['project_id'] for r in results]
                        similarities.append(results[0]['similarity_score'] if results else 0)
                    else:
                        retrieved_projects = []
                        similarities.append(0)
                
                elif model_name.startswith("OpenAI"):
                    # Use OpenAI embeddings with simple retrieval
                    results = self._openai_retrieve(question, top_k=top_k)
                    retrieved_projects = [r['project_id'] for r in results]
                    similarities.append(results[0]['similarity_score'] if results else 0)
                
                else:
                    retrieved_projects = []
                    similarities.append(0)
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                # Check if correct project is in top-k results
                if correct_project in retrieved_projects:
                    correct_retrievals += 1
                    rank = retrieved_projects.index(correct_project) + 1
                    retrieval_ranks.append(rank)
                else:
                    retrieval_ranks.append(top_k + 1)  # Penalty for not found
                
            except Exception as e:
                print(f"Error processing case {i}: {e}")
                response_times.append(0)
                retrieval_ranks.append(top_k + 1)
                similarities.append(0)
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{total_cases} cases...")
        
        # Calculate metrics
        accuracy = correct_retrievals / total_cases
        avg_rank = np.mean(retrieval_ranks)
        avg_similarity = np.mean(similarities)
        avg_response_time = np.mean(response_times)
        
        return {
            'accuracy': accuracy,
            'avg_rank': avg_rank,
            'avg_similarity': avg_similarity,
            'avg_response_time': avg_response_time,
            'total_cases': total_cases,
            'correct_retrievals': correct_retrievals
        }
    
    def _openai_retrieve(self, question: str, top_k: int = 5) -> List[Dict]:
        """Simple retrieval using OpenAI embeddings"""
        if not self.openai_client:
            return []
        
        # Get question embedding
        question_emb = self.get_openai_embedding(question)
        
        # Get all project content embeddings (simplified approach)
        project_similarities = []
        
        for project in self.data['projects']:
            project_content = project['readme_content'][:1000]  # Truncate for API limits
            
            try:
                content_emb = self.get_openai_embedding(project_content)
                similarity = cosine_similarity([question_emb], [content_emb])[0][0]
                
                project_similarities.append({
                    'project_id': project['project_id'],
                    'similarity_score': similarity
                })
            except:
                project_similarities.append({
                    'project_id': project['project_id'],
                    'similarity_score': 0
                })
        
        # Sort by similarity and return top-k
        project_similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return project_similarities[:top_k]
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive benchmark comparing all models"""
        print("🚀 Starting Comprehensive Embedding Benchmark")
        print("=" * 60)
        
        test_cases = self.create_test_cases()
        models_to_test = []
        
        # Add ColTrast if available
        if self.coltrast_model:
            models_to_test.append("ColTrast")
        
        # Add OpenAI if available
        if self.openai_client:
            models_to_test.append("OpenAI-text-embedding-3-large")
        
        benchmark_results = {}
        
        for model_name in models_to_test:
            print(f"\n📊 Benchmarking {model_name}")
            print("-" * 40)
            
            results = self.evaluate_retrieval_accuracy(test_cases, model_name)
            benchmark_results[model_name] = results
            
            # Print results
            print(f"✅ {model_name} Results:")
            print(f"   Accuracy: {results['accuracy']:.1%}")
            print(f"   Average Rank: {results['avg_rank']:.2f}")
            print(f"   Average Similarity: {results['avg_similarity']:.3f}")
            print(f"   Average Response Time: {results['avg_response_time']:.3f}s")
        
        self.results = benchmark_results
        return benchmark_results
    
    def analyze_query_types(self) -> Dict:
        """Analyze performance by query type"""
        test_cases = self.create_test_cases()
        
        query_categories = {
            'model_related': ['model', 'algorithm', 'architecture', 'network'],
            'data_related': ['data', 'dataset', 'source', 'size'],
            'performance_related': ['accuracy', 'performance', 'metric', 'score'],
            'general': ['objective', 'purpose', 'goal', 'name']
        }
        
        categorized_results = {}
        
        for category, keywords in query_categories.items():
            category_cases = []
            
            for case in test_cases:
                question_lower = case['question'].lower()
                if any(keyword in question_lower for keyword in keywords):
                    category_cases.append(case)
            
            if category_cases:
                print(f"\n📋 Testing {category} queries ({len(category_cases)} cases)")
                
                category_results = {}
                for model_name in self.results.keys():
                    results = self.evaluate_retrieval_accuracy(category_cases, model_name)
                    category_results[model_name] = results
                
                categorized_results[category] = category_results
        
        return categorized_results
    
    def plot_comparison(self):
        """Create comparison plots"""
        if not self.results:
            print("No benchmark results available. Run benchmark first.")
            return
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'avg_response_time', 'avg_similarity']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=['#1f77b4', '#ff7f0e'])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('embedding_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report"""
        if not self.results:
            return "No benchmark results available."
        
        report = []
        report.append("# Embedding Models Benchmark Report")
        report.append("=" * 50)
        report.append("")
        
        # Overall Results
        report.append("## Overall Performance")
        report.append("")
        
        for model_name, results in self.results.items():
            report.append(f"### {model_name}")
            report.append(f"- **Accuracy**: {results['accuracy']:.1%}")
            report.append(f"- **Average Rank**: {results['avg_rank']:.2f}")
            report.append(f"- **Average Similarity**: {results['avg_similarity']:.3f}")
            report.append(f"- **Average Response Time**: {results['avg_response_time']:.3f}s")
            report.append(f"- **Successful Retrievals**: {results['correct_retrievals']}/{results['total_cases']}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        fastest_model = min(self.results.items(), key=lambda x: x[1]['avg_response_time'])
        
        report.append(f"- **Best Accuracy**: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.1%})")
        report.append(f"- **Fastest Response**: {fastest_model[0]} ({fastest_model[1]['avg_response_time']:.3f}s)")
        report.append("")
        
        if len(self.results) >= 2:
            models = list(self.results.keys())
            model1, model2 = models[0], models[1]
            acc_diff = self.results[model1]['accuracy'] - self.results[model2]['accuracy']
            time_diff = self.results[model2]['avg_response_time'] - self.results[model1]['avg_response_time']
            
            report.append(f"- **Accuracy Difference**: {model1} vs {model2}: {acc_diff:+.1%}")
            report.append(f"- **Speed Difference**: {model1} vs {model2}: {time_diff:+.3f}s")
        
        return "\n".join(report)

# Example usage
def main():
    """Main benchmarking function"""
    
    # Configuration
    DATASET_PATH = "ai_projects_dataset.json"
    COLTRAST_MODEL_PATH = "coltrast_model.pth"
    OPENAI_API_KEY = None  # Set your OpenAI API key here
    
    print("🔬 Embedding Models Benchmark")
    print("=" * 40)
    
    # Initialize benchmark
    benchmark = EmbeddingBenchmark(DATASET_PATH, OPENAI_API_KEY)
    
    # Setup models
    try:
        benchmark.setup_coltrast(COLTRAST_MODEL_PATH)
    except:
        print("⚠️  ColTrast model not available. Train the model first.")
    
    if not OPENAI_API_KEY:
        print("⚠️  OpenAI API key not provided. Set OPENAI_API_KEY to compare.")
    
    # Run benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate report
    report = benchmark.generate_report()
    print("\n" + report)
    
    # Save report
    with open("benchmark_report.md", "w") as f:
        f.write(report)
    
    # Create plots
    benchmark.plot_comparison()
    
    # Analyze by query type
    print("\n🔍 Analyzing by Query Type...")
    query_analysis = benchmark.analyze_query_types()
    
    return results

if __name__ == "__main__":
    main()
