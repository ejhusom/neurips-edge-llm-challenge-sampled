import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass
import utils

@dataclass
class ModelMetrics:
    """Class to store aggregated metrics for a model-dataset combination"""
    model_name: str
    dataset_name: str
    mean_energy: float
    std_energy: float
    median_energy: float
    minimum_energy: float
    maximum_energy: float
    q1_energy: float
    q3_energy: float
    iqr_energy: float
    mean_energy_per_token: float
    std_energy_per_token: float
    median_energy_per_token: float
    minimum_energy_per_token: float
    maximum_energy_per_token: float
    q1_energy_per_token: float
    q3_energy_per_token: float
    iqr_energy_per_token: float
    accuracy: float
    total_samples: int
    quantization_level: str

class EnergyAnalyzer:
    def __init__(self, file_paths: List[str]):
        """Initialize with list of CSV file paths"""
        self.file_paths = file_paths
        self.metrics: Dict[Tuple[str, str], ModelMetrics] = {}
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self._load_and_process_data()
    
    def _extract_quantization_level(self, model_name: str) -> str:
        """Extract quantization level from model name"""
        quantization_level = "_".join(model_name.split('_')[3:])
        # if 'q' in model_name:
        #     parts = model_name.split('_')
        #     for i, part in enumerate(parts):
        #         if part.startswith('q'):
        #             return "_".join(parts[i:])
        # return 'none'
        return quantization_level
    
    def _load_and_process_data(self):
        """Load all CSV files and compute basic metrics"""
        for file_path in self.file_paths:
            path = Path(file_path)
            if not path.exists():
                print(f"Warning: File {file_path} not found")
                continue
                
            df = pd.read_csv(file_path)
            dataset_name = utils.get_dataset_name(path.name)
            model_name = utils.get_model_name(path.name)
            
            # Compute energy per token
            df['energy_per_token'] = df['energy_consumption_joules'] / df['eval_count']
            
            # Store raw data
            self.raw_data[f"{dataset_name}_{model_name}"] = df
            
            # Compute metrics
            metrics = ModelMetrics(
                model_name=model_name,
                dataset_name=dataset_name,
                mean_energy=df['energy_consumption_joules'].mean(),
                std_energy=df['energy_consumption_joules'].std(),
                median_energy=df['energy_consumption_joules'].median(),
                minimum_energy=df['energy_consumption_joules'].min(),
                maximum_energy=df['energy_consumption_joules'].max(),
                q1_energy=df['energy_consumption_joules'].quantile(0.25),
                q3_energy=df['energy_consumption_joules'].quantile(0.75),
                iqr_energy=df['energy_consumption_joules'].quantile(0.75) - df['energy_consumption_joules'].quantile(0.25),
                mean_energy_per_token=df['energy_per_token'].mean(),
                std_energy_per_token=df['energy_per_token'].std(),
                median_energy_per_token=df['energy_per_token'].median(),
                minimum_energy_per_token=df['energy_per_token'].min(),
                maximum_energy_per_token=df['energy_per_token'].max(),
                q1_energy_per_token=df['energy_per_token'].quantile(0.25),
                q3_energy_per_token=df['energy_per_token'].quantile(0.75),
                iqr_energy_per_token=df['energy_per_token'].quantile(0.75) - df['energy_per_token'].quantile(0.25),
                accuracy=df['evaluation'].mean(),
                total_samples=len(df),
                quantization_level=self._extract_quantization_level(model_name)
            )
            
            self.metrics[(dataset_name, model_name)] = metrics
    
    def plot_energy_accuracy_tradeoff(self, dataset_name: str = None):
        """Plot energy consumption vs accuracy for all models"""
        metrics_list = []
        for (ds_name, model_name), metrics in self.metrics.items():
            if dataset_name is None or ds_name.lower() == dataset_name.lower():
                metrics_list.append({
                    'Model': model_name,
                    'Dataset': ds_name,
                    'Mean Energy (J)': metrics.mean_energy,
                    'Accuracy': metrics.accuracy,
                    'Quantization': metrics.quantization_level
                })
        
        df = pd.DataFrame(metrics_list)
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x='Mean Energy (J)',
            y='Accuracy',
            hue='Dataset' if dataset_name is None else 'Quantization',
            style='Quantization',
            s=100
        )
        
        # plt.xscale('log')
        plt.title('Energy-Accuracy Trade-off')
        plt.xlabel('Mean Energy Consumption (Joules)')
        plt.ylabel('Accuracy')
        
        # Add annotations for interesting points
        for _, row in df.iterrows():
            plt.annotate(
                row['Model'].split('_')[0],  # Use first part of model name
                (row['Mean Energy (J)'], row['Accuracy']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8
            )
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_energy_distribution(self, dataset_name: str = None):
        """Plot energy consumption distribution for each model"""
        plt.figure(figsize=(12, 6))
        
        data = []
        for experiment_name, df in self.raw_data.items():
            ds_name = utils.get_dataset_name(experiment_name)
            model_name = utils.get_model_name(experiment_name)
            if dataset_name is None or ds_name.lower() == dataset_name.lower():
                data.extend([{
                    'Model': model_name,
                    'Dataset': ds_name,
                    'Energy (J)': energy
                } for energy in df['energy_consumption_joules']])
        
        df = pd.DataFrame(data)
        
        sns.boxplot(
            data=df,
            x='Model',
            y='Energy (J)',
            hue='Dataset' if dataset_name is None else None
        )
        
        plt.xticks(rotation=45, ha='right')
        # plt.yscale('log')
        plt.title('Energy Consumption Distribution by Model')
        plt.tight_layout()
        return plt.gcf()
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate a summary table of all metrics"""
        rows = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            rows.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Mean Energy (J)': metrics.mean_energy,
                'Std Energy (J)': metrics.std_energy,
                'Mean Energy per Token (J)': metrics.mean_energy_per_token,
                'Std Energy per Token (J)': metrics.std_energy_per_token,
                'Accuracy': metrics.accuracy,
                'Samples': metrics.total_samples,
                'Quantization': metrics.quantization_level
            })
        
        return pd.DataFrame(rows)
    
    def analyze_token_length_impact(self, dataset_name: str = None):
        """Analyze relationship between token length and energy consumption"""
        plt.figure(figsize=(12, 8))
        
        for experiment_name, df in self.raw_data.items():
            ds_name = utils.get_dataset_name(experiment_name)
            model_name = utils.get_model_name(experiment_name)
            if dataset_name is None or ds_name.lower() == dataset_name.lower():
                plt.scatter(
                    df['eval_count'],
                    df['energy_consumption_joules'],
                    alpha=0.5,
                    label=f"{model_name} ({ds_name})"
                )
        
        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('Token Count')
        plt.ylabel('Energy Consumption (Joules)')
        plt.title('Token Length vs Energy Consumption')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description='Analyze LLM energy consumption data')
    parser.add_argument('files', nargs='+', help='CSV files to analyze')
    parser.add_argument('--dataset', help='Filter analysis to specific dataset')
    parser.add_argument('--output-dir', default='outputs', help='Directory for output files')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = EnergyAnalyzer(args.files)
    
    # Generate and save plots
    plots = {
        'energy_accuracy': analyzer.plot_energy_accuracy_tradeoff(args.dataset),
        'energy_distribution': analyzer.plot_energy_distribution(args.dataset),
        'token_impact': analyzer.analyze_token_length_impact(args.dataset)
    }
    
    for name, fig in plots.items():
        dataset_name = args.dataset if args.dataset is not None else 'all'
        fig.savefig(Path(args.output_dir) / f"{name}_{dataset_name}.pdf", bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    # Generate and save summary table
    summary_df = analyzer.generate_summary_table()
    summary_df.to_csv(Path(args.output_dir) / "summary_metrics.csv", index=False)
    
    # Print some basic statistics
    print("\nSummary Statistics:")
    print(f"Total number of models analyzed: {len(summary_df['Model'].unique())}")
    print(f"Datasets analyzed: {', '.join(summary_df['Dataset'].unique())}")
    print("\nTop 5 most energy-efficient models (by mean energy per token):")
    print(summary_df.nsmallest(5, 'Mean Energy per Token (J)')[['Dataset', 'Model', 'Mean Energy per Token (J)', 'Accuracy']])

if __name__ == "__main__":
    main()