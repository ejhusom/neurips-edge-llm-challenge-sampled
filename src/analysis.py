import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass
import utils
import re

from adjustText import adjust_text

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
        self.colors = utils.get_colors()
        self.markers = utils.get_markers()
        self.dataset_markers = utils.get_markers_for_dataset()
        self._load_and_process_data()

    QUANTIZATION_ORDER = [
        'fp16',
        'q8_0',
        'q4_1',
        'q4_K_M',
        'q4_0',
        'q4_K_S',
        'q3_K_L',
        'q3_K_M',
        'q3_K_S'
    ]
    
    def _extract_quantization_level(self, model_name: str) -> str:
        """Extract quantization level from model name"""
        quantization_level = "_".join(model_name.split('_')[3:])
        return quantization_level
    
    def _get_quantization_rank(self, quant_level: str) -> int:
        """Get the rank of a quantization level for sorting"""
        try:
            return self.QUANTIZATION_ORDER.index(quant_level)
        except ValueError:
            return len(self.QUANTIZATION_ORDER)  # Put unknown levels at the end
    
    def _load_and_process_data(self):
        """Load all CSV files and compute basic metrics"""
        # First, collect and sort all data by model family and quantization level
        model_family_data = {}
        
        for file_path in self.file_paths:
            path = Path(file_path)
            if not path.exists():
                print(f"Warning: File {file_path} not found")
                continue
                
            df = pd.read_csv(file_path)
            dataset_name = utils.get_dataset_name(path.name)
            model_name = utils.get_model_name(path.name)
            
            # Extract model family (base model name without quantization)
            model_family = "_".join(model_name.split("_")[:3])
            quant_level = self._extract_quantization_level(model_name)
            
            # Create a tuple for sorting: (model_family, quantization_rank, dataset_name)
            sort_key = (model_family, self._get_quantization_rank(quant_level), dataset_name)
            
            model_family_data[sort_key] = {
                'df': df,
                'dataset_name': dataset_name,
                'model_name': model_name,
                'quant_level': quant_level
            }

        # Process the data in sorted order
        for sort_key, data in sorted(model_family_data.items()):
            df = data['df']
            dataset_name = data['dataset_name']
            model_name = data['model_name']
            
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

    def plot_energy_accuracy_tradeoff(self, dataset_name: str = None, log_scale=False):
        """Plot energy consumption vs accuracy for all models"""
        metrics_list = []
        for (ds_name, model_name), metrics in self.metrics.items():
            if dataset_name is None or ds_name.lower() == dataset_name.lower():
                model_family = "_".join(model_name.split('_')[:2])  # Extract model family (e.g., "gemma_2b")
                metrics_list.append({
                    'Model': model_name,
                    'Dataset': ds_name,
                    'Mean Energy (J)': metrics.mean_energy,
                    'Accuracy': metrics.accuracy,
                    'Quantization': metrics.quantization_level,
                    'Base Model': model_family
                })
        
        df = pd.DataFrame(metrics_list)
        
        # Find Pareto-optimal points
        def is_pareto_optimal(energy, accuracy, df):
            """
            Determine if a point is Pareto optimal by checking if it is dominated by any other point.
            
            A point P1(energy1, accuracy1) dominates another point P2(energy2, accuracy2) if:
            1. P1 has less or equal energy consumption (energy1 ≤ energy2)
            2. P1 has higher or equal accuracy (accuracy1 ≥ accuracy2)
            3. P1 is strictly better in at least one dimension 
            (either energy1 < energy2 OR accuracy1 > accuracy2)
            
            In our case, we want to minimize energy and maximize accuracy.
            """
            return not any(
                (df['Mean Energy (J)'] <= energy) & 
                (df['Accuracy'] >= accuracy) & 
                ((df['Mean Energy (J)'] < energy) | (df['Accuracy'] > accuracy))
            )
        
        df['Pareto_Optimal'] = [
            is_pareto_optimal(row['Mean Energy (J)'], row['Accuracy'], df)
            for _, row in df.iterrows()
        ]
        
        # Create plot
        plt.figure(figsize=(7, 4.5))
        
        # Plot non-Pareto points
        non_pareto = df[~df['Pareto_Optimal']]
        sns.scatterplot(
            data=non_pareto,
            x='Mean Energy (J)',
            y='Accuracy',
            hue='Dataset' if dataset_name is None else 'Base Model',
            style='Quantization',
            s=100,
            alpha=0.6,
            markers=self.markers,
            palette=self.colors,
        )
        
        # Plot Pareto points with larger markers
        pareto = df[df['Pareto_Optimal']]
        sns.scatterplot(
            data=pareto,
            x='Mean Energy (J)',
            y='Accuracy',
            hue='Dataset' if dataset_name is None else 'Base Model',
            style='Quantization',
            s=200,
            legend=False,
            markers=self.markers,
            palette=self.colors
        )
        
        # Connect Pareto frontier with a line
        pareto_sorted = pareto.sort_values('Mean Energy (J)')
        plt.plot(pareto_sorted['Mean Energy (J)'], 
                pareto_sorted['Accuracy'], 
                'k--', 
                alpha=0.5, 
                label='Pareto Frontier')
        
        if log_scale:
            plt.xscale('log')
        plt.title('Energy-Accuracy Trade-off')
        plt.xlabel('Mean Energy Consumption (Joules)')
        plt.ylabel('Accuracy')
        
        # Add annotations for Pareto-optimal points and high performers
        texts = []
        for _, row in df.iterrows():
            if row['Pareto_Optimal']:
                model_name = "_".join(row['Model'].split('_')[:2])  # Show model family and size
                texts.append(plt.text(
                    row['Mean Energy (J)'], row['Accuracy'],
                    f"{model_name} ({row['Quantization']})",
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                ))

        # Adjust text to avoid overlap
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
    
        
        # Add a text box with key statistics
        stats_text = (
            f"Number of models: {len(df)}\n"
            f"Pareto-optimal models: {len(pareto)}\n"
            f"Accuracy range: {df['Accuracy'].min():.2f}-{df['Accuracy'].max():.2f}\n"
            f"Energy range: {df['Mean Energy (J)'].min():.2f}-{df['Mean Energy (J)'].max():.2f}J"
        )
        plt.text(0.02, 0.02, stats_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                fontsize=8,
                verticalalignment='bottom')

        # Add legend outside the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("output/energy_accuracy_tradeoff.pdf")
        
        # Save average accuracy and energy consumption for Pareto optimal models to a file
        pareto_avg_accuracy = pareto['Accuracy'].mean()
        pareto_avg_energy = pareto['Mean Energy (J)'].mean()
        dataset_name_str = dataset_name if dataset_name else 'all_datasets'
        
        with open(f"output/pareto_optimal_results_{dataset_name_str}.txt", "w") as f:
            f.write(f"Results for dataset {dataset_name_str}:\n")
            f.write(f"Pareto-optimal models average accuracy: {pareto_avg_accuracy:.2f}\n")
            f.write(f"Pareto-optimal models average energy consumption: {pareto_avg_energy:.2f} J\n")
            
            # Save accuracy and energy consumption for each Pareto optimal model
            f.write("\nPareto-optimal models details:\n")
            for _, row in pareto.iterrows():
                f.write(f"Model: {row['Model']}, Accuracy: {row['Accuracy']:.2f}, Mean Energy (J): {row['Mean Energy (J)']:.2f}\n")

        return plt.gcf()
    
    def plot_energy_distribution(self, dataset_name: str = None, per_token=False, log_scale=False, subplots=False):
        """Plot energy consumption distribution for each model.
        
        If subplots is True, plot each dataset in a subplot with shared x-axis.
        """
        if per_token:
            column = 'energy_per_token'
        else:
            column = 'energy_consumption_joules'

        data = []
        for experiment_name, df in self.raw_data.items():
            ds_name = utils.get_dataset_name(experiment_name)
            model_name = utils.get_model_name(experiment_name)
            model_family = "_".join(model_name.split("_")[:3])  # Extract model family
            if dataset_name is None or ds_name.lower() == dataset_name.lower():
                data.extend([{
                    'Model': model_name,
                    'Dataset': ds_name,
                    'Energy (J)': energy,
                    'Base Model': model_family
                } for energy in df[column]])
        
        df = pd.DataFrame(data)

        if subplots:
            datasets = df['Dataset'].unique()
            num_datasets = len(datasets)
            fig, axes = plt.subplots(num_datasets, 1, figsize=(12, 3 * num_datasets), sharex=True)
            if num_datasets == 1:
                axes = [axes]

            for ax, dataset in zip(axes, datasets):
                sns.boxplot(
                    data=df[df['Dataset'] == dataset],
                    x='Model',
                    y='Energy (J)',
                    hue='Base Model',
                    ax=ax,
                    palette=self.colors,
                    legend=False
                )
                ax.set_title(f'Energy Consumption Distribution by Model for {dataset}')
                ax.set_xlabel('Model')
                ax.set_ylabel('Energy (J)')
                # ax.tick_params(axis='x', rotation=45)
                # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

                if log_scale:
                    ax.set_yscale('log')
            plt.tight_layout()

            # Create legend
            model_families = { "_".join(model.split("_")[:2]) for model in df['Model'].unique() }
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors.get(family, 'gray'), markersize=10) for family in model_families]
            labels = list(model_families)

            fig.legend(handles, labels, title="Base Model", loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
        else:
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=df,
                x='Model',
                y='Energy (J)',
                hue='Base Model',
                palette=self.colors
            )
            plt.xticks(rotation=45, ha='right')
            if log_scale:
                plt.yscale('log')
            plt.title('Energy Consumption Distribution by Model')
            plt.tight_layout()
        
        plt.savefig("output/energy_distribution.pdf")
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
    
    def analyze_quantization_impact(self, dataset_name: str = None):
        """Analyze the impact of quantization on energy consumption and accuracy"""
        # Group models by family (excluding quantization level)
        model_families = {}
        for (ds_name, model_name), metrics in self.metrics.items():
            if dataset_name is None or ds_name.lower() == dataset_name.lower():
                # Extract base model name (family) by removing quantization part
                base_model = "_".join(model_name.split("_")[:3])  # Adjust based on your naming convention
                if base_model not in model_families:
                    model_families[base_model] = []
                model_families[base_model].append({
                    'full_name': model_name,
                    'quantization': metrics.quantization_level,
                    'energy': metrics.mean_energy,
                    'accuracy': metrics.accuracy,
                    'dataset': ds_name
                })

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Process each model family
        for base_model, variants in model_families.items():
            if len(variants) < 2:  # Skip if only one quantization level
                continue
                
            # Find the baseline (non-quantized or highest precision) variant
            # Assuming the variant with the highest energy consumption is the baseline
            baseline = max(variants, key=lambda x: x['energy'])
            # Assume instead that the first available item from QUANTIZATION_ORDER is the baseline, keeping in mind that not all models may have this quantization level
            baseline = next((v for v in variants if v['quantization'] in self.QUANTIZATION_ORDER), None)

            
            # Calculate relative changes
            relative_metrics = []
            for variant in variants:
                relative_metrics.append({
                    'Model': base_model,
                    'Quantization': variant['quantization'],
                    'Energy Saving (%)': 100 * (1 - variant['energy'] / baseline['energy']),
                    'Accuracy Loss (pp)': 100 * (baseline['accuracy'] - variant['accuracy'])
                })
            
            # Plot energy savings
            ax1.scatter(
                [m['Quantization'] for m in relative_metrics],
                [m['Energy Saving (%)'] for m in relative_metrics],
                label=base_model,
                marker='o'
            )
            
            # Plot accuracy loss
            ax2.scatter(
                [m['Quantization'] for m in relative_metrics],
                [m['Accuracy Loss (pp)'] for m in relative_metrics],
                label=base_model,
                marker='o'
            )
            
            # Connect points with lines
            ax1.plot(
                [m['Quantization'] for m in relative_metrics],
                [m['Energy Saving (%)'] for m in relative_metrics],
                alpha=0.5
            )
            ax2.plot(
                [m['Quantization'] for m in relative_metrics],
                [m['Accuracy Loss (pp)'] for m in relative_metrics],
                alpha=0.5
            )

        # Customize plots
        ax1.set_title('Energy Savings by Quantization Level')
        ax1.set_xlabel('Quantization Level')
        ax1.set_ylabel('Energy Saving (%)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.set_title('Accuracy Impact by Quantization Level')
        ax2.set_xlabel('Quantization Level')
        ax2.set_ylabel('Accuracy Loss (percentage points)')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add legend
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
        
        plt.tight_layout()
        return fig

    def analyze_quantization_impact2(self, dataset_name: str = None, average: bool = True):
        """Analyze the impact of quantization on energy consumption and accuracy"""
        def get_quantization_rank(quant_level):
            try:
                return self.QUANTIZATION_ORDER.index(quant_level)
            except ValueError:
                return len(self.QUANTIZATION_ORDER)  # Put unknown levels at the end

        # Group models by family (excluding quantization level)
        model_families = {}
        for (ds_name, model_name), metrics in self.metrics.items():
            if dataset_name is None or ds_name.lower() == dataset_name.lower():
                # Extract base model name (family) by removing quantization part
                base_model = "_".join(model_name.split("_")[:3])  # Adjust based on your naming convention
                if base_model not in model_families:
                    model_families[base_model] = []
                model_families[base_model].append({
                    'full_name': model_name,
                    'quantization': metrics.quantization_level,
                    'energy': metrics.mean_energy,
                    'accuracy': metrics.accuracy,
                    'dataset': ds_name
                })

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Create a set to store all quantization levels actually present in the data
        all_quant_levels = set()
        
        # Process each model family
        for base_model, variants in model_families.items():
            if len(variants) < 2:  # Skip if only one quantization level
                continue
                
            # The baseline will be the "q3_K_S" quantization level
            baseline = next((v for v in variants if v['quantization'] == 'q3_K_S'), None)
            
            # Calculate relative changes
            relative_metrics = []
            for variant in variants:
                relative_metrics.append({
                    'Model': base_model,
                    'Quantization': variant['quantization'],
                    'Energy Saving (%)': 100 * (1 - variant['energy'] / baseline['energy']),
                    'Accuracy Loss (pp)': 100 * (baseline['accuracy'] - variant['accuracy']),
                    'Dataset': variant['dataset']
                })
                all_quant_levels.add(variant['quantization'])
            
            # Sort relative_metrics based on the defined order
            relative_metrics.sort(key=lambda x: get_quantization_rank(x['Quantization']))
            
            if average:
                # Average the data points for each model on any given quantization level
                avg_metrics = {}
                for metric in relative_metrics:
                    quant_level = metric['Quantization']
                    if quant_level not in avg_metrics:
                        avg_metrics[quant_level] = {'Energy Saving (%)': [], 'Accuracy Loss (pp)': []}
                    avg_metrics[quant_level]['Energy Saving (%)'].append(metric['Energy Saving (%)'])
                    avg_metrics[quant_level]['Accuracy Loss (pp)'].append(metric['Accuracy Loss (pp)'])
                
                avg_relative_metrics = []
                for quant_level, values in avg_metrics.items():
                    avg_relative_metrics.append({
                        'Quantization': quant_level,
                        'Energy Saving (%)': np.mean(values['Energy Saving (%)']),
                        'Accuracy Loss (pp)': np.mean(values['Accuracy Loss (pp)'])
                    })
                
                # Plot energy savings
                ax1.scatter(
                    [m['Quantization'] for m in avg_relative_metrics],
                    [m['Energy Saving (%)'] for m in avg_relative_metrics],
                    label=base_model,
                    marker='o'
                )
                
                # Plot accuracy loss
                ax2.scatter(
                    [m['Quantization'] for m in avg_relative_metrics],
                    [m['Accuracy Loss (pp)'] for m in avg_relative_metrics],
                    label=base_model,
                    marker='o'
                )
            else:
                # Plot energy savings
                for metric in relative_metrics:
                    ax1.scatter(
                        metric['Quantization'],
                        metric['Energy Saving (%)'],
                        label=base_model,
                        marker=self.dataset_markers[metric['Dataset']]
                    )
                
                # Plot accuracy loss
                for metric in relative_metrics:
                    ax2.scatter(
                        metric['Quantization'],
                        metric['Accuracy Loss (pp)'],
                        label=base_model,
                        marker=self.dataset_markers[metric['Dataset']]
                    )

        # Sort all quantization levels according to our ordering
        ordered_levels = sorted(list(all_quant_levels), key=get_quantization_rank)
        
        # Customize plots
        for ax in [ax1, ax2]:
            ax.set_xticks(range(len(ordered_levels)))
            ax.set_xticklabels(ordered_levels, rotation=45, ha='right')
        
        ax1.set_title('Energy Savings by Quantization Level')
        ax1.set_xlabel('Quantization Level')
        ax1.set_ylabel('Energy Saving (%)')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Accuracy Impact by Quantization Level')
        ax2.set_xlabel('Quantization Level')
        ax2.set_ylabel('Accuracy Loss (percentage points)')
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
        
        plt.tight_layout()
        return fig

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
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Token Count')
        plt.ylabel('Energy Consumption (Joules)')
        plt.title('Token Length vs Energy Consumption')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return plt.gcf()

    def plot_energy_per_token_vs_size(self, dataset_name: str = None):
        """Plot energy consumption per token against model size.
        
        The function extracts model size from model names (e.g., '7b', '2b', '70b')
        and plots it against the mean energy consumption per token.
        """
        metrics_list = []
        for (ds_name, model_name), metrics in self.metrics.items():
            if dataset_name is None or ds_name.lower() == dataset_name.lower():
                # Extract model size from name (e.g., "llama_7b" -> 7)
                size = None
                name_parts = model_name.split('_')
                for part in name_parts:
                    if part.lower().endswith('b'):
                        try:
                            size = float(part[:-1])  # Remove 'b' and convert to float
                        except ValueError:
                            continue
                
                if size is not None:
                    model_family = "_".join(model_name.split('_')[:2])  # e.g., "llama_2"
                    metrics_list.append({
                        'Model': model_name,
                        'Dataset': ds_name,
                        'Size (B)': size,
                        'Mean Energy per Token (J)': metrics.mean_energy_per_token,
                        'Std Energy per Token (J)': metrics.std_energy_per_token,
                        'Quantization': metrics.quantization_level,
                        'Base Model': model_family
                    })
        
        df = pd.DataFrame(metrics_list)
        
        # Create the plot
        plt.figure(figsize=(10, 7))
        
        # Plot points with error bars
        for quant_level in sorted(df['Quantization'].unique(), key=lambda x: self._get_quantization_rank(x)):
            quant_data = df[df['Quantization'] == quant_level]
            
            plt.errorbar(
                quant_data['Size (B)'],
                quant_data['Mean Energy per Token (J)'],
                yerr=quant_data['Std Energy per Token (J)'],
                fmt='o',
                label=quant_level,
                capsize=5,
                alpha=0.7,
                markersize=8
            )
        
        # Add trend line
        x = df['Size (B)'].values.reshape(-1, 1)
        y = df['Mean Energy per Token (J)'].values
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        try:
            model.fit(np.log(x), np.log(y))
        except ValueError:
            breakpoint()
        
        # Plot trend line
        x_trend = np.array([df['Size (B)'].min(), df['Size (B)'].max()])
        y_trend = np.exp(model.predict(np.log(x_trend.reshape(-1, 1))))
        plt.plot(x_trend, y_trend, 'k--', alpha=0.5, 
                label=f'Trend (slope: {model.coef_[0]:.2f})')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.xlabel('Model Size (Billion Parameters)')
        plt.ylabel('Mean Energy per Token (Joules)')
        plt.title('Energy Consumption per Token vs Model Size')
        
        # Add annotations for interesting points
        for _, row in df.iterrows():
            if (row['Mean Energy per Token (J)'] > df['Mean Energy per Token (J)'].mean() + 2*df['Mean Energy per Token (J)'].std() or
                row['Size (B)'] > df['Size (B)'].mean() + 2*df['Size (B)'].std()):
                
                plt.annotate(
                    f"{row['Base Model']}\n({row['Quantization']})",
                    (row['Size (B)'], row['Mean Energy per Token (J)']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                )
        
        # Add statistics textbox
        stats_text = (
            f"Models shown: {len(df)}\n"
            f"Correlation: {df['Size (B)'].corr(df['Mean Energy per Token (J)']):.2f}\n"
            f"Size range: {df['Size (B)'].min():.1f}B-{df['Size (B)'].max():.1f}B\n"
            f"Energy/token range: {df['Mean Energy per Token (J)'].min():.2e}-{df['Mean Energy per Token (J)'].max():.2e}J"
        )
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                fontsize=8,
                verticalalignment='top')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return plt.gcf()

    def generate_accuracy_latex_table(self) -> str:
        """Generate a LaTeX table of average and std accuracy for each model on each dataset."""
        accuracy_data = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            df = self.raw_data[f"{dataset_name}_{model_name}"]
            accuracy_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Average Accuracy': df['evaluation'].mean(),
                'Std Accuracy': df['evaluation'].std()
            })
        
        df = pd.DataFrame(accuracy_data)
        pivot_df_avg = df.pivot(index='Model', columns='Dataset', values='Average Accuracy')
        pivot_df_std = df.pivot(index='Model', columns='Dataset', values='Std Accuracy')

        # Combine the average and std tables into one with the format "avg ± std"
        combined_df = pd.DataFrame()
        for col in pivot_df_avg.columns:
            combined_df[col] = pivot_df_avg[col].map(lambda x: f"{x:.2f}") + " ± " + pivot_df_std[col].map(lambda x: f"{x:.2f}")

        # Add a column for the average accuracy across datasets for each model
        combined_df['Average Accuracy'] = pivot_df_avg.mean(axis=1).map(lambda x: f"{x:.2f}") + " ± " + pivot_df_std.mean(axis=1).map(lambda x: f"{x:.2f}")

        # Add a row for the average accuracy across models for each dataset
        combined_df.loc['Average Accuracy'] = pivot_df_avg.mean(axis=0).map(lambda x: f"{x:.2f}") + " ± " + pivot_df_std.mean(axis=0).map(lambda x: f"{x:.2f}")

        # Sort the pivot_df based on model family and quantization rank
        combined_df = combined_df.sort_index(key=lambda x: x.map(lambda y: ("_".join(y.split('_')[:2]), self._get_quantization_rank("_".join(y.split('_')[3:])))))

        latex_table = combined_df.to_latex(
            na_rep="-",
            caption="Average and Std Accuracy of Models on Various Datasets",
            label="tab:accuracy"
        )

        # Make sure to escape any underscores
        latex_table = latex_table.replace("_", r"\_")

        return latex_table

    def generate_energy_latex_table(self) -> str:
        """Generate a LaTeX table of mean and std energy consumption for each model on each dataset."""
        energy_data = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            df = self.raw_data[f"{dataset_name}_{model_name}"]
            energy_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Mean Energy per Token (J/tok)': df['energy_per_token'].mean(),
                'Std Energy per Token (J/tok)': df['energy_per_token'].std()
            })
        
        df = pd.DataFrame(energy_data)
        pivot_df_avg = df.pivot(index='Model', columns='Dataset', values='Mean Energy per Token (J/tok)')
        pivot_df_std = df.pivot(index='Model', columns='Dataset', values='Std Energy per Token (J/tok)')

        # Combine the average and std tables into one with the format "avg ± std"
        combined_df = pd.DataFrame()
        for col in pivot_df_avg.columns:
            combined_df[col] = pivot_df_avg[col].map(lambda x: f"{x:.2f}") + " ± " + pivot_df_std[col].map(lambda x: f"{x:.2f}")

        # Add a column for the average energy consumption across datasets for each model
        combined_df['Average Energy per Token (J/tok)'] = pivot_df_avg.mean(axis=1).map(lambda x: f"{x:.2f}") + " ± " + pivot_df_std.mean(axis=1).map(lambda x: f"{x:.2f}")

        # Add a row for the average energy consumption across models for each dataset
        combined_df.loc['Average Energy per Token (J/tok)'] = pivot_df_avg.mean(axis=0).map(lambda x: f"{x:.2f}") + " ± " + pivot_df_std.mean(axis=0).map(lambda x: f"{x:.2f}")

        # Sort the pivot_df based on model family and quantization rank
        combined_df = combined_df.sort_index(key=lambda x: x.map(lambda y: ("_".join(y.split('_')[:2]), self._get_quantization_rank("_".join(y.split('_')[3:])))))

        latex_table = combined_df.to_latex(
            na_rep="-",
            caption="Mean and Std Energy Consumption of Models on Various Datasets",
            label="tab:energy"
        )

        # Make sure to escape any underscores
        latex_table = latex_table.replace("_", r"\_")

        return latex_table

    def generate_energy_per_token_latex_table(self) -> str:
        """Generate a LaTeX table of mean and std energy per token for each model family, averaged across all datasets."""
        energy_data = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            model_family = "_".join(model_name.split('_')[:2])
            energy_data.append({
                'Base Model': model_family,
                'Mean Energy per Token (J/tok)': metrics.mean_energy_per_token,
                'Std Energy per Token (J/tok)': metrics.std_energy_per_token
            })
        
        df = pd.DataFrame(energy_data)
        summary_df = df.groupby('Base Model').agg({
            'Mean Energy per Token (J/tok)': 'mean',
            'Std Energy per Token (J/tok)': 'mean'
        }).reset_index()

        latex_table = summary_df.to_latex(
            float_format="%.2f",
            na_rep="-",
            caption="Mean and Std Energy per Token for Base Models Averaged Across Datasets",
            label="tab:energy_per_token"
        )

        # Make sure to escape any underscores
        latex_table = latex_table.replace("_", r"\_")

        return latex_table

    def generate_energy_per_response_latex_table(self) -> str:
        """Generate a LaTeX table of mean and std energy per token for each model family, averaged across all datasets."""
        energy_data = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            model_family = "_".join(model_name.split('_')[:2])
            energy_data.append({
                'Base Model': model_family,
                'Mean Energy per Response (J/tok)': metrics.mean_energy,
                'Std Energy per Response (J/tok)': metrics.std_energy
            })
        
        df = pd.DataFrame(energy_data)
        summary_df = df.groupby('Base Model').agg({
            'Mean Energy per Response (J/tok)': 'mean',
            'Std Energy per Response (J/tok)': 'mean'
        }).reset_index()

        latex_table = summary_df.to_latex(
            float_format="%.2f",
            na_rep="-",
            caption="Mean and Std Energy per Response for Base Models Averaged Across Datasets",
            label="tab:energy_per_response"
        )

        # Make sure to escape any underscores
        latex_table = latex_table.replace("_", r"\_")

        return latex_table

    def generate_energy_per_response_latex_table_by_dataset(self) -> str:
        """Generate a LaTeX table of mean and std energy per response for each model on each dataset."""
        energy_data = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            energy_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Mean Energy per Response (J)': metrics.mean_energy,
                'Std Energy per Response (J)': metrics.std_energy
            })
        
        df = pd.DataFrame(energy_data)
        pivot_df_avg = df.pivot(index='Model', columns='Dataset', values='Mean Energy per Response (J)')
        pivot_df_std = df.pivot(index='Model', columns='Dataset', values='Std Energy per Response (J)')

        # Combine the average and std tables into one with the format "avg ± std"
        combined_df = pd.DataFrame()
        for col in pivot_df_avg.columns:
            combined_df[col] = pivot_df_avg[col].map(lambda x: f"{x:.2f}") + " ± " + pivot_df_std[col].map(lambda x: f"{x:.2f}")

        # Add a column for the average energy consumption across datasets for each model
        combined_df['Average Energy per Response (J)'] = pivot_df_avg.mean(axis=1).map(lambda x: f"{x:.2f}") + " ± " + pivot_df_std.mean(axis=1).map(lambda x: f"{x:.2f}")

        # Add a row for the average energy consumption across models for each dataset
        combined_df.loc['Average Energy per Response (J)'] = pivot_df_avg.mean(axis=0).map(lambda x: f"{x:.2f}") + " ± " + pivot_df_std.mean(axis=0).map(lambda x: f"{x:.2f}")

        # Sort the pivot_df based on model family and quantization rank
        combined_df = combined_df.sort_index(key=lambda x: x.map(lambda y: ("_".join(y.split('_')[:2]), self._get_quantization_rank("_".join(y.split('_')[3:])))))

        latex_table = combined_df.to_latex(
            na_rep="-",
            caption="Mean and Std Energy per Response for Models on Various Datasets",
            label="tab:energy_per_response_by_dataset"
        )

        # Make sure to escape any underscores
        latex_table = latex_table.replace("_", r"\_")

        return latex_table

    def generate_response_length_latex_table(self) -> str:
        """Generate a LaTeX table of average and std response length (eval_count) for each model on each dataset."""
        response_length_data = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            df = self.raw_data[f"{dataset_name}_{model_name}"]
            response_length_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Average Response Length': df['eval_count'].mean(),
                'Std Response Length': df['eval_count'].std()
            })
        
        df = pd.DataFrame(response_length_data)
        pivot_df_avg = df.pivot(index='Model', columns='Dataset', values='Average Response Length')
        pivot_df_std = df.pivot(index='Model', columns='Dataset', values='Std Response Length')

        # Combine the average and std tables into one with the format "avg ± std"
        combined_df = pd.DataFrame()
        for col in pivot_df_avg.columns:
            combined_df[col] = pivot_df_avg[col].map(lambda x: f"{x:.2f}") + " ± " + pivot_df_std[col].map(lambda x: f"{x:.2f}")

        # Add a column for the average response length across datasets for each model
        combined_df['Average Response Length'] = pivot_df_avg.mean(axis=1).map(lambda x: f"{x:.2f}") + " ± " + pivot_df_std.mean(axis=1).map(lambda x: f"{x:.2f}")

        # Add a row for the average response length across models for each dataset
        combined_df.loc['Average Response Length'] = pivot_df_avg.mean(axis=0).map(lambda x: f"{x:.2f}") + " ± " + pivot_df_std.mean(axis=0).map(lambda x: f"{x:.2f}")

        # Add the average of the average response length across all models and datasets
        combined_df.loc['Average Response Length', 'Average Response Length'] = df['Average Response Length'].mean()

        # Sort the pivot_df based on model family and quantization rank
        combined_df = combined_df.sort_index(key=lambda x: x.map(lambda y: ("_".join(y.split('_')[:2]), self._get_quantization_rank("_".join(y.split('_')[3:])))))

        latex_table = combined_df.to_latex(
            na_rep="-",
            caption="Average and Std Response Length for Models on Various Datasets",
            label="tab:response_length"
        )

        # Make sure to escape any underscores
        latex_table = latex_table.replace("_", r"\_")

        return latex_table

    def plot_average_accuracy(self):
        """Plot the average accuracy across models and datasets."""
        accuracy_data = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            accuracy_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': metrics.accuracy
            })
        
        df = pd.DataFrame(accuracy_data)
        pivot_df = df.pivot(index='Model', columns='Dataset', values='Accuracy')

        # Calculate the average accuracy for each model
        pivot_df['Average Accuracy'] = pivot_df.mean(axis=1)

        # Plot the average accuracy
        plt.figure(figsize=(10, 6))
        pivot_df['Average Accuracy'].sort_values().plot(kind='barh', color='skyblue')
        plt.xlabel('Average Accuracy')
        plt.title('Average Accuracy Across Models and Datasets')
        plt.tight_layout()
        # plt.savefig("average_accuracy_across_models_datasets.pdf")
        # plt.show()
        return plt.gcf()

    def plot_average_accuracy_per_dataset(self):
        """Plot the average accuracy across models per dataset, grouped by dataset."""
        accuracy_data = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            accuracy_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': metrics.accuracy
            })
        
        df = pd.DataFrame(accuracy_data)
        pivot_df = df.pivot(index='Model', columns='Dataset', values='Accuracy')

        # Plot the average accuracy per dataset, grouped by dataset
        pivot_df.plot(kind='bar', subplots=True, layout=(len(pivot_df.columns), 1), figsize=(14, 8 * len(pivot_df.columns)), sharex=True)
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.suptitle('Average Accuracy Across Models Per Dataset', y=1.02)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("average_accuracy_per_dataset_grouped.pdf")
        plt.show()

    def plot_accuracy_subplots_vertical_bars(self):
        """Plot the accuracy results with colors representing the model family."""
        accuracy = {}
        datasets = set()

        for (dataset, model), metrics in self.metrics.items():
            if model not in accuracy:
                accuracy[model] = {}
            accuracy[model][dataset] = metrics.accuracy
            datasets.update(accuracy[model].keys())

        num_datasets = len(datasets)

        fig, axes = plt.subplots(num_datasets, 1, figsize=(8, 2 * num_datasets + 1), sharex=True)
        if num_datasets == 1:
            axes = [axes]

        for ax, dataset in zip(axes, datasets):
            ax.set_title(f"Accuracy for {dataset}")
            ax.set_xlabel("Model")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)

            models = []
            accuracies = []
            colors = []

            for model in accuracy:
                if dataset in accuracy[model]:
                    models.append(model)
                    accuracies.append(accuracy[model][dataset])
                    model_family = "_".join(model.split("_")[:2])
                    colors.append(self.colors.get(model_family, 'gray'))
                else:
                    models.append(model)
                    accuracies.append(0)
                    colors.append('gray')

            ax.bar(models, accuracies, color=colors)
            ax.set_xticklabels(models, rotation=45, ha='right')

        # Create legend
        model_families = { "_".join(model.split("_")[:2]) for model in accuracy }
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors.get(family, 'gray'), markersize=10) for family in model_families]
        labels = list(model_families)

        fig.legend(handles, labels, title="Base Model", loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()
        return plt.gcf()

    def plot_tokens_per_second(self, use_total_duration=False):
        """Plot the tokens per second for all models."""
        tokens_per_second_data = []

        if use_total_duration:
            duration_col = "total_duration"
        else:
            duration_col = "eval_duration"

        for (dataset_name, model_name), metrics in self.metrics.items():
            df = self.raw_data[f"{dataset_name}_{model_name}"]

            # Convert timedelta to nanoseconds
            df["total_duration"] = pd.to_timedelta(df["total_duration"]).dt.total_seconds() * 10**9

            tokens_per_second = df["eval_count"] / (df[duration_col] * 10**-9)
            tokens_per_second_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Tokens per Second': tokens_per_second.mean(),
                'Quantization': metrics.quantization_level,
                'Base Model': "_".join(model_name.split("_")[:2])  # Extract model family
            })

        
        df = pd.DataFrame(tokens_per_second_data)

        # Sort the models according to the QUANTIZATION_ORDER
        df['Quantization_Rank'] = df['Quantization'].apply(self._get_quantization_rank)
        df = df.sort_values(by=['Base Model', 'Quantization_Rank'])

        pivot_df = df.pivot(index='Model', columns='Dataset', values='Tokens per Second')
        # Sort the rows according to the quantization rank
        pivot_df["Base Model"] = pivot_df.index.map(lambda x: "_".join(x.split("_")[:2]))
        pivot_df["Quantization_Rank"] = pivot_df.index.map(lambda x: self._get_quantization_rank(df.loc[df["Model"] == x, "Quantization"].values[0]))
        pivot_df = pivot_df.sort_values(by=["Base Model", "Quantization_Rank"], ascending=[True, False])
        pivot_df.drop(columns=["Base Model", "Quantization_Rank"], inplace=True)

        # Plot the tokens per second with horizontal bars
        ax = pivot_df.plot(kind='barh', figsize=(7, 9), color=[self.colors.get(dataset, 'gray') for dataset in pivot_df.columns])

        plt.ylabel('Model')
        
        if use_total_duration:
            plt.xlabel('Tokens per Second (Total Duration)')
            plt.title('Tokens per Second for All Models (Total Duration)')
        else:
            plt.xlabel('Tokens per Second')
            plt.title('Tokens per Second for All Models')

        # Color the background of the tick labels according to the model family
        for tick_label in ax.get_yticklabels():
            model_name = tick_label.get_text()
            model_family = "_".join(model_name.split("_")[:2])
            color = self.colors.get(model_family, 'gray')
            tick_label.set_bbox(dict(facecolor=color, edgecolor='none', alpha=0.5))
            # tick_label.set_color(color)

        # Add legend for model families
        model_families = df['Base Model'].unique()
        family_handles = [plt.Line2D([0], [0], color=self.colors.get(family, 'gray'), lw=4) for family in model_families]
        family_labels = model_families

        # Add legend for datasets
        dataset_handles = [plt.Line2D([0], [0], color=self.colors.get(dataset, 'gray'), lw=4) for dataset in pivot_df.columns]
        dataset_labels = pivot_df.columns

        # Combine legends
        first_legend = plt.legend(family_handles, family_labels, title='Base Model', bbox_to_anchor=(-0.27, -0.05), loc='upper center', ncol=1)
        for lh in first_legend.legend_handles: 
            lh.set_alpha(0.5)
        ax.add_artist(first_legend)
        plt.legend(dataset_handles, dataset_labels, title='Dataset', bbox_to_anchor=(0.35, -0.23), loc='lower center', ncol=2)

        plt.tight_layout()
        # plt.savefig("output/tokens_per_second.pdf")

        # Add average values to pivot_df, both per model and per dataset
        pivot_df["Average"] = pivot_df.mean(axis=1)
        pivot_df.loc["Average"] = pivot_df.mean(axis=0)


        # Write the results (values) to a CSV file
        pivot_df.to_csv("output/tokens_per_second.csv")

        # Write to latex
        latex_table = pivot_df.to_latex(
            float_format="%.2f",
            na_rep="-",
            caption="Tokens per Second for All Models",
            label="tab:tokens_per_second"
        )

        # Escape underscores
        latex_table = latex_table.replace("_", r"\_")

        with open("output/tokens_per_second.tex", "w") as f:
            f.write(latex_table)

        return plt.gcf()

    def plot_model_size_vs_metrics(self, fit_regression=False, in_bytes=False, log_scale=True):
        """Plot model size against accuracy and energy consumption."""
        metrics_list = []
        model_info = utils.get_model_info()

        for (ds_name, model_name), metrics in self.metrics.items():
            # Extract model size from name (e.g., "llama_7b" -> 7)
            size = None
            if in_bytes:
                # Translate model to model key. Need to handle cases like:
                #   qwen25_05b_instruct_q4_0 -> qwen2.5:0.5b-instruct-q4_0
                #   qwen25_15b_instruct_q4_0 -> qwen2.5:1.5b-instruct-q4_0
                #   llama32_1b_instruct_q3_K_L -> llama3.2:1b-instruct-q3_K_L
                #   gemma2_2b_instruct_q3_K_L -> gemma2:2b-instruct-q3_K_L
                model_key = model_name.replace("_", ":", 1).replace("_", "-", 2)
                model_key = re.sub(r"(\d)(\d)(?=\D)", r"\1.\2", model_key, count=2)
                model_key = next((item['model'] for item in model_info['models'] if item['name'] == model_key), None)
                if model_key:
                    size = next(item['size'] for item in model_info['models'] if item['model'] == model_key)
            else:
                name_parts = model_name.split('_')
                for part in name_parts:
                    if part.lower().endswith('b'):
                        try:
                            size = float(part[:-1])  # Remove 'b' and convert to float
                        except ValueError:
                            continue
            
            if size is not None:
                model_family = "_".join(model_name.split('_')[:2])  # e.g., "llama_2"
                metrics_list.append({
                    'Model': model_name,
                    'Dataset': ds_name,
                    'Size': size,
                    'Mean Energy per Token (J)': metrics.mean_energy_per_token,
                    'Accuracy': metrics.accuracy,
                    'Quantization': metrics.quantization_level,
                    'Base Model': model_family
                })
        
        df = pd.DataFrame(metrics_list)
        
        # Plot model size vs accuracy
        datasets = df['Dataset'].unique()
        num_datasets = len(datasets)
        fig, axes = plt.subplots(num_datasets, 1, figsize=(10, 5 * num_datasets), sharex=True)
        if num_datasets == 1:
            axes = [axes]

        for ax, dataset in zip(axes, datasets):
            dataset_df = df[df['Dataset'] == dataset]
            sns.scatterplot(
                data=dataset_df,
                x='Size',
                y='Accuracy',
                hue='Base Model',
                style='Quantization',
                s=100,
                ax=ax,
                markers=self.markers,
                palette=self.colors
            )
            ax.set_title(f'Model Size vs Accuracy for {dataset}')
            ax.set_xlabel('Model Size (Bytes)' if in_bytes else 'Model Size (Billion Parameters)')
            ax.set_ylabel('Accuracy')
            if log_scale:
                ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            if fit_regression:
                from sklearn.linear_model import LinearRegression
                x = dataset_df['Size'].values.reshape(-1, 1)
                y = dataset_df['Accuracy'].values
                model = LinearRegression()
                model.fit(np.log(x) if log_scale else x, y)
                x_trend = np.array([dataset_df['Size'].min(), dataset_df['Size'].max()])
                y_trend = model.predict(np.log(x_trend.reshape(-1, 1)) if log_scale else x_trend.reshape(-1, 1))
                ax.plot(x_trend, y_trend, 'k--', alpha=0.5, label=f'Trend (slope: {model.coef_[0]:.2e})')
                ax.legend()

        plt.tight_layout()
        accuracy_figure = plt.gcf()

        # Plot model size vs energy consumption
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=df,
            x='Size',
            y='Mean Energy per Token (J)',
            hue='Dataset',
            style='Quantization',
            s=100,
            markers=self.markers,
            palette=self.colors
        )
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
        plt.xlabel('Model Size (Bytes)' if in_bytes else 'Model Size (Billion Parameters)')
        plt.ylabel('Mean Energy per Token (Joules)')
        plt.title('Model Size vs Mean Energy per Token')
        plt.grid(True, alpha=0.3)
        if fit_regression:
            x = df['Size'].values.reshape(-1, 1)
            y = df['Mean Energy per Token (J)'].values
            model = LinearRegression()
            model.fit(np.log(x) if log_scale else x, np.log(y) if log_scale else y)
            x_trend = np.array([df['Size'].min(), df['Size'].max()])
            y_trend = np.exp(model.predict(np.log(x_trend.reshape(-1, 1)))) if log_scale else model.predict(x_trend.reshape(-1, 1))
            plt.plot(x_trend, y_trend, 'k--', alpha=0.5, label=f'Trend (slope: {model.coef_[0]:.2e})')
            plt.legend()

        plt.tight_layout()
        energy_figure = plt.gcf()

        return accuracy_figure, energy_figure

    def plot_model_size_vs_metrics_grid(self, fit_regression=False, in_bytes=False, log_scale=True):
        """Plot model size against accuracy and energy consumption in a grid of subplots."""
        metrics_list = []
        model_info = utils.get_model_info()

        for (ds_name, model_name), metrics in self.metrics.items():
            # Extract model size from name (e.g., "llama_7b" -> 7)
            size = None
            if in_bytes:
                # Translate model to model key. Need to handle cases like:
                #   qwen25_05b_instruct_q4_0 -> qwen2.5:0.5b-instruct-q4_0
                #   qwen25_15b_instruct_q4_0 -> qwen2.5:1.5b-instruct-q4_0
                #   llama32_1b_instruct_q3_K_L -> llama3.2:1b-instruct-q3_K_L
                #   gemma2_2b_instruct_q3_K_L -> gemma2:2b-instruct-q3_K_L
                #   llama32_1b_instruct_fp16 -> llama3.2:1b-instruct-fp16
                model_key = model_name.replace("_", ":", 1).replace("_", "-", 2)
                model_key = re.sub(r"(\d)(\d)(?=\D)", r"\1.\2", model_key, count=2)
                # model_key = re.sub(r"(\d)(\d)", r"\1.\2", model_key, count=2)
                model_key = next((item['model'] for item in model_info['models'] if item['name'] == model_key), None)
                if model_key:
                    size = next(item['size'] for item in model_info['models'] if item['model'] == model_key)
            else:
                name_parts = model_name.split('_')
                for part in name_parts:
                    if part.lower().endswith('b'):
                        try:
                            size = float(part[:-1])  # Remove 'b' and convert to float
                        except ValueError:
                            continue
            
            if size is not None:
                model_family = "_".join(model_name.split('_')[:2])  # e.g., "llama_2"
                metrics_list.append({
                    'Model': model_name,
                    'Dataset': ds_name,
                    'Size': size,
                    'Mean Energy per Token (J)': metrics.mean_energy_per_token,
                    'Accuracy': metrics.accuracy,
                    'Quantization': metrics.quantization_level,
                    'Base Model': model_family
                })

        df = pd.DataFrame(metrics_list)
        datasets = df['Dataset'].unique()
        model_families = df['Base Model'].unique()
        num_datasets = len(datasets)
        num_model_families = len(model_families)

        fig, axes = plt.subplots(num_datasets, num_model_families, figsize=(3 * num_model_families, 3 * num_datasets), sharex=True, sharey=True)
        if num_datasets == 1:
            axes = [axes]
        if num_model_families == 1:
            axes = [[ax] for ax in axes]

        for i, dataset in enumerate(datasets):
            for j, model_family in enumerate(model_families):
                ax = axes[i][j]
                dataset_df = df[(df['Dataset'] == dataset) & (df['Base Model'] == model_family)]
                sns.scatterplot(
                    data=dataset_df,
                    x='Size',
                    y='Accuracy',
                    style='Quantization',
                    s=100,
                    ax=ax,
                    markers=self.markers,
                    # palette=self.colors
                )
                ax.set_title(f'{model_family} on {dataset}')
                ax.set_xlabel('Model Size (Bytes)' if in_bytes else 'Model Size (Billion Parameters)')
                ax.set_ylabel('Accuracy')
                if log_scale:
                    ax.set_xscale('log')
                ax.grid(True, alpha=0.3)

                if fit_regression and not dataset_df.empty:
                    from sklearn.linear_model import LinearRegression
                    x = dataset_df['Size'].values.reshape(-1, 1)
                    y = dataset_df['Accuracy'].values
                    model = LinearRegression()
                    model.fit(np.log(x) if log_scale else x, y)
                    x_trend = np.array([dataset_df['Size'].min(), dataset_df['Size'].max()])
                    y_trend = model.predict(np.log(x_trend.reshape(-1, 1)) if log_scale else x_trend.reshape(-1, 1))
                    ax.plot(x_trend, y_trend, 'k--', alpha=0.5, label=f'Trend (slope: {model.coef_[0]:.2e})')
                    # Add only the label for the trend line to the legend, leave out all other handles in the legend
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles[-1:], labels[-1:])

        # Add shared legend for quantization levels
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(handles))
        # --- Create a shared legend below all subplots ---
        handles = [plt.Line2D([0], [0], marker=self.markers[key], color='w', 
                            markerfacecolor='black', markersize=8, linestyle='') 
                for key in self.markers]

        labels = list(self.markers.keys())

        fig.legend(handles, labels, title="Quantization Levels", loc="lower center", ncol=len(self.markers), frameon=False, bbox_to_anchor=(0.5, -0.05))

        # Adjust layout to make space for the global legend
        plt.subplots_adjust(bottom=0.2)

        plt.tight_layout()
        accuracy_figure = plt.gcf()

        fig, axes = plt.subplots(num_datasets, num_model_families, figsize=(3 * num_model_families, 3 * num_datasets), sharex=True, sharey=True)
        if num_datasets == 1:
            axes = [axes]
        if num_model_families == 1:
            axes = [[ax] for ax in axes]

        for i, dataset in enumerate(datasets):
            for j, model_family in enumerate(model_families):
                ax = axes[i][j]
                dataset_df = df[(df['Dataset'] == dataset) & (df['Base Model'] == model_family)]
                sns.scatterplot(
                    data=dataset_df,
                    x='Size',
                    y='Mean Energy per Token (J)',
                    style='Quantization',
                    s=100,
                    ax=ax,
                    markers=self.markers,
                    # palette=self.colors
                )
                ax.set_title(f'{model_family} on {dataset}')
                ax.set_xlabel('Model Size (Bytes)' if in_bytes else 'Model Size (Billion Parameters)')
                ax.set_ylabel('Mean Energy per Token (J)')
                if log_scale:
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                if fit_regression and not dataset_df.empty:
                    from sklearn.linear_model import LinearRegression
                    x = dataset_df['Size'].values.reshape(-1, 1)
                    y = dataset_df['Mean Energy per Token (J)'].values
                    model = LinearRegression()
                    model.fit(np.log(x) if log_scale else x, np.log(y) if log_scale else y)
                    x_trend = np.array([dataset_df['Size'].min(), dataset_df['Size'].max()])
                    y_trend = np.exp(model.predict(np.log(x_trend.reshape(-1, 1)))) if log_scale else model.predict(x_trend.reshape(-1, 1))
                    ax.plot(x_trend, y_trend, 'k--', alpha=0.5, label=f'Trend (slope: {model.coef_[0]:.2e})')
                    # Add only the label for the trend line to the legend, leave out all other handles in the legend
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles[-1:], labels[-1:])


        # --- Create a shared legend below all subplots ---
        handles = [plt.Line2D([0], [0], marker=self.markers[key], color='w', 
                            markerfacecolor='black', markersize=8, linestyle='') 
                for key in self.markers]

        labels = list(self.markers.keys())

        fig.legend(handles, labels, title="Quantization Levels", loc="lower center", ncol=len(self.markers), frameon=False, bbox_to_anchor=(0.5, -0.05))

        # Adjust layout to make space for the global legend
        plt.subplots_adjust(bottom=0.2)

        plt.tight_layout()
        energy_figure = plt.gcf()

        return accuracy_figure, energy_figure

    def plot_quantization_impact(self, dataset_name: str = None):
        """Analyze the impact of quantization on energy savings and accuracy with 'q3_K_S' as the baseline."""
        baseline_quantization = 'q3_K_S'
        metrics_list = []
        
        for (ds_name, model_name), metrics in self.metrics.items():
            if dataset_name is None or ds_name.lower() == dataset_name.lower():
                model_family = "_".join(model_name.split('_')[:2])  # Extract model family (e.g., "gemma_2b")
                metrics_list.append({
                    'Model': model_name,
                    'Dataset': ds_name,
                    'Mean Energy (J)': metrics.mean_energy,
                    'Accuracy': metrics.accuracy,
                    'Quantization': metrics.quantization_level,
                    'Base Model': model_family
                })
        
        df = pd.DataFrame(metrics_list)
        
        # Filter out models that do not have the baseline quantization level
        baseline_models = df[df['Quantization'] == baseline_quantization]['Model'].unique()
        df = df[df['Model'].isin(baseline_models)]
        
        # Calculate relative metrics
        relative_metrics = []
        for model in baseline_models:
            baseline_metrics = df[(df['Model'] == model) & (df['Quantization'] == baseline_quantization)]
            if baseline_metrics.empty:
                continue
            baseline_energy = baseline_metrics['Mean Energy (J)'].values[0]
            baseline_accuracy = baseline_metrics['Accuracy'].values[0]
            
            for _, row in df[df['Model'] == model].iterrows():
                relative_metrics.append({
                    'Model': model,
                    'Quantization': row['Quantization'],
                    'Energy Saving (%)': 100 * (1 - row['Mean Energy (J)'] / baseline_energy),
                    'Accuracy Loss (pp)': 100 * (baseline_accuracy - row['Accuracy'])
                })
        
        relative_df = pd.DataFrame(relative_metrics)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sort relative_metrics based on the defined order
        relative_df['Quantization_Rank'] = relative_df['Quantization'].apply(self._get_quantization_rank)
        relative_df = relative_df.sort_values(by='Quantization_Rank')
        
        # Plot energy savings
        sns.barplot(
            data=relative_df,
            x='Quantization',
            y='Energy Saving (%)',
            hue='Model',
            ax=ax1,
            palette=self.colors,
            order=self.QUANTIZATION_ORDER
        )
        ax1.set_title('Energy Savings by Quantization Level')
        ax1.set_xlabel('Quantization Level')
        ax1.set_ylabel('Energy Saving (%)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot accuracy loss
        sns.barplot(
            data=relative_df,
            x='Quantization',
            y='Accuracy Loss (pp)',
            hue='Model',
            ax=ax2,
            palette=self.colors,
            order=self.QUANTIZATION_ORDER
        )
        ax2.set_title('Accuracy Impact by Quantization Level')
        ax2.set_xlabel('Quantization Level')
        ax2.set_ylabel('Accuracy Loss (percentage points)')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add legend
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
        
        plt.tight_layout()
        return fig

    def plot_tokens_per_joule_distribution(self, dataset_name: str = None, log_scale=False, subplots=False):
        """Plot tokens per joule distribution for each model.
        
        If subplots is True, plot each dataset in a subplot with shared x-axis.
        """
        column = 'tokens_per_joule'

        data = []
        for experiment_name, df in self.raw_data.items():
            ds_name = utils.get_dataset_name(experiment_name)
            model_name = utils.get_model_name(experiment_name)
            model_family = "_".join(model_name.split("_")[:3])  # Extract model family
            if dataset_name is None or ds_name.lower() == dataset_name.lower():
                df[column] = df['eval_count'] / df['energy_consumption_joules']
                data.extend([{
                    'Model': model_name,
                    'Dataset': ds_name,
                    'Tokens per Joule': tokens_per_joule,
                    'Base Model': model_family
                } for tokens_per_joule in df[column]])
        
        df = pd.DataFrame(data)

        if subplots:
            datasets = df['Dataset'].unique()
            num_datasets = len(datasets)
            fig, axes = plt.subplots(num_datasets, 1, figsize=(12, 3 * num_datasets), sharex=True)
            if num_datasets == 1:
                axes = [axes]

            for ax, dataset in zip(axes, datasets):
                sns.boxplot(
                    data=df[df['Dataset'] == dataset],
                    x='Model',
                    y='Tokens per Joule',
                    hue='Base Model',
                    ax=ax,
                    palette=self.colors
                )
                ax.set_title(f'Tokens per Joule Distribution by Model for {dataset}')
                ax.set_xlabel('Model')
                ax.set_ylabel('Tokens per Joule')
                ax.tick_params(axis='x', rotation=45)
                if log_scale:
                    ax.set_yscale('log')
            plt.tight_layout()

            # Add shared legend at the bottom
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
        else:
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=df,
                x='Model',
                y='Tokens per Joule',
                hue='Base Model',
                palette=self.colors
            )
            plt.xticks(rotation=45, ha='right')
            if log_scale:
                plt.yscale('log')
            plt.title('Tokens per Joule Distribution by Model')
            plt.tight_layout()
        

        # plt.savefig("output/tokens_per_joule_distribution.pdf")
        return plt.gcf()

    def print_energy_per_token_summary(self):
        """Print a summary of the energy consumption per token for each model and dataset."""
        summary_data = []

        for (dataset_name, model_name), metrics in self.metrics.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Mean Energy per Token (J)': metrics.mean_energy_per_token,
                'Std Energy per Token (J)': metrics.std_energy_per_token,
                'Median Energy per Token (J)': metrics.median_energy_per_token,
                'Min Energy per Token (J)': metrics.minimum_energy_per_token,
                'Max Energy per Token (J)': metrics.maximum_energy_per_token,
                'Q1 Energy per Token (J)': metrics.q1_energy_per_token,
                'Q3 Energy per Token (J)': metrics.q3_energy_per_token,
                'IQR Energy per Token (J)': metrics.iqr_energy_per_token
            })

        df = pd.DataFrame(summary_data)

        # Print overall summary
        # print("Overall Energy Consumption per Token Summary:")
        overall_summary = df.describe()
        overall_summary.to_csv("output/energy_per_token_summary.csv")

        # Print summary for each model
        # print("\nEnergy Consumption per Token Summary by Model:")
        model_summary = df.groupby('Model').describe()
        model_summary.to_csv("output/energy_per_token_summary_by_model.csv")

        # Print summary for each dataset
        # print("\nEnergy Consumption per Token Summary by Dataset:")
        dataset_summary = df.groupby('Dataset').describe()
        dataset_summary.to_csv("output/energy_per_token_summary_by_dataset.csv")

    def print_accuracy_summary(self):
        """Print a summary of the accuracy for each model and dataset."""
        summary_data = []

        for (dataset_name, model_name), metrics in self.metrics.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': metrics.accuracy
            })

        df = pd.DataFrame(summary_data)

        # Print overall summary
        overall_summary = df.describe()
        overall_summary.to_csv("output/accuracy_summary.csv")

        # Print summary for each model
        model_summary = df.groupby('Model').describe()
        model_summary.to_csv("output/accuracy_summary_by_model.csv")

        # Print summary for each dataset
        dataset_summary = df.groupby('Dataset').describe()
        dataset_summary.to_csv("output/accuracy_summary_by_dataset.csv")

    def plot_response_length_distribution(self, dataset_name: str = None, log_scale=False, subplots=False):
        """Plot response length distribution for each model.
        
        If subplots is True, plot each dataset in a subplot with shared x-axis.
        """
        column = 'eval_count'

        data = []
        for experiment_name, df in self.raw_data.items():
            ds_name = utils.get_dataset_name(experiment_name)
            model_name = utils.get_model_name(experiment_name)
            model_family = "_".join(model_name.split("_")[:3])  # Extract model family
            if dataset_name is None or ds_name.lower() == dataset_name.lower():
                data.extend([{
                    'Model': model_name,
                    'Dataset': ds_name,
                    'Response Length': length,
                    'Base Model': model_family
                } for length in df[column]])
        
        df = pd.DataFrame(data)

        if log_scale:
            showfliers = True
        else:
            showfliers = False

        if subplots:
            datasets = df['Dataset'].unique()
            num_datasets = len(datasets)
            fig, axes = plt.subplots(num_datasets, 1, figsize=(12, 3 * num_datasets), sharex=True)
            if num_datasets == 1:
                axes = [axes]

            for ax, dataset in zip(axes, datasets):
                sns.boxplot(
                    data=df[df['Dataset'] == dataset],
                    x='Model',
                    y='Response Length',
                    showfliers=showfliers
                )
                ax.set_title(f'Response Length Distribution by Model for {dataset}')
                ax.set_xlabel('Model')
                ax.set_ylabel('Response Length')
                ax.tick_params(axis='x', rotation=45)
                if log_scale:
                    ax.set_yscale('log')
            plt.tight_layout()

            # Add shared legend at the bottom
            model_families = df['Base Model'].unique()
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors.get(family, 'gray'), markersize=10) for family in model_families]
            labels = list(model_families)

            fig.legend(handles, labels, title="Base Model", loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
        else:
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=df,
                x='Model',
                y='Response Length',
                hue='Base Model',
                palette=self.colors,
                showfliers=showfliers
            )
            plt.xticks(rotation=45, ha='right')
            if log_scale:
                plt.yscale('log')
            plt.title('Response Length Distribution by Model')
            plt.tight_layout()
        
        plt.savefig("output/response_length_distribution.pdf")
        return plt.gcf()

    def analyze_response_length_energy_correlation(self, dataset_name: str = None):
        """Analyze the correlation between response length and energy consumption per prompt."""
        correlation_data = []
        for (ds_name, model_name), metrics in self.metrics.items():
            if dataset_name is None or ds_name.lower() == dataset_name.lower():
                df = self.raw_data[f"{ds_name}_{model_name}"]
                correlation_data.append({
                    'Dataset': ds_name,
                    'Model': model_name,
                    'Correlation': df['eval_count'].corr(df['energy_consumption_joules'])
                })
        
        df = pd.DataFrame(correlation_data)
        pivot_df = df.pivot(index='Model', columns='Dataset', values='Correlation')

        # Add a column for the average correlation across datasets for each model
        pivot_df['Average Correlation'] = pivot_df.mean(axis=1)

        # Add a row for the average correlation across models for each dataset
        pivot_df.loc['Average Correlation'] = pivot_df.mean(axis=0)

        # Sort the pivot_df based on model family and quantization rank
        pivot_df = pivot_df.sort_index(key=lambda x: x.map(lambda y: ("_".join(y.split('_')[:2]), self._get_quantization_rank("_".join(y.split('_')[3:])))))

        # Plot the correlation
        plt.figure(figsize=(10, 7))
        sns.heatmap(pivot_df, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation between Response Length and Energy Consumption')
        plt.tight_layout()
        plt.savefig("output/response_length_energy_correlation.pdf")
        
        # Save the correlation data as a LaTeX table
        latex_table = pivot_df.to_latex(
            na_rep="-",
            caption="Correlation between Response Length and Energy Consumption",
            label="tab:response_length_energy_correlation"
        )
        with open("output/response_length_energy_correlation_table.tex", "w") as f:
            f.write(latex_table)
        
        return plt.gcf()

    def plot_energy_average_accuracy_tradeoff(self, log_scale=False, dataset_name: str = None):
        """Plot energy consumption vs average accuracy across all datasets for all models"""
        metrics_list = []
        for (ds_name, model_name), metrics in self.metrics.items():
            model_family = "_".join(model_name.split('_')[:2])  # Extract model family (e.g., "gemma_2b")
            metrics_list.append({
                'Model': model_name,
                'Dataset': ds_name,
                'Mean Energy (J)': metrics.mean_energy,
                'Accuracy': metrics.accuracy,
                'Quantization': metrics.quantization_level,
                'Base Model': model_family
            })
        
        df = pd.DataFrame(metrics_list)
        
        # Calculate average accuracy across all datasets for each model
        avg_accuracy_df = df.groupby('Model')['Accuracy'].mean().reset_index()
        avg_accuracy_df.columns = ['Model', 'Average Accuracy']
        
        # Merge average accuracy with the original dataframe
        df = df.merge(avg_accuracy_df, on='Model')
        
        # Find Pareto-optimal points
        def is_pareto_optimal(energy, accuracy, df):
            """
            Determine if a point is Pareto optimal by checking if it is dominated by any other point.
            
            A point P1(energy1, accuracy1) dominates another point P2(energy2, accuracy2) if:
            1. P1 has less or equal energy consumption (energy1 ≤ energy2)
            2. P1 has higher or equal accuracy (accuracy1 ≥ accuracy2)
            3. P1 is strictly better in at least one dimension 
            (either energy1 < energy2 OR accuracy1 > accuracy2)
            
            In our case, we want to minimize energy and maximize accuracy.
            """
            return not any(
                (df['Mean Energy (J)'] <= energy) & 
                (df['Average Accuracy'] >= accuracy) & 
                ((df['Mean Energy (J)'] < energy) | (df['Average Accuracy'] > accuracy))
            )
        
        df['Pareto_Optimal'] = [
            is_pareto_optimal(row['Mean Energy (J)'], row['Average Accuracy'], df)
            for _, row in df.iterrows()
        ]
        
        # Create plot
        plt.figure(figsize=(7, 4.5))
        
        # Plot non-Pareto points
        non_pareto = df[~df['Pareto_Optimal']]
        sns.scatterplot(
            data=non_pareto,
            x='Mean Energy (J)',
            y='Average Accuracy',
            hue='Base Model',
            style='Quantization',
            s=100,
            alpha=0.6,
            markers=self.markers,
            palette=self.colors,
        )
        
        # Plot Pareto points with larger markers
        pareto = df[df['Pareto_Optimal']]
        sns.scatterplot(
            data=pareto,
            x='Mean Energy (J)',
            y='Average Accuracy',
            hue='Base Model',
            style='Quantization',
            s=200,
            legend=False,
            markers=self.markers,
            palette=self.colors
        )
        
        # Connect Pareto frontier with a line
        pareto_sorted = pareto.sort_values('Mean Energy (J)')
        plt.plot(pareto_sorted['Mean Energy (J)'], 
                pareto_sorted['Average Accuracy'], 
                'k--', 
                alpha=0.5, 
                label='Pareto Frontier')
        
        if log_scale:
            plt.xscale('log')
        plt.title('Energy-Average Accuracy Trade-off')
        plt.xlabel('Mean Energy Consumption (Joules)')
        plt.ylabel('Average Accuracy')
        
        # Add annotations for Pareto-optimal points and high performers
        texts = []
        for _, row in df.iterrows():
            if row['Pareto_Optimal']:
                model_name = "_".join(row['Model'].split('_')[:2])  # Show model family and size
                texts.append(plt.text(
                    row['Mean Energy (J)'], row['Average Accuracy'],
                    f"{model_name} ({row['Quantization']})",
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                ))

        # Adjust text to avoid overlap
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
    
        
        # Add a text box with key statistics
        stats_text = (
            f"Number of models: {len(df)}\n"
            f"Pareto-optimal models: {len(pareto)}\n"
            f"Accuracy range: {df['Average Accuracy'].min():.2f}-{df['Average Accuracy'].max():.2f}\n"
            f"Energy range: {df['Mean Energy (J)'].min():.2f}-{df['Mean Energy (J)'].max():.2f}J"
        )
        plt.text(0.02, 0.02, stats_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                fontsize=8,
                verticalalignment='bottom')

        # Add legend outside the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("output/energy_average_accuracy_tradeoff.pdf")
        
        # Write average accuracy and energy consumption for Pareto optimal models to a file
        pareto_avg_accuracy = pareto['Average Accuracy'].mean()
        pareto_avg_energy = pareto['Mean Energy (J)'].mean()
        dataset_name_str = dataset_name if dataset_name else 'all_datasets'
        
        with open(f"output/pareto_optimal_results_average_{dataset_name_str}.txt", "w") as f:
            f.write(f"Results for dataset {dataset_name_str}:\n")
            f.write(f"Pareto-optimal models average accuracy: {pareto_avg_accuracy:.2f}\n")
            f.write(f"Pareto-optimal models average energy consumption: {pareto_avg_energy:.2f} J\n")
            
            # Write accuracy and energy consumption for each Pareto optimal model
            f.write("\nPareto-optimal models details:\n")
            for _, row in pareto.iterrows():
                f.write(f"Model: {row['Model']}, Average Accuracy: {row['Average Accuracy']:.2f}, Mean Energy (J): {row['Mean Energy (J)']:.2f}\n")
        
        return plt.gcf()

    def analyze_inference_latency(self, separate_correct_incorrect=False):
        """Analyze inference latency for each model on each dataset."""
        latency_data = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            df = self.raw_data[f"{dataset_name}_{model_name}"]
            df['total_duration'] = pd.to_timedelta(df["total_duration"]).dt.total_seconds()
            base_model = "_".join(model_name.split("_")[:2])  # Extract base model name
            if separate_correct_incorrect:
                correct_latency = df[df['evaluation'] == True]['total_duration'].mean()
                incorrect_latency = df[df['evaluation'] == False]['total_duration'].mean()
                latency_data.extend([{
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Base Model': base_model,
                    'Latency (s)': latency,
                    'Evaluation': 'Correct' if evaluation else 'Incorrect'
                } for latency, evaluation in zip(df['total_duration'], df['evaluation'])])
            else:
                latency_data.extend([{
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Base Model': base_model,
                    'Latency (s)': latency
                } for latency in df['total_duration']])
        
        df = pd.DataFrame(latency_data)
        
        # Calculate average latency across all datasets for each model
        avg_latency_df = df.groupby('Model')['Latency (s)'].mean().reset_index()
        avg_latency_df.columns = ['Model', 'Average Latency (s)']
        
        # Merge average latency with the original dataframe
        df = df.merge(avg_latency_df, on='Model')
        
        # Calculate average latency across all models for each dataset
        avg_latency_per_dataset = df.groupby('Dataset')['Latency (s)'].mean().reset_index()
        avg_latency_per_dataset.columns = ['Dataset', 'Average Latency per Dataset (s)']
        
        # Merge average latency per dataset with the original dataframe
        df = df.merge(avg_latency_per_dataset, on='Dataset')

        # Calculate average latency across all models and datasets
        avg_latency = df['Latency (s)'].mean()
        # Calculate the average latency for each model
        avg_latency_per_model = df.groupby('Model')['Latency (s)'].mean().reset_index()
        avg_latency_per_model.columns = ['Model', 'Average Latency (s)']
        # Calculate the average latency for each dataset
        avg_latency_per_dataset = df.groupby('Dataset')['Latency (s)'].mean().reset_index()
        avg_latency_per_dataset.columns = ['Dataset', 'Average Latency (s)']
        # Calculate the average latency for each model on each dataset, i.e., m*n rows where m is the number of models and n is the number of datasets
        avg_latency_per_model_dataset = df.groupby(['Model', 'Dataset'])['Latency (s)'].mean().reset_index()
        avg_latency_per_model_dataset.columns = ['Model', 'Dataset', 'Average Latency (s)']

        # Write the average latency to a file
        with open("output/inference_latency_average.txt", "w") as f:
            f.write(f"Average Latency across all models and datasets: {avg_latency:.2f} s\n")
            f.write("\nAverage Latency per Model:\n")
            for _, row in avg_latency_per_model.iterrows():
                f.write(f"{row['Model']}: {row['Average Latency (s)']:.2f} s\n")
            f.write("\nAverage Latency per Dataset:\n")
            for _, row in avg_latency_per_dataset.iterrows():
                f.write(f"{row['Dataset']}: {row['Average Latency (s)']:.2f} s\n")
            f.write("\nAverage Latency per Model on each Dataset:\n")
            for _, row in avg_latency_per_model_dataset.iterrows():
                f.write(f"{row['Model']} on {row['Dataset']}: {row['Average Latency (s)']:.2f} s\n")

        # Create LaTeX table for average inference latency by model family and quantization level
        model_families = df['Base Model'].unique()
        quantization_levels = ['fp16', 'q8_0', 'q4 variants (avg)', 'q3 variants (avg)']
        latency_table = pd.DataFrame(index=quantization_levels, columns=model_families)

        for model_family in model_families:
            for quant_level in quantization_levels:
                if quant_level == 'q4 variants (avg)':
                    quant_levels = ['q4_1', 'q4_K_M', 'q4_0', 'q4_K_S']
                elif quant_level == 'q3 variants (avg)':
                    quant_levels = ['q3_K_L', 'q3_K_M', 'q3_K_S']
                else:
                    quant_levels = [quant_level]

                latency_values = df[(df['Base Model'] == model_family) & (df['Model'].str.contains('|'.join(quant_levels)))]['Latency (s)']
                if not latency_values.empty:
                    latency_table.loc[quant_level, model_family] = latency_values.mean()
                else:
                    latency_table.loc[quant_level, model_family] = 'N/A'

        # Replace non-numeric values with NaN
        latency_table_numeric = latency_table.apply(pd.to_numeric, errors='coerce')
        # Compute the row-wise mean, ignoring NaN values
        latency_table['Average'] = latency_table_numeric.mean(axis=1, skipna=True)
        # Compute the column-wise mean, ignoring NaN values
        latency_table.loc['Average'] = latency_table_numeric.mean(axis=0, skipna=True)
        # Compute the overall average latency

        latex_table = latency_table.to_latex(
            float_format="%.2f",
            na_rep="N/A",
            caption="Average Inference Latency (seconds) by Model Family and Quantization Level",
            label="tab:avg_latency"
        )

        with open("output/average_inference_latency_table.tex", "w") as f:
            f.write(latex_table)

        # Sort models based on base model and quantization rank
        df['Quantization_Rank'] = df['Model'].apply(lambda x: self._get_quantization_rank(self._extract_quantization_level(x)))
        df = df.sort_values(by=['Base Model', 'Quantization_Rank'])

        # Plot the results
        datasets = df['Dataset'].unique()
        num_datasets = len(datasets)
        fig, axes = plt.subplots(num_datasets, 1, figsize=(12, 2 * num_datasets), sharex=True)
        if num_datasets == 1:
            axes = [axes]

        for ax, dataset in zip(axes, datasets):
            sns.boxplot(
                data=df[df['Dataset'] == dataset],
                x='Model',
                y='Latency (s)',
                hue='Evaluation' if separate_correct_incorrect else 'Base Model',
                ax=ax,
                palette=utils.get_correctness_colors() if separate_correct_incorrect else self.colors,
                showfliers=False,
                legend=separate_correct_incorrect,
            )
            ax.set_title(f'Inference Latency Distribution for {dataset}')
            ax.set_xlabel('Model')
            ax.set_ylabel('Latency (s)')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Add shared legend at the bottom
        model_families = df['Base Model'].unique()
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors.get(family, 'gray'), markersize=10) for family in model_families]
        labels = list(model_families)
        fig.legend(handles, labels, title="Base Model", loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))


        plt.tight_layout()
        plt.savefig("output/inference_latency_distribution.pdf")
        
        # Print summary statistics
        print("\nInference Latency Summary:")
        print(df[['Dataset', 'Model', 'Latency (s)'] + (['Evaluation'] if separate_correct_incorrect else [])])
        
        return plt.gcf()

    def generate_energy_per_response_latex_table_by_model_family(self) -> str:
        """Generate a LaTeX table of average energy consumption per response by model family and quantization level."""
        energy_data = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            base_model = "_".join(model_name.split("_")[:2])  # Extract base model name
            energy_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Base Model': base_model,
                'Mean Energy per Response (J)': metrics.mean_energy
            })
        
        df = pd.DataFrame(energy_data)
        
        # Create LaTeX table for average energy consumption by model family and quantization level
        model_families = df['Base Model'].unique()
        quantization_levels = ['fp16', 'q8_0', 'q4 variants (avg)', 'q3 variants (avg)']
        energy_table = pd.DataFrame(index=quantization_levels, columns=model_families)

        for model_family in model_families:
            for quant_level in quantization_levels:
                if quant_level == 'q4 variants (avg)':
                    quant_levels = ['q4_1', 'q4_K_M', 'q4_0', 'q4_K_S']
                elif quant_level == 'q3 variants (avg)':
                    quant_levels = ['q3_K_L', 'q3_K_M', 'q3_K_S']
                else:
                    quant_levels = [quant_level]

                energy_values = df[(df['Base Model'] == model_family) & (df['Model'].str.contains('|'.join(quant_levels)))]['Mean Energy per Response (J)']
                if not energy_values.empty:
                    energy_table.loc[quant_level, model_family] = energy_values.mean()
                else:
                    energy_table.loc[quant_level, model_family] = 'N/A'

        # Replace non-numeric values with NaN
        energy_table_numeric = energy_table.apply(pd.to_numeric, errors='coerce')
        # Compute the row-wise mean, ignoring NaN values
        energy_table['Average'] = energy_table_numeric.mean(axis=1, skipna=True)
        # Compute the column-wise mean, ignoring NaN values
        energy_table.loc['Average'] = energy_table_numeric.mean(axis=0, skipna=True)

        latex_table = energy_table.to_latex(
            float_format="%.2f",
            na_rep="N/A",
            caption="Average Energy Consumption per Response (Joules) by Model Family and Quantization Level",
            label="tab:avg_energy_per_response"
        )

        # Make sure to escape any underscores
        latex_table = latex_table.replace("_", r"\_")

        return latex_table

def main():
    parser = argparse.ArgumentParser(description='Analyze LLM energy consumption data')
    parser.add_argument('files', nargs='+', help='CSV files to analyze')
    parser.add_argument('--dataset', help='Filter analysis to specific dataset')
    parser.add_argument('--output-dir', default='output', help='Directory for output files')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = EnergyAnalyzer(args.files)
    
    # Generate and save plots
    plots = {
        'energy_accuracy': analyzer.plot_energy_accuracy_tradeoff(args.dataset),
        'energy_accuracy_log': analyzer.plot_energy_accuracy_tradeoff(args.dataset, log_scale=True),
        'energy_distribution': analyzer.plot_energy_distribution(args.dataset),
        'energy_distribution_log': analyzer.plot_energy_distribution(args.dataset, log_scale=True),
        'energy_distribution_per_token': analyzer.plot_energy_distribution(args.dataset, per_token=True),
        'energy_distribution_per_token_log': analyzer.plot_energy_distribution(args.dataset, per_token=True, log_scale=True),
        'energy_distribution_subplots': analyzer.plot_energy_distribution(args.dataset, subplots=True),
        'energy_distribution_subplots_log': analyzer.plot_energy_distribution(args.dataset, log_scale=True, subplots=True),
        'token_impact': analyzer.analyze_token_length_impact(args.dataset),
        'quantization_impact1': analyzer.analyze_quantization_impact(args.dataset),
        'quantization_impact2': analyzer.analyze_quantization_impact2(args.dataset, average=False),
        'quantization_impact2_avg': analyzer.analyze_quantization_impact2(args.dataset, average=True),
        'quantization_impact3': analyzer.plot_quantization_impact(args.dataset),
        'size_impact': analyzer.plot_energy_per_token_vs_size(args.dataset),
        'tokens_per_second': analyzer.plot_tokens_per_second(),
        'tokens_per_second_total_duration': analyzer.plot_tokens_per_second(use_total_duration=True),
        'accuracy_comparison': analyzer.plot_accuracy_subplots_vertical_bars(),
        'average_accuracy': analyzer.plot_average_accuracy(),
        'model_size_vs_energy': analyzer.plot_model_size_vs_metrics(fit_regression=True, in_bytes=False, log_scale=True)[0],
        'model_size_vs_accuracy': analyzer.plot_model_size_vs_metrics(fit_regression=True, in_bytes=False, log_scale=True)[1],
        'model_size_vs_energy_bytes': analyzer.plot_model_size_vs_metrics(fit_regression=True, in_bytes=True, log_scale=False)[0],
        'model_size_vs_accuracy_bytes': analyzer.plot_model_size_vs_metrics(fit_regression=True, in_bytes=True, log_scale=False)[1],
        'model_size_vs_energy_bytes_log': analyzer.plot_model_size_vs_metrics(fit_regression=True, in_bytes=True, log_scale=True)[0],
        'model_size_vs_accuracy_bytes_log': analyzer.plot_model_size_vs_metrics(fit_regression=True, in_bytes=True, log_scale=True)[1],
        'model_size_vs_metrics_grid_log': analyzer.plot_model_size_vs_metrics_grid(fit_regression=True, in_bytes=True, log_scale=True)[0],
        'model_size_vs_metrics_grid_energy_log': analyzer.plot_model_size_vs_metrics_grid(fit_regression=True, in_bytes=True, log_scale=True)[1],
        'model_size_vs_metrics_grid': analyzer.plot_model_size_vs_metrics_grid(fit_regression=True, in_bytes=True, log_scale=False)[0],
        'model_size_vs_metrics_grid_energy': analyzer.plot_model_size_vs_metrics_grid(fit_regression=True, in_bytes=True, log_scale=False)[1],
        'tokens_per_joule_distribution': analyzer.plot_tokens_per_joule_distribution(args.dataset, log_scale=False, subplots=False),
        'tokens_per_joule_distribution_log': analyzer.plot_tokens_per_joule_distribution(args.dataset, log_scale=True, subplots=False),
        'tokens_per_joule_distribution_log_subplots': analyzer.plot_tokens_per_joule_distribution(args.dataset, log_scale=True, subplots=True),
        'response_length_distribution': analyzer.plot_response_length_distribution(args.dataset, log_scale=False, subplots=False),
        'response_length_distribution_log': analyzer.plot_response_length_distribution(args.dataset, log_scale=True, subplots=False),
        'response_length_distribution_log_subplots': analyzer.plot_response_length_distribution(args.dataset, log_scale=True, subplots=True),
        'response_length_distribution_subplots': analyzer.plot_response_length_distribution(args.dataset, log_scale=False, subplots=True),
        'response_length_energy_correlation': analyzer.analyze_response_length_energy_correlation(args.dataset),
        'energy_average_accuracy_tradeoff': analyzer.plot_energy_average_accuracy_tradeoff(log_scale=False),
        'energy_average_accuracy_tradeoff_log': analyzer.plot_energy_average_accuracy_tradeoff(log_scale=True),
        'inference_latency': analyzer.analyze_inference_latency(),
    }
    
    for name, fig in plots.items():
        dataset_name = args.dataset if args.dataset is not None else 'all'
        fig.savefig(Path(args.output_dir) / f"{name}_{dataset_name}.pdf", bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    # Generate and save summary table
    summary_df = analyzer.generate_summary_table()
    summary_df.to_csv(Path(args.output_dir) / "summary_metrics.csv", index=False)
    
    # Generate and save LaTeX tables
    accuracy_latex_table = analyzer.generate_accuracy_latex_table()
    with open(Path(args.output_dir) / "accuracy_table.tex", "w") as f:
        f.write(accuracy_latex_table)
    
    energy_latex_table = analyzer.generate_energy_latex_table()
    with open(Path(args.output_dir) / "energy_table.tex", "w") as f:
        f.write(energy_latex_table)

    energy_per_token_latex_table = analyzer.generate_energy_per_token_latex_table()
    with open(Path(args.output_dir) / "energy_per_token_table.tex", "w") as f:
        f.write(energy_per_token_latex_table)

    energy_per_response_latex_table = analyzer.generate_energy_per_response_latex_table()
    with open(Path(args.output_dir) / "energy_per_response_table.tex", "w") as f:
        f.write(energy_per_response_latex_table)

    response_time_latex_table = analyzer.generate_response_length_latex_table()
    with open(Path(args.output_dir) / "response_time_table.tex", "w") as f:
        f.write(response_time_latex_table)

    energy_per_response_latex_table_by_dataset = analyzer.generate_energy_per_response_latex_table_by_dataset()
    with open(Path(args.output_dir) / "energy_per_response_table_by_dataset.tex", "w") as f:
        f.write(energy_per_response_latex_table_by_dataset)

    energy_per_response_latex_table_by_model_family = analyzer.generate_energy_per_response_latex_table_by_model_family()
    with open(Path(args.output_dir) / "energy_per_response_table_by_model_family.tex", "w") as f:
        f.write(energy_per_response_latex_table_by_model_family)

    # Print some basic statistics
    print("\nSummary Statistics:")
    print(f"Total number of models analyzed: {len(summary_df['Model'].unique())}")
    print(f"Datasets analyzed: {', '.join(summary_df['Dataset'].unique())}")
    print("\nTop 5 most energy-efficient models (by mean energy per token):")
    print(summary_df.nsmallest(5, 'Mean Energy per Token (J)')[['Dataset', 'Model', 'Mean Energy per Token (J)', 'Accuracy']])

    analyzer.print_energy_per_token_summary()
    analyzer.print_accuracy_summary()

if __name__ == "__main__":
    main()

