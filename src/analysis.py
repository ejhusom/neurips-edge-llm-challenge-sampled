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
        self.colors = utils.get_colors()
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
                    'Model Family': model_family
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
        plt.figure(figsize=(9, 7))
        
        # Plot non-Pareto points
        non_pareto = df[~df['Pareto_Optimal']]
        sns.scatterplot(
            data=non_pareto,
            x='Mean Energy (J)',
            y='Accuracy',
            hue='Dataset' if dataset_name is None else 'Model Family',
            style='Quantization',
            s=100,
            alpha=0.6,
            palette=self.colors
        )
        
        # Plot Pareto points with larger markers
        pareto = df[df['Pareto_Optimal']]
        sns.scatterplot(
            data=pareto,
            x='Mean Energy (J)',
            y='Accuracy',
            hue='Dataset' if dataset_name is None else 'Model Family',
            style='Quantization',
            s=200,
            legend=False,
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
        for _, row in df.iterrows():
            if row['Pareto_Optimal']:
                model_name = "_".join(row['Model'].split('_')[:2])  # Show model family and size
                plt.annotate(
                    f"{model_name}\n({row['Quantization']})",
                    (row['Mean Energy (J)'], row['Accuracy']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.0)
                )
        
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
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
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
                    'Model Family': model_family
                } for energy in df[column]])
        
        df = pd.DataFrame(data)

        if subplots:
            datasets = df['Dataset'].unique()
            num_datasets = len(datasets)
            fig, axes = plt.subplots(num_datasets, 1, figsize=(12, 6 * num_datasets), sharex=True)
            if num_datasets == 1:
                axes = [axes]

            for ax, dataset in zip(axes, datasets):
                sns.boxplot(
                    data=df[df['Dataset'] == dataset],
                    x='Model',
                    y='Energy (J)',
                    hue='Model Family',
                    ax=ax,
                    palette=self.colors
                )
                ax.set_title(f'Energy Consumption Distribution by Model for {dataset}')
                ax.set_xlabel('Model')
                ax.set_ylabel('Energy (J)')
                ax.tick_params(axis='x', rotation=45)
                if log_scale:
                    ax.set_yscale('log')
            plt.tight_layout()
        else:
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=df,
                x='Model',
                y='Energy (J)',
                hue='Model Family',
                palette=self.colors
            )
            plt.xticks(rotation=45, ha='right')
            if log_scale:
                plt.yscale('log')
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

    def analyze_quantization_impact2(self, dataset_name: str = None):
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
                all_quant_levels.add(variant['quantization'])
            
            # Sort relative_metrics based on the defined order
            relative_metrics.sort(key=lambda x: get_quantization_rank(x['Quantization']))
            
            # Get x-positions that maintain the order
            quant_levels = [m['Quantization'] for m in relative_metrics]
            
            # Plot energy savings
            ax1.scatter(
                quant_levels,
                [m['Energy Saving (%)'] for m in relative_metrics],
                label=base_model,
                marker='o'
            )
            
            # Plot accuracy loss
            ax2.scatter(
                quant_levels,
                [m['Accuracy Loss (pp)'] for m in relative_metrics],
                label=base_model,
                marker='o'
            )
            
            # Connect points with lines
            ax1.plot(
                quant_levels,
                [m['Energy Saving (%)'] for m in relative_metrics],
                alpha=0.5
            )
            ax2.plot(
                quant_levels,
                [m['Accuracy Loss (pp)'] for m in relative_metrics],
                alpha=0.5
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
                        'Model Family': model_family
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
        model.fit(np.log(x), np.log(y))
        
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
                    f"{row['Model Family']}\n({row['Quantization']})",
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
        """Generate a LaTeX table of average accuracy for each model on each dataset."""
        accuracy_data = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            accuracy_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': metrics.accuracy
            })
        
        df = pd.DataFrame(accuracy_data)
        pivot_df = df.pivot(index='Model', columns='Dataset', values='Accuracy')

        latex_table = pivot_df.to_latex(
            float_format="%.2f",
            na_rep="-",
            caption="Average Accuracy of Models on Various Datasets",
            label="tab:accuracy"
        )

        # Make sure to escape any underscores
        latex_table = latex_table.replace("_", r"\_")

        return latex_table

    def generate_energy_latex_table(self) -> str:
        """Generate a LaTeX table of mean energy consumption for each model on each dataset."""
        energy_data = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            energy_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Mean Energy per Token (J/tok)': metrics.mean_energy_per_token
            })
        
        df = pd.DataFrame(energy_data)
        pivot_df = df.pivot(index='Model', columns='Dataset', values='Mean Energy per Token (J/tok)')

        latex_table = pivot_df.to_latex(
            float_format="%.2f",
            na_rep="-",
            caption="Mean Energy Consumption of Models on Various Datasets",
            label="tab:energy"
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
        plt.savefig("average_accuracy_across_models_datasets.pdf")
        plt.show()

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
        # Plot the accuracy results
        accuracy = {}
        datasets = set()

        for (dataset, model), metrics in self.metrics.items():
            if model not in accuracy:
                accuracy[model] = {}
            accuracy[model][dataset] = metrics.accuracy
            datasets.update(accuracy[model].keys())

        # datasets = sorted(datasets)
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
                    colors.append(utils.get_color_for_model(model))
                else:
                    models.append(model)
                    accuracies.append(0)
                    colors.append('gray')

            ax.bar(models, accuracies, color=colors)
            ax.set_xticklabels(models, rotation=45, ha='right')

        # Add legend
        utils.plot_legend(axes[0], location='outside')
        plt.tight_layout()
        plt.savefig("accuracy_comparison.pdf")
        plt.show()

    def plot_tokens_per_second(self):
        """Plot the tokens per second for all models."""
        tokens_per_second_data = []
        for (dataset_name, model_name), metrics in self.metrics.items():
            df = self.raw_data[f"{dataset_name}_{model_name}"]
            tokens_per_second = df["eval_count"] / (df["eval_duration"] * 10**-9)
            tokens_per_second_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Tokens per Second': tokens_per_second.mean(),
                'Quantization': metrics.quantization_level,
                'Model Family': "_".join(model_name.split("_")[:2])  # Extract model family
            })
        
        df = pd.DataFrame(tokens_per_second_data)
        pivot_df = df.pivot(index='Model', columns='Dataset', values='Tokens per Second')

        # Sort the models according to the QUANTIZATION_ORDER
        df['Quantization_Rank'] = df['Quantization'].apply(self._get_quantization_rank)
        df = df.sort_values(by='Quantization_Rank')

        # Plot the tokens per second with horizontal bars
        ax = pivot_df.plot(kind='barh', figsize=(7, 9), color=[self.colors.get(dataset, 'gray') for dataset in pivot_df.columns])
        plt.xlabel('Tokens per Second')
        plt.ylabel('Model')
        plt.title('Tokens per Second for All Models')

        # Color the background of the tick labels according to the model family
        for tick_label in ax.get_yticklabels():
            model_name = tick_label.get_text()
            model_family = "_".join(model_name.split("_")[:2])
            color = self.colors.get(model_family, 'gray')
            tick_label.set_bbox(dict(facecolor=color, edgecolor='none', alpha=0.5))
            # tick_label.set_color(color)

        # Add legend for model families
        model_families = df['Model Family'].unique()
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
        plt.savefig("tokens_per_second.pdf")
        # plt.show()


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
        'energy_accuracy_log': analyzer.plot_energy_accuracy_tradeoff(args.dataset, log_scale=True),
        'energy_distribution': analyzer.plot_energy_distribution(args.dataset),
        'energy_distribution_log': analyzer.plot_energy_distribution(args.dataset, log_scale=True),
        'energy_distribution_per_token': analyzer.plot_energy_distribution(args.dataset, per_token=True),
        'energy_distribution_per_token_log': analyzer.plot_energy_distribution(args.dataset, per_token=True, log_scale=True),
        'energy_distribution_subplots': analyzer.plot_energy_distribution(args.dataset, subplots=True),
        'energy_distribution_subplots_log': analyzer.plot_energy_distribution(args.dataset, log_scale=True, subplots=True),
        'token_impact': analyzer.analyze_token_length_impact(args.dataset),
        # 'quantization_impact': analyzer.analyze_quantization_impact2(args.dataset)
        'size_impact': analyzer.plot_energy_per_token_vs_size(args.dataset),
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
    
    # Plot and save average accuracy
    # analyzer.plot_average_accuracy()
    # analyzer.plot_average_accuracy_per_dataset()
    # analyzer.plot_accuracy_subplots_vertical_bars()
    analyzer.plot_tokens_per_second()

    # Print some basic statistics
    print("\nSummary Statistics:")
    print(f"Total number of models analyzed: {len(summary_df['Model'].unique())}")
    print(f"Datasets analyzed: {', '.join(summary_df['Dataset'].unique())}")
    print("\nTop 5 most energy-efficient models (by mean energy per token):")
    print(summary_df.nsmallest(5, 'Mean Energy per Token (J)')[['Dataset', 'Model', 'Mean Energy per Token (J)', 'Accuracy']])

if __name__ == "__main__":
    main()
