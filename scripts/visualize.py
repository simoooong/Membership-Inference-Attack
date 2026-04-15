import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Configuration
CLASSIFIER_CHOICE = "logisticregression"
DATASET = "CIFAR-10"
FILE_PATH = os.path.join("..", "artifacts", "results.jsonl")
HYPERPARAMS = ["target_model_learning_rate", "scale", "target_model_epochs_requested"]
METRIC_COL = "mia_precision"

def load_data(file_path, dataset):
    """Loads the JSONL data from the specified path into a Pandas DataFrame."""
    print(f"Loading data from {file_path}...")
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                if record.get("dataset_name") == dataset:
                    data.append(record)
        df = pd.DataFrame(data)
        # Ensure numeric types for analysis
        for hp in HYPERPARAMS:
            if hp in df.columns:
                df[hp] = pd.to_numeric(df[hp], errors='coerce')
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def box_plot_hyperparameter(df, hyperparameter_col, classifier_name, root_path, metric_col=METRIC_COL, dataset_name=DATASET):
    """Generates a box plot for a specific hyperparameter and classifier."""
    filedir = os.path.join(root_path, "box_plots")
    os.makedirs(filedir, exist_ok=True)
    filepath = os.path.join(filedir, f"{hyperparameter_col}.png")

    if df is None or hyperparameter_col not in df.columns: return

    plot_df = df[[hyperparameter_col, metric_col]].copy()
    plot_df[hyperparameter_col] = plot_df[hyperparameter_col].astype(str)

    try:
        sorted_values = sorted(plot_df[hyperparameter_col].unique(), key=float)
    except ValueError:
        sorted_values = sorted(plot_df[hyperparameter_col].unique())

    plt.figure(figsize=(12, 7))
    sns.boxplot(
        x=hyperparameter_col, 
        y=metric_col, 
        data=plot_df, 
        order=sorted_values, 
        palette="viridis", 
        hue=hyperparameter_col, 
        legend=False
    )
    sns.stripplot(x=hyperparameter_col, y=metric_col, data=plot_df, order=sorted_values, color='black', size=4, jitter=True, alpha=0.3)
    plt.title(f'{classifier_name}: {metric_col.upper()} by {hyperparameter_col.replace("_", " ").title()}', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filepath)
    plt.close()

def visualize_distribution(df, classifier_name, root_path, metric_col=METRIC_COL, dataset_name=DATASET):
    """Generates frequency distribution (Histogram + KDE) for a specific classifier."""
    filedir = os.path.join(root_path, "distribution_plots")
    os.makedirs(filedir, exist_ok=True)
    filepath = os.path.join(filedir, f"{classifier_name}.png")


    values = df[metric_col].dropna()
    if values.empty: return

    plt.figure(figsize=(10, 6))
    hist_bins = np.linspace(0.4, 1.0, 31) 
    plt.hist(values, bins=hist_bins, density=True, alpha=0.6, color='#DC143C', edgecolor='black', label='Frequency')
    
    if len(values) > 1:
        try:
            sns.kdeplot(values, color='#B22222', linewidth=3, label='Smoothed Density')
        except: pass
        
    plt.axvline(0.5, color='gray', linestyle='--', label='Random Guess (0.5)')
    plt.axvline(values.mean(), color='blue', linestyle='-', label=f"Mean ({values.mean():.3f})")
    plt.title(f'{dataset_name} | {classifier_name}\nDistribution of {metric_col.upper()}', fontsize=14)
    plt.xlim(0.45, 1.05) 
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(filepath)
    plt.close()

def compare_classifiers(df, root_path):
    """Initial plot to show all classifiers perform similarly."""
    file_path = os.path.join(root_path, "classfier_comparision.png")

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x='mia_classifier_choice', 
        y=METRIC_COL, 
        data=df, 
        palette="Set2", 
        hue='mia_classifier_choice', 
        legend=False
    )
    plt.title(f"Classifier Comparison")
    plt.savefig(file_path)
    plt.close()

def create_heatmap_mean(df: pd.DataFrame, fixed_factor: str, x_axis: str, y_axis: str, root_path, metric=METRIC_COL):
    """Generates a mean heatmap."""
    file_dir = os.path.join(root_path, "heatmaps")
    os.makedirs(file_dir, exist_ok=True)
    filepath = os.path.join(file_dir, f"heatmap_{fixed_factor}_mean.png")

    LABEL_MAP = {
        "target_model_learning_rate": "Learning Rate",
        "scale": "Scale",
        "target_model_epochs_requested": "Number of Epochs",
    }

    def fmt(col):
        return LABEL_MAP.get(col, col)

    group_cols = [y_axis, x_axis]

    for col in group_cols + [metric]:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found for heatmap generation.")
            return None

    summary_df = df.groupby(group_cols)[metric].mean().reset_index()
    mean_pivot = summary_df.pivot(index=y_axis, columns=x_axis, values=metric)
    mean_pivot = mean_pivot.sort_index(ascending=True).sort_index(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.style.use('default')

    X, Y = np.meshgrid(np.arange(len(mean_pivot.columns) + 1), np.arange(len(mean_pivot.index) + 1))
    mesh = ax.pcolormesh(X, Y, mean_pivot.values, cmap='viridis',
                         vmin=df[metric].min(), vmax=df[metric].max(),
                         edgecolors='none', shading='flat')

    fig.colorbar(mesh, ax=ax, label=f'Mean {metric}')
    ax.set_title(f'Mean {metric}\n{fmt(y_axis)} vs. {fmt(x_axis)} (fixed: {fmt(fixed_factor)})', fontsize=13)
    ax.set_xlabel(fmt(x_axis), fontsize=12)
    ax.set_ylabel(fmt(y_axis), fontsize=12)

    ax.set_xticks(np.arange(len(mean_pivot.columns)) + 0.5)
    ax.set_xticklabels([f'{e}' for e in mean_pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mean_pivot.index)) + 0.5)
    ax.set_yticklabels([f'{s:.1f}' if isinstance(s, (float, np.float64)) else f'{s}' for s in mean_pivot.index])

    for i in range(len(mean_pivot.index)):
        for j in range(len(mean_pivot.columns)):
            val = mean_pivot.iloc[i, j]
            if not np.isnan(val):
                txt_color = "white" if val < 0.6 else "black"
                ax.text(j + 0.5, i + 0.5, f"{val:.3f}",
                        ha="center", va="center", color=txt_color, fontsize=9)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_plots():
    data_df = load_data(FILE_PATH, DATASET)
    if data_df is None:
        return

    root_path = os.path.join("..","plots", DATASET, METRIC_COL)
    os.makedirs(root_path, exist_ok=True)
    

    compare_classifiers(data_df, root_path)

    clf_df = data_df[data_df['mia_classifier_choice'] == CLASSIFIER_CHOICE]

    visualize_distribution(clf_df, CLASSIFIER_CHOICE, root_path)

    for hp in HYPERPARAMS:
        box_plot_hyperparameter(clf_df, hp, CLASSIFIER_CHOICE, root_path)

    create_heatmap_mean(
        df=clf_df, 
        fixed_factor='target_model_epochs_requested', 
        y_axis='target_model_learning_rate',
        x_axis='scale',
        root_path=root_path
    )

    create_heatmap_mean(
        df=clf_df, 
        fixed_factor='scale', 
        y_axis='target_model_epochs_requested',
        x_axis='target_model_learning_rate',
        root_path=root_path
    )

    create_heatmap_mean(
        df=clf_df, 
        fixed_factor='target_model_learning_rate', 
        y_axis='target_model_epochs_requested',
        x_axis='scale',
        root_path=root_path
    )

if __name__ == "__main__":
    generate_plots()