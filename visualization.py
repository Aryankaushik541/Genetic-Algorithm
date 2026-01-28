"""
Visualization Module for Genetic Algorithm Results
Provides plotting functions for individual and combined function analysis
"""

import matplotlib.pyplot as plt
import numpy as np
from ga_config import FIGURE_SIZE, DPI, PLOT_TOP_N_FUNCTIONS
from ga_algorithm import calculate_statistics

def plot_individual_function(func_name, results, history, stats):
    """
    Create visualization for a single function
    Shows convergence plot and distribution histogram
    
    Args:
        func_name: Name of the function
        results: List of fitness values from multiple runs
        history: Convergence history from one run
        stats: Statistics dictionary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Convergence History
    ax1.plot(history, linewidth=2, color='#2E86AB')
    ax1.set_title(f'{func_name} - Convergence History', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Normalized Fitness (0-1)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Add statistics text
    textstr = f"Final: {history[-1]:.6f}\nMean: {stats['mean']:.6f}\nStd: {stats['std']:.6f}"
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Distribution of Results
    ax2.hist(results, bins=15, color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.6f}")
    ax2.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.6f}")
    ax2.set_title(f'{func_name} - Distribution of Results', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Normalized Fitness (0-1)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.show()

def plot_all_functions_combined(all_results, plot_data):
    """
    Create combined visualization for all functions
    Shows 4 subplots: best performance, average performance, convergence, and std dev
    Fixed to show ALL 15 functions properly
    
    Args:
        all_results: Dictionary with function names as keys and results lists as values
        plot_data: Dictionary with convergence histories
    """
    # Calculate statistics for all functions
    stats_dict = {}
    for func_name, results in all_results.items():
        stats_dict[func_name] = calculate_statistics(results)
    
    # Sort functions by mean performance
    sorted_functions = sorted(stats_dict.items(), key=lambda x: x[1]['mean'])
    function_names = [f[0] for f in sorted_functions]
    
    # Create figure with 4 subplots - larger figure for better visibility
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14), dpi=DPI)
    
    # Plot 1: Best Performance (Top Left)
    mins = [stats_dict[name]['min'] for name in function_names]
    bars1 = ax1.barh(range(len(function_names)), mins, color='#2E86AB', alpha=0.8)
    ax1.set_yticks(range(len(function_names)))
    ax1.set_yticklabels(function_names, fontsize=9)
    ax1.set_title('Best Normalized Fitness by Function', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Best Fitness (0-1)', fontsize=12)
    ax1.set_xlim([0, 1])
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()  # Best at top
    
    # Add value labels
    for i, val in enumerate(mins):
        ax1.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=8)
    
    # Plot 2: Average Performance (Top Right)
    means = [stats_dict[name]['mean'] for name in function_names]
    bars2 = ax2.barh(range(len(function_names)), means, color='#A23B72', alpha=0.8)
    ax2.set_yticks(range(len(function_names)))
    ax2.set_yticklabels(function_names, fontsize=9)
    ax2.set_title('Average Normalized Fitness by Function', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Average Fitness (0-1)', fontsize=12)
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()  # Best at top
    
    # Add value labels
    for i, val in enumerate(means):
        ax2.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=8)
    
    # Plot 3: Convergence of Top Functions (Bottom Left)
    top_functions = function_names[:PLOT_TOP_N_FUNCTIONS]
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_functions)))
    
    for i, func_name in enumerate(top_functions):
        if func_name in plot_data:
            linewidth = 3 if i == 0 else 2
            ax3.plot(plot_data[func_name], label=func_name, 
                    linewidth=linewidth, color=colors[i], alpha=0.8)
    
    ax3.set_title(f'Convergence - Top {PLOT_TOP_N_FUNCTIONS} Functions', 
                  fontweight='bold', fontsize=14)
    ax3.set_xlabel('Generation', fontsize=12)
    ax3.set_ylabel('Normalized Fitness (0-1)', fontsize=12)
    ax3.set_ylim([0, 1])
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Standard Deviation (Bottom Right)
    stds = [stats_dict[name]['std'] for name in function_names]
    bars4 = ax4.barh(range(len(function_names)), stds, color='#F18F01', alpha=0.8)
    ax4.set_yticks(range(len(function_names)))
    ax4.set_yticklabels(function_names, fontsize=9)
    ax4.set_title('Standard Deviation by Function', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Std Dev', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()  # Best at top
    
    # Add value labels
    for i, val in enumerate(stds):
        ax4.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8)
    
    # Main title
    plt.suptitle(f'GA Performance Analysis - All {len(function_names)} Functions (Normalized 0-1)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.show()

def plot_comparison_boxplot(all_results):
    """
    Create box plot comparison of all functions
    
    Args:
        all_results: Dictionary with function names as keys and results lists as values
    """
    # Sort by median performance
    sorted_items = sorted(all_results.items(), 
                         key=lambda x: np.median(x[1]))
    
    function_names = [item[0] for item in sorted_items]
    data = [item[1] for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    bp = ax.boxplot(data, labels=function_names, vert=False, patch_artist=True)
    
    # Customize box plot colors
    for patch in bp['boxes']:
        patch.set_facecolor('#2E86AB')
        patch.set_alpha(0.7)
    
    ax.set_title('Distribution Comparison - All Functions', 
                 fontweight='bold', fontsize=16)
    ax.set_xlabel('Normalized Fitness (0-1)', fontsize=12)
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()

def create_performance_heatmap(all_results):
    """
    Create heatmap showing performance metrics for all functions
    
    Args:
        all_results: Dictionary with function names as keys and results lists as values
    """
    # Calculate statistics
    function_names = list(all_results.keys())
    metrics = ['Min', 'Mean', 'Median', 'Std Dev']
    
    data_matrix = []
    for func_name in function_names:
        stats = calculate_statistics(all_results[func_name])
        data_matrix.append([stats['min'], stats['mean'], 
                          stats['median'], stats['std']])
    
    data_matrix = np.array(data_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(function_names)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(function_names)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(function_names)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.4f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Performance Metrics Heatmap - All Functions', 
                 fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

# Test function
if __name__ == "__main__":
    print("Visualization module loaded successfully")
    print("Available functions:")
    print("  - plot_individual_function()")
    print("  - plot_all_functions_combined()")
    print("  - plot_comparison_boxplot()")
    print("  - create_performance_heatmap()")
