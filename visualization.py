"""
Visualization for GA Results
"""

import matplotlib.pyplot as plt
import numpy as np
from config import FIGURE_SIZE, DPI
from ga_algorithm import calculate_statistics

def plot_individual_function(func_name, results, history, stats):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convergence plot
    ax1.plot(history, linewidth=2, color='#2E86AB')
    ax1.set_title(f'{func_name} - Convergence', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness (0-1)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Stats text
    textstr = f"Final: {history[-1]:.6f}\nMean: {stats['mean']:.6f}\nStd: {stats['std']:.6f}"
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Distribution plot
    ax2.hist(results, bins=15, color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.6f}")
    ax2.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.6f}")
    ax2.set_title(f'{func_name} - Distribution', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Fitness (0-1)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.show()

def plot_all_functions_combined(all_results, plot_data):
    # Calculate stats
    stats_dict = {}
    for func_name, results in all_results.items():
        stats_dict[func_name] = calculate_statistics(results)
    
    # Sort by mean
    sorted_functions = sorted(stats_dict.items(), key=lambda x: x[1]['mean'])
    function_names = [f[0] for f in sorted_functions]
    
    # Create 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14), dpi=DPI)
    
    # 1. Best Performance
    mins = [stats_dict[name]['min'] for name in function_names]
    ax1.barh(range(len(function_names)), mins, color='#2E86AB', alpha=0.8)
    ax1.set_yticks(range(len(function_names)))
    ax1.set_yticklabels(function_names, fontsize=9)
    ax1.set_title('Best Fitness', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Best (0-1)', fontsize=12)
    ax1.set_xlim([0, 1])
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    for i, val in enumerate(mins):
        ax1.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=8)
    
    # 2. Average Performance
    means = [stats_dict[name]['mean'] for name in function_names]
    ax2.barh(range(len(function_names)), means, color='#A23B72', alpha=0.8)
    ax2.set_yticks(range(len(function_names)))
    ax2.set_yticklabels(function_names, fontsize=9)
    ax2.set_title('Average Fitness', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Mean (0-1)', fontsize=12)
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    for i, val in enumerate(means):
        ax2.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=8)
    
    # 3. Convergence (top 5)
    top_functions = function_names[:5]
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_functions)))
    
    for i, func_name in enumerate(top_functions):
        if func_name in plot_data:
            linewidth = 3 if i == 0 else 2
            ax3.plot(plot_data[func_name], label=func_name, 
                    linewidth=linewidth, color=colors[i], alpha=0.8)
    
    ax3.set_title('Convergence - Top 5', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Generation', fontsize=12)
    ax3.set_ylabel('Fitness (0-1)', fontsize=12)
    ax3.set_ylim([0, 1])
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Standard Deviation
    stds = [stats_dict[name]['std'] for name in function_names]
    ax4.barh(range(len(function_names)), stds, color='#F18F01', alpha=0.8)
    ax4.set_yticks(range(len(function_names)))
    ax4.set_yticklabels(function_names, fontsize=9)
    ax4.set_title('Standard Deviation', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Std Dev', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()
    
    for i, val in enumerate(stds):
        ax4.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8)
    
    plt.suptitle(f'GA Performance - All {len(function_names)} Functions', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.show()
