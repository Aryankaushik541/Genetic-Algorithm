import matplotlib.pyplot as plt
import numpy as np

def create_all_graphs(results, plot_data, all_raw_scores, best_function):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    function_names = [r[0] for r in results]
    
    # Graph 1: Best Performance (Top Left)
    mins = [r[1] for r in results]
    ax1.barh(function_names, mins, color='skyblue', alpha=0.8)
    ax1.set_title('Best f(x) by Function', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Best f(x)', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Use log scale only if values vary significantly
    if max(mins) / min([m for m in mins if m > 0] + [1]) > 100:
        ax1.set_xscale('log')
    
    # Graph 2: Average Performance (Top Right)
    means = [r[2] for r in results]
    ax2.barh(function_names, means, color='lightgreen', alpha=0.8)
    ax2.set_title('Average f(x) by Function', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Avg f(x)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    if max(means) / min([m for m in means if m > 0] + [1]) > 100:
        ax2.set_xscale('log')
    
    # Graph 3: Convergence of Best Functions (Bottom Left)
    # Show top 5 best performing functions
    sorted_results = sorted(results, key=lambda x: x[2])[:5]
    
    for i, result in enumerate(sorted_results):
        func_name = result[0]
        if func_name in plot_data:
            linewidth = 3 if func_name == best_function else 2
            ax3.plot(plot_data[func_name], label=func_name, linewidth=linewidth)
    
    ax3.set_title('Convergence - Top 5 Functions', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Generations', fontsize=12)
    ax3.set_ylabel('f(x)', fontsize=12)
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Graph 4: Standard Deviation (Bottom Right)
    stds = [r[4] for r in results]
    bars = ax4.bar(range(len(function_names)), stds, color='salmon', alpha=0.8)
    ax4.set_title('Standard Deviation by Function', fontweight='bold', fontsize=14)
    ax4.set_ylabel('Std Dev', fontsize=12)
    ax4.set_xticks(range(len(function_names)))
    ax4.set_xticklabels(function_names, rotation=45, ha='right', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Main title
    plt.suptitle('GA Performance Analysis - Raw Values', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

if __name__ == "__main__":
    pass