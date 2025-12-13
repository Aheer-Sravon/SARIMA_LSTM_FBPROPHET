import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set publication-quality style
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'STIXGeneral']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 300

# Load the CSV data
walmart_data = pd.read_csv('../data/walmart_metrices.csv')

# Extract algorithm names and MAPE values
algorithms = walmart_data['Algorithm'].tolist()
mape_values = walmart_data['MAPE'].tolist()

# Create the plot with appropriate sizing for journal
fig, ax = plt.subplots(figsize=(5, 2.8))  # Same width and height

# Create vertical bars - using grayscale with patterns for accessibility
colors = ['#4D4D4D', '#7B7B7B', '#A3A3A3', '#C7C7C7', '#E0E0E0']  # Sequential grayscale

bars = ax.bar(range(len(algorithms)), mape_values, color=colors, 
              edgecolor='black', linewidth=0.8, width=0.6, 
              zorder=2)  # zorder to ensure bars are above grid

# Find the best (minimum) MAPE
best_idx = np.argmin(mape_values)
best_mape = mape_values[best_idx]
best_algorithm = algorithms[best_idx]

# Highlight the best bar with a different pattern instead of color
bars[best_idx].set_hatch('///')
bars[best_idx].set_edgecolor('#D62728')  # IEEE red for highlight
bars[best_idx].set_linewidth(1.2)

# Add labels and title
ax.set_xlabel('Algorithm', fontweight='medium')
ax.set_ylabel('MAPE (%)', fontweight='medium')  # Added % for MAPE
ax.set_title('MAPE Comparison - Walmart Dataset (Store 3)', fontsize=11, fontweight='medium', pad=10)

# Set x-ticks with algorithm names
algorithm_labels = algorithms  # Use as is, or create abbreviations
ax.set_xticks(range(len(algorithms)))
ax.set_xticklabels(algorithm_labels, rotation=45, ha='right', fontsize=9)

# Add value labels on top of bars
for i, (bar, value) in enumerate(zip(bars, mape_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
            f'{value:.2f}%', ha='center', va='bottom', fontsize=8)  # Added % symbol

# Annotate the best performer
# Calculate arrow length as percentage of data range
y_range = max(mape_values) - min(mape_values)
arrow_length = y_range * 0.5  # 20% of the data range

ax.annotate(f'Best: {best_algorithm}\n({best_mape:.2f}%)',
            xy=(best_idx, best_mape),
            xytext=(best_idx, best_mape + arrow_length),
            arrowprops=dict(arrowstyle='->', color='#2E7F8F', lw=1.0, 
                           connectionstyle="arc3,rad=0.1"),
            ha='center', va='bottom',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                     edgecolor="#2E7F8F", linewidth=0.8, alpha=0.95))

# Add grid for better readability - subtle
ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=1)
ax.set_axisbelow(True)  # Grid behind bars

# Adjust y-axis limit to accommodate annotation
ax.set_ylim(0, max(mape_values) * 1.25)

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tight layout with appropriate padding
plt.tight_layout(pad=0.8)

# Save in high resolution formats suitable for publication
plt.savefig('./assets/walmart_mape.png', dpi=600, bbox_inches='tight')
plt.savefig('./assets/walmart_mape.pdf', bbox_inches='tight')

plt.show()
