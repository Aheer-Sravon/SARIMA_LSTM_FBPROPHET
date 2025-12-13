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
weather_data = pd.read_csv('../data/weather_metrices.csv')

# Extract algorithm names and MAE values
algorithms = weather_data['Algorithm'].tolist()
mae_values = weather_data['MAE'].tolist()

# Create the plot with appropriate sizing for journal (usually single or double column)
# Single column width for journals is typically around 3.3-3.5 inches
fig, ax = plt.subplots(figsize=(3.5, 2.8))  # Single column width, height as needed

# Create vertical bars - using grayscale with patterns for accessibility
colors = ['#4D4D4D', '#7B7B7B', '#A3A3A3', '#C7C7C7', '#E0E0E0']  # Sequential grayscale

bars = ax.bar(range(len(algorithms)), mae_values, color=colors, 
              edgecolor='black', linewidth=0.8, width=0.6, 
              zorder=2)  # zorder to ensure bars are above grid

# Find the best (minimum) MAE
best_idx = np.argmin(mae_values)
best_mae = mae_values[best_idx]
best_algorithm = algorithms[best_idx]

# Highlight the best bar with a different pattern instead of color
bars[best_idx].set_hatch('///')
bars[best_idx].set_edgecolor('#D62728')  # IEEE red for highlight
bars[best_idx].set_linewidth(1.2)

# Add labels and title
ax.set_xlabel('Algorithm', fontweight='medium')
ax.set_ylabel('MAE', fontweight='medium')
ax.set_title('MAE Comparison - Coffe Sales Dataset', fontsize=11, fontweight='medium', pad=10)

# Set x-ticks with algorithm names (consider abbreviations for space)
# If names are too long, consider shorter versions
algorithm_labels = algorithms  # Use as is, or create abbreviations
ax.set_xticks(range(len(algorithms)))
ax.set_xticklabels(algorithm_labels, rotation=45, ha='right', fontsize=9)

# Add value labels on top of bars
for i, (bar, value) in enumerate(zip(bars, mae_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
            f'{value:.2f}', ha='center', va='bottom', fontsize=8)

# Annotate the best performer - subtle and professional
ax.annotate(f'Best: {best_algorithm}\n({best_mae:.2f})',
            xy=(best_idx, best_mae),
            xytext=(best_idx, best_mae + 3.5),
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
ax.set_ylim(0, max(mae_values) * 1.25)

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tight layout with appropriate padding
plt.tight_layout(pad=0.8)

# Save in high resolution formats suitable for publication
plt.savefig('weather_mae.png', dpi=600, bbox_inches='tight')
plt.savefig('weather_mae.pdf', bbox_inches='tight')

plt.show()
