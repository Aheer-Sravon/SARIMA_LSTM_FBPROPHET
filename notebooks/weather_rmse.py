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

# Extract algorithm names and RMSE values
algorithms = weather_data['Algorithm'].tolist()
rmse_values = weather_data['RMSE'].tolist()

# Create the plot
fig, ax = plt.subplots(figsize=(5, 2.8))  # Same width and height

# Create vertical bars - using grayscale with patterns for accessibility
colors = ['#4D4D4D', '#7B7B7B', '#A3A3A3', '#C7C7C7', '#E0E0E0']  # Sequential grayscale

bars = ax.bar(range(len(algorithms)), rmse_values, color=colors, 
              edgecolor='black', linewidth=0.8, width=0.6, 
              zorder=2)  # zorder to ensure bars are above grid

# Find the best (minimum) RMSE
best_idx = np.argmin(rmse_values)
best_rmse = rmse_values[best_idx]
best_algorithm = algorithms[best_idx]

# Highlight the best bar with a different pattern instead of color
bars[best_idx].set_hatch('///')
bars[best_idx].set_edgecolor('#D62728')  # IEEE red for highlight
bars[best_idx].set_linewidth(1.2)

# Add labels and title
ax.set_xlabel('Algorithm', fontweight='medium')
ax.set_ylabel('RMSE', fontweight='medium')  # Changed from MAE to RMSE
ax.set_title('RMSE Comparison - Coffe Sales Dataset', fontsize=11, fontweight='medium', pad=10)  # Changed title

# Set x-ticks
ax.set_xticks(range(len(algorithms)))
ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)

# Add value labels on top of bars - use scientific notation for large values
for i, (bar, value) in enumerate(zip(bars, rmse_values)):
    height = bar.get_height()
    # Use scientific notation for large values
    if value > 10000:
        label = f'{value:.2e}'
    else:
        label = f'{value:.2f}'
    ax.text(bar.get_x() + bar.get_width()/2, height * 1.01,
            label, ha='center', va='bottom', fontsize=8)

# Annotate the best performer
# Calculate arrow length as percentage of data range
y_range = max(rmse_values) - min(rmse_values)
arrow_length = y_range * 1.25  # 15% of the data range

ax.annotate(f'Best: {best_algorithm}\n({best_rmse:.2f})',  # Use scientific notation
            xy=(best_idx, best_rmse),
            xytext=(best_idx, best_rmse + arrow_length),
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
ax.set_ylim(0, max(rmse_values) * 1.25)

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tight layout with appropriate padding
plt.tight_layout(pad=0.8)

# Save in high resolution formats suitable for publication
plt.savefig('weather_rmse.png', dpi=600, bbox_inches='tight')  # Changed filename
plt.savefig('weather_rmse.pdf', bbox_inches='tight')  # Changed filename

plt.show()
