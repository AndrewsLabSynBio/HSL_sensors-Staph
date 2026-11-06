import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import argparse
from matplotlib.ticker import FormatStrFormatter  # Import FormatStrFormatter

# Set up argument parser to accept the CSV file from the command line
parser = argparse.ArgumentParser(description='Plot Saureus vs Bsub data with error bars and linear regression.')
parser.add_argument('csv_file', type=str, help='Path to the CSV file')
args = parser.parse_args()

# Reading the CSV file from command-line argument
df = pd.read_csv(args.csv_file)

# Averaging *Bsub* and *Saureus* data across the days
df['Bsub_avg'] = df[['Bsub_day1', 'Bsub_day2', 'Bsub_day3']].mean(axis=1)
df['Saureus_avg'] = df[['Saureus_day1', 'Saureus_day2', 'Saureus_day3']].mean(axis=1)

# Calculating standard deviations for error bars
df['Bsub_std'] = df[['Bsub_day1', 'Bsub_day2', 'Bsub_day3']].std(axis=1)
df['Saureus_std'] = df[['Saureus_day1', 'Saureus_day2', 'Saureus_day3']].std(axis=1)

# Log-transforming the averaged values and std
df['Log_Bsub_avg'] = np.log10(df['Bsub_avg'])
df['Log_Saureus_avg'] = np.log10(df['Saureus_avg'])
df['Log_Bsub_std'] = df['Bsub_std'] / (df['Bsub_avg'] * np.log(10))  # Error propagation in log scale
df['Log_Saureus_std'] = df['Saureus_std'] / (df['Saureus_avg'] * np.log(10))  # Error propagation in log scale

# Set font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Plotting the data with different marker shapes from the CSV file
plt.figure(figsize=(3, 3))

for idx, row in df.iterrows():
    marker = row['marker_shape']  # Get the marker shape from the CSV file
    plt.errorbar(row['Log_Bsub_avg'], row['Log_Saureus_avg'], 
                 xerr=row['Log_Bsub_std'], yerr=row['Log_Saureus_std'], 
                 fmt=marker, ecolor=row['c_map'], capsize=5, 
                 markerfacecolor=row['c_map'], markersize=5, markeredgecolor='black')

# Linear regression on log-transformed values
X = df['Log_Bsub_avg'].values.reshape(-1, 1)
y = df['Log_Saureus_avg'].values
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

# R-squared value
r2 = r2_score(y, y_pred)

# Plot the regression line (in black) with reduced thickness
x_vals = np.linspace(df['Log_Bsub_avg'].min(), df['Log_Bsub_avg'].max(), 100).reshape(-1, 1)
y_vals = model.predict(x_vals)
plt.plot(x_vals, y_vals, color='black', linestyle='--', linewidth=1.5, label='Trendline', zorder=0)  # Set trendline to black

# Add R-squared value to the plot (with 2 digits after the decimal)
plt.text(0.05, 0.9, f'R² = {r2:.2f}', transform=plt.gca().transAxes, fontname='Arial')

# Labeling the axes with italic "B. subtilis" and "S. aureus" and placing log10 on a new line
ax = plt.gca()
ax.set_xlabel(r'Promoter output in $\it{B. subtilis}$' + '\n' + r'$\log_{10}$(RPU)', fontname='Arial')
ax.set_ylabel(r'Promoter output in $\it{S. aureus}$' + '\n' + r'$\log_{10}$(RPU)', fontname='Arial')

# Match x-axis tick marks to y-axis tick marks
y_ticks = ax.get_yticks()
ax.set_xticks(np.linspace(-2, 1, len(y_ticks)))  # Ensure consistent tick distribution

# Set x-axis limits from -2 to 1
ax.set_xlim(-2, 1)

# Format x-axis tick labels to show only one digit after the decimal
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Remove gridlines
plt.grid(False)

# Remove the top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save the plot as an image
plt.savefig('saureus_vs_bsub_log_plot_with_regression_3-1-25.svg', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()