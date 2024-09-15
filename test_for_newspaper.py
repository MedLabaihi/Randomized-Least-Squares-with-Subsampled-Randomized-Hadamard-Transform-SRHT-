import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SRHT import*

# Load the data
df = pd.read_csv("Advertising.csv")
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Extract features and target
Newspaper = df['newspaper'].values.reshape(-1, 1)
Sales = df['sales'].values

# Classical least squares regression
x_ls, residual, _, _ = np.linalg.lstsq(Newspaper, Sales, rcond=None)

# Epsilon values to test
epsilons = [0.90, 0.96, 0.99, 1.2]

# Create subplots
fig, axs = plt.subplots(1, 4, figsize=(24, 6))

for i, epsilon in enumerate(epsilons):
    # Perform SRHT sketching and least squares regression
    B = SRHT(Newspaper, Sales, epsilon)
    x_rls, _, _, _ = np.linalg.lstsq(B[0], np.matmul(B[1], B[2]), rcond=None)

    # Plot scatterplot of Newspaper vs Sales
    axs[i].scatter(Newspaper, Sales, color='blue', label='Data points')

    # Plot classical least squares regression line
    axs[i].plot(Newspaper, np.matmul(Newspaper, x_ls), color='black', linewidth=2, label='Classical LS')

    # Plot SRHT randomized least squares regression line
    axs[i].plot(Newspaper, np.matmul(Newspaper, x_rls), color='red', linewidth=1, label=f'RLS (epsilon={epsilon})')

    # Set title and labels
    axs[i].set_title(f'SRHT (epsilon={epsilon})')
    axs[i].set_xlabel('Newspaper advertisement budgets')
    axs[i].set_ylabel('Sales')
    axs[i].legend()

# Main title for the whole figure
fig.suptitle('Newspaper Advertisement Budgets vs Sales - Scatterplot and Fitted Regression Lines', fontsize=16)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the plot as an image
plt.savefig('Newspaper_sales_SRHT_comparison.png', bbox_inches='tight')

# Show the plot
plt.show()
