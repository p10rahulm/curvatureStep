import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Create the dataframe
data = {
    'Optimizer Name': ['Adadelta-ACSS', 'Adagrad-ACSS', 'Adam-ACSS', 'AdamW-ACSS', 'AMSGrad-ACSS', 'HeavyBall-ACSS', 'NAdam-ACSS', 'NAdamW-ACSS', 'NAG-ACSS', 'RMSProp-ACSS', 'RMSPropMomentum-ACSS', 'SimpleSGD-ACSS', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'AMSGrad', 'HeavyBall', 'NAdam', 'NAdamW', 'NAG', 'RMSProp', 'RMSPropMomentum', 'SimpleSGD'],
    'Caltech101_epoch5': [23, 14, 5, 4, 2, 10, 8, 7, 11, 16, 18, 17, 24, 13, 1, 6, 3, 20, 12, 9, 21, 19, 15, 22],
    'Caltech101_epoch10': [22, 19, 7, 8, 4, 1, 10, 9, 2, 14, 17, 13, 24, 16, 3, 6, 5, 20, 11, 12, 21, 18, 15, 23],
    'CIFAR10_epoch5': [22, 20, 7, 11, 9, 5, 1, 4, 6, 15, 14, 17, 24, 21, 8, 12, 10, 18, 2, 3, 18, 16, 13, 23],
    'CIFAR10_epoch10': [23, 21, 5, 6, 1, 9, 15, 14, 8, 10, 12, 3, 24, 20, 4, 7, 2, 19, 17, 16, 18, 13, 11, 22],
    'Flowers102_epoch5': [23, 2, 8, 3, 4, 9, 15, 13, 10, 16, 18, 11, 24, 1, 5, 7, 6, 20, 14, 12, 21, 17, 19, 22],
    'Flowers102_epoch10': [23, 1, 8, 3, 6, 9, 13, 15, 10, 16, 18, 11, 24, 2, 5, 7, 4, 21, 12, 14, 20, 17, 19, 22],
    'MNIST_epoch5': [24, 20, 7, 8, 5, 2, 14, 17, 1, 11, 12, 3, 23, 21, 6, 9, 4, 18, 15, 16, 18, 10, 13, 22],
    'MNIST_epoch10': [24, 20, 7, 9, 5, 2, 15, 17, 3, 10, 12, 1, 23, 21, 6, 8, 4, 18, 14, 16, 19, 11, 13, 22],
    'STL10_epoch5': [22, 16, 9, 13, 14, 18, 2, 1, 19, 5, 7, 15, 24, 17, 12, 11, 10, 21, 4, 3, 20, 6, 8, 23],
    'STL10_epoch10': [22, 18, 9, 8, 7, 15, 1, 10, 14, 12, 13, 11, 24, 19, 5, 6, 4, 20, 3, 2, 21, 16, 17, 23]
}

df = pd.DataFrame(data)
df.set_index('Optimizer Name', inplace=True)

# Create the heatmap
plt.figure(figsize=(12, 6))


# sns.heatmap(df, cmap="RdYlGn_r", annot=True, fmt='d', cbar_kws={'label': 'Rank'}, linewidths=0)

# Create a custom colormap
colors_list = ['#4575b4', '#ffffff', '#d73027']  # Blue, White, Red
n_bins = 24  # Number of ranks
cmap = colors.LinearSegmentedColormap.from_list("custom", colors_list, N=n_bins)

# In the heatmap function, use:
# sns.heatmap(df, cmap=cmap, annot=True, fmt='d', cbar_kws={'label': 'Rank'})
sns.heatmap(df, cmap="viridis", annot=True, fmt='d', cbar_kws={'label': 'Rank'}, linewidths=0, center=15,vmin=5, vmax=25)
# sns.heatmap(df, cmap="crest_r", annot=True, fmt='d', cbar_kws={'label': 'Rank'}, linewidths=0)

plt.title("Optimizer Rankings Across Datasets and Epochs", fontsize=20)
plt.xlabel("Dataset and Epoch", fontsize=14)
plt.ylabel("Optimizer", fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Show the plot
plt.savefig("outputs/plots/vision_datasets_rank_plots.pdf")
# plt.savefig("outputs/plots/vision_datasets_rank_plots2.pdf")
plt.show()