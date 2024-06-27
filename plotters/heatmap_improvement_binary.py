import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Create the dataframe
data = {
    'Optimizer Name': ['SimpleSGD', 'HeavyBall', 'Adadelta', 'RMSProp', 'NAG', 'Adagrad', 'NAdam', 'NAdamW', 'Adam', 'AdamW', 'AMSGrad', 'RMSPropMomentum'],
    'imdb reviews': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    'CoLA': [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    'fmnist': [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    'cifar-10': [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    'fmnist.1': [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1],
    'caltech 101': [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    'reuters': [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    'ag-news': [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    'Flowers102': [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    'cifar-100': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    'stl10': [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    'eurosat': [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    'caltech 101 resnet': [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    'yelp': [1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0],
    'amazon full': [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1],
    'sogou': [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    'amazon reviews polarity': [1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    'mnist': [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
    'oxford pet': [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    'dbpedia': [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
}

df = pd.DataFrame(data)
df.set_index('Optimizer Name', inplace=True)

# Set the font size for the entire plot
# plt.rcParams.update({'font.size': 12})

# Create the heatmap
# plt.figure(figsize=(20, 10))
plt.figure(figsize=(12, 6))
# sns.heatmap(df, cmap="YlGnBu", cbar_kws={'label': 'Method worked'}, annot=True, fmt='d')
# sns.heatmap(df, cmap="viridis_r", cbar_kws={'label': 'Method worked'}, annot=True, fmt='d')
cmap = sns.color_palette("Paired", 2)



# Extract blue and green from viridis
# viridis = plt.cm.get_cmap('viridis')
# blue = viridis(0.1)  # Blue color from viridis
# green = viridis(0.6)  # Green color from viridis

# Create a custom colormap
# cmap = ListedColormap([blue, green])
# cmap = sns.color_palette(["#0077BB", "#33BBEE"])
cmap = sns.color_palette(["#60d76f", "#3761c3"])


sns.heatmap(df, cmap=cmap, cbar_kws={'label': 'ACSS Effectiveness'}, annot=True, fmt='d')

# Set title with increased font size
# plt.title("Effectiveness of ACSS Across Optimizers and Datasets", fontsize=27)
plt.title("Effectiveness of ACSS Across Optimizers and Datasets", fontsize=20)
# Set axis labels with increased font size
plt.xlabel("Datasets", fontsize=17)
plt.ylabel("Optimizers", fontsize=17)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent cutting off labels
plt.tight_layout()

plt.savefig("outputs/plots/heatmap_better_rank.pdf")
# Show the plot
plt.show()