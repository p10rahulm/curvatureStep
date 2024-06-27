import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create the dataframe with the specified order
data = {
    'Optimizer Name': ['SimpleSGD', 'HeavyBall', 'NAG', 'NAdamW', 'Adadelta', 'Adagrad', 'RMSProp', 'AdamW', 'RMSPropMomentum', 'Adam', 'AMSGrad', 'NAdam'],
    'caltech 101 resnet': [3.298125, 2.330425, 2.316275, 0.02085, 0.273625, -0.4591, 0.0631, -0.010675, -0.011125, -0.022525, 0.0066, 0.010425],
    'cifar-100': [3.15575, 1.36945, 1.39505, 0.0107, 0.0183, 0.26425, 0.01035, -0.00125, -0.01335, -0.00995, -0.0152, 0.0384],
    'Flowers102': [2.052975, 1.8138, 1.76505, -0.000625, 0.0506, 0.00205, 0.04495, 0.038525, 0.013375, -0.01185, -0.026475, -0.007],
    'oxford pet': [2.010675, 1.8383, 1.8967, -0.00465, 0.059475, -0.020375, 0.00505, 0.028925, -0.0227, -0.116175, -0.039175, -0.017925],
    'caltech 101': [2.773275, 0.954825, 0.9572, -0.011875, 0.0272, 0.264025, 0.012075, 0.0129, -0.0011, 0.0078, -0.000525, 0.057025],
    'stl10': [1.756166667, 0.867833333, 0.872316667, -0.020066667, 0.136233333, 0.2412, 0.0865, -0.008133333, 0.103433333, -0.009, -0.005466667, -0.000266667],
    'cifar-10': [1.71422, 0.88785, 0.89203, 0.00071, 0.11367, -9E-05, 0.01402, 0.00199, -0.00234, -0.00075, 0.00246, 0.00369],
    'reuters': [0.4029, 0.4471, 0.442, -0.3059, 0.1374, 0.2496, 0.2502, 0.7772, -0.1723, 0.0523, -0.1007, 0.3533],
    'ag-news': [0.1829, 0.26855, 0.2859, 1.3958, 0.1291, -1E-04, 0.18595, -0.13145, -0.0224, -5E-05, 0.15645, 0.07615],
    'CoLA': [0.02271, 0.03448, 0.03448, 0.28107, -0.04081, -0.04313, 0.35693, 0.26255, 0.36382, 0.2625, 0.2602, 0.27792],
    'stl10 resnet': [0.830233333, 0.392533333, 0.329266667, 0.001633333, 0.167133333, -0.005633333, -0.006166667, -0.006433333, -0.0236, 0.001766667, 0.0161, 0.019233333],
    'fmnist': [0.46515, 0.28895, 0.2925, 0.0047, 0.101, 0.06765, -0.02555, 1E-04, 0.0041, 0.00005, 0, 0.00515],
    'dbpedia': [2.29705, -0.6127, -0.6179, 0.16675, -0.00735, 0.0055, -0.0038, -0.0066, 0.0022, -0.0518, -0.02815, -0.1436],
    'fmnist.1': [0.39766, 0.22342, 0.22356, -0.00077, 0.00655, 7E-05, 0.00087, 0.00077, 0.00109, -0.00011, -4E-05, 0.00208],
    'amazon full': [0.2921, -0.0513, -0.05535, -0.02695, 0.09875, 0.2235, -0.00455, 0.04765, 0.09355, 0.06425, -0.05035, 0.0325],
    'sogou': [0.08265, 0.0797, 0.082, 0.9489, -0.13535, 0.0185, 0.0161, -0.01605, -0.0007, 0.01985, -0.02805, -0.59615],
    'eurosat': [0.4675, -0.09155, -0.14665, 0.02705, 0.1703, 0.0189, 0.0027, -0.00755, -0.02655, 0.0179, 0.0093, 0.0135],
    'yelp': [0.1717, 0.0571, -0.0803, -0.0046, 0.0082, 0.08235, 0.0525, 0.04315, -0.0569, 0.0855, 0.01135, -0.00055],
    'imdb reviews': [0.14657, 0.00785, 0.00785, 0.00046, 0.06718, 0.00001, 0.00082, -0.00031, -0.00019, 0.00105, 0.00025, 0.0057],
    'mnist': [0.07264, 0.14454, 0.1443, -0.00115, -0.28863, 3E-05, 6E-05, -0.00043, 0.00071, -0.00024, -0.00037, -0.00139],
    'amazon reviews polarity': [0.27215, 0.13325, -1.8373, 0.0031, 0.12735, 0.22605, 0.01475, -0.05975, 0.13145, -0.00625, -0.01025, -0.09515]
}

df = pd.DataFrame(data)
df.set_index('Optimizer Name', inplace=True)

# Set the font size for the entire plot
# plt.rcParams.update({'font.size': 12})

# Create the heatmap
plt.figure(figsize=(12, 6))
# sns.heatmap(df, cmap="crest", vmin=-2, vmax=2, center=0, annot=True, fmt='.2f', cbar_kws={'label': 'Performance'})


# sns.heatmap(df, cmap="RdYlBu", center=+0.2, annot=True, fmt='.2f', cbar_kws={'label': 'Performance'})
# sns.heatmap(df, cmap="Blues", center=0, annot=True, fmt='.2f', cbar_kws={'label': 'Performance'})


cmap = sns.diverging_palette(10, 220, as_cmap=True)
# sns.heatmap(df, cmap=cmap, center=-0.5, annot=True, fmt='.2f', cbar_kws={'label': 'Performance'})

# sns.heatmap(df, cmap="viridis", annot=True, fmt='.2f', cbar_kws={'label': 'Performance'}, linewidths=0)
sns.heatmap(df, cmap="viridis_r", vmax=2, center=+0.5, annot=True, fmt='.2f', cbar_kws={'label': 'Performance'}, linewidths=0)


# Set title with increased font size
plt.title("Optimizer Improvement by Using ACSS Across Datasets", fontsize=20)
# plt.title("Optimizer Improvement by Using ACSS Across Datasets")
# Set axis labels with increased font size
plt.xlabel("Datasets")
plt.ylabel("Optimizers")
plt.xlabel("Datasets", fontsize=17)
plt.ylabel("Optimizers", fontsize=17)
# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent cutting off labels
plt.tight_layout()

plt.savefig("outputs/plots/heatmap_improvement2.pdf")
# Show the plot
plt.show()