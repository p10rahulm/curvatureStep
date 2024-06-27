import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the provided text
data = {
    "Optimizer": ["SGD", "HeavyBall", "NAG", "RMSProp", "Adagrad", "NAdam", "NAdamW", "Adadelta", "RMSProp-Mom.", "AMSGrad", "Adam", "AdamW"],
    "Mem": ["2x", "3x", "3x", "3x", "3x", "4x", "4x", "3x", "4x", "4x", "4x", "4x"],
    "MEAN": [12.5, 7.9, 6.7, 2.9, 1.5, 1.3, 0.8, 0.8, 0.5, 0.3, 0.2, -0.1],
    "IMD": [19, 18, 18, 6, 1, 5, 1, 1, -4, 2, 6, -5],
    "COL": [6, 4, 4, 20, -7, 6, 6, -9, 18, 5, 7, 6],
    "AGN": [9, 11, 12, 7, -1, 1, 21, 2, -1, 7, -1.5, -7.5],
    "MNI": [22, 17, 17, 1, 1, -1, -1, -1, 1, -1, -1, -1],
    "C101": [20.5, 8, 10, 4, 3, 3, -1, 1, -1, -1.5, 2, 3],
    "C101R": [10, 19, 19, 4, -3, 1, 3, 2, -2, 1, -4, -2],
    "C-10": [20, 10, 10, 3, -1, 2, 2, 1, -1, 1, -1, 1],
    "R-LB": [11, 10.5, 9.5, 6, 3, 7, -6, 3, -3, -3, 3, 3],
    "OP": [13, 16, 18, 1, -1, -1, -1, 2, -1, -2, -6, 2],
    "ARF": [17, -3, -3, -1, 7, 6, -2, 5, 7, -5, 5, 5],
    "ARP": [18, 7, -4, 2, 13, -6, 1, 6, 10, -4, -3, -4],
    "REU": [11, 9, 9, 9, 2, 3, -6, -1, -7, 9, -1, -1],
    "F102": [11, 12, 10, 1, 1, -1, -1, 1, 1, -2, -3, 4],
    "STL10R": [8, 11, 11, -1, -1, 3, 1, 2, -3, 4, 1, -2],
    "C-100": [19, 6, 6, 1, 1, 3, 1, 1, -1, -1, 1, -1],
    "F-LB": [6, 7, 8, -8, 3, 2, -2, 2, 5, -7, 10, 2],
    "yelp": [13, 6, -10, 3, 6, -1, 1, 2, -7, 1, -1, 3],
    "FMNI": [6, 7, 7, 1, 6, 1, -1, 2, 1, -1, -1, 1],
    "STL10": [12, 5, 7, 4, 1, -1, -9, 1, 4, -3, -4, -2],
    "SOG": [6, 3.5, 4.5, 1, 3, -2, 3, 2, -1, -2, 2, -2],
    "DBP": [15, -3.5, -4.5, -1, 1, -3, 5, -6, 1, -1, -7, -2],
    "EUR": [3, -6, -12, 1, 1, 2, 2, -2, -4, 3, -3, -3]
}

df = pd.DataFrame(data)

# Set the 'Optimizer' column as the index
df.set_index('Optimizer', inplace=True)



# Create the heatmap with the specified requirements
plt.figure(figsize=(12, 6))
heatmap = sns.heatmap(df.iloc[:, 1:], annot=True, cmap='RdYlGn', center=0, cbar_kws={'label': 'Performance'})
heatmap.set_yticklabels(heatmap.get_yticklabels())
heatmap.set_xticklabels(heatmap.get_xticklabels(), horizontalalignment='right')
heatmap.set_xlabel('Datasets', fontsize=15)
heatmap.set_ylabel('Optimizers', fontsize=15)
plt.title('Improvement in Rank of Optimizers on using ACSS across Datasets', fontsize=17)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

plt.savefig("outputs/plots/overall_rank_heatmap.pdf")
# Show the plot
plt.show()