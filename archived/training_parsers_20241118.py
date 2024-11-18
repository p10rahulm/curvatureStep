import pandas as pd
import numpy as np
import re
import os


# ----------------------------
# Write the mean loss to file.
# ----------------------------

# List of file paths
file_paths = [
    "outputs/cifar100/cifar100_grouped_train.csv",
    "outputs/dbpedia/dbpedia_grouped_train.csv",
    "outputs/caltech101_resnet/caltech101_resnet_grouped_train.csv",
    "outputs/caltech101_cnn/caltech101_cnn_grouped_train.csv",
    "outputs/amazon-review-polarity/amazon-review-polarity_train_grouped_logs.csv",
    "outputs/cifar10/cifar10_grouped_train.csv",
    "outputs/sogou/sogou_grouped_train.csv",
    "outputs/yelp/yelp_grouped_train.csv",
    "outputs/cola/cola_grouped_train.csv",
    "outputs/oxford_pet/oxford_pet_grouped_train.csv",
    "outputs/fashionmnist/fashionmnist_grouped_train.csv",
    "outputs/arfull/arfull_grouped_train.csv",
    "outputs/flowers102/flowers102_grouped_train.csv",
    "outputs/agnews/agnews_grouped_train.csv",
    "outputs/mnist/mnist_grouped_train.csv",
    "outputs/eurosat/eurosat_grouped_train.csv",
    "outputs/stl10_resnet/stl10_resnet_grouped_train.csv",
    "outputs/cola/cola_grouped_train.csv",
    "outputs/stl10_cnn/stl10_cnn_grouped_train.csv",
    "outputs/imdb/imdb_grouped_train.csv",
    "outputs/reuters/reuters_grouped_train.csv"
]

# Function to process files for a given num_epochs
def process_files(num_epochs, less_discard=True):
    # Dictionary to store aggregated data
    aggregated_data = {}
    datasets = set()
    # Read each file and aggregate the data
    for file_path in file_paths:
        df = pd.read_csv(file_path, delimiter='|')
        
        for index, row in df.iterrows():
            optimizer = row['Optimizer Name']
            mean_losses = row.filter(like='Mean_Training_Loss').values[:num_epochs]  # Restrict to first num_epochs
            dataset_name = file_path.split('/')[1]
            if dataset_name=="mnist":
                print(f"dataset={dataset_name}, len(mean_losses)={len(mean_losses)}")

            # Skip if there are less than num_epochs
            if len(mean_losses) < num_epochs:
                if less_discard:
                    continue
                else:
                    mean_losses = np.pad(mean_losses, (0, num_epochs - len(mean_losses)), constant_values=np.nan)
            else:
                if less_discard:
                    
                    datasets.add(dataset_name)
            
            if optimizer not in aggregated_data:
                aggregated_data[optimizer] = []
            
            aggregated_data[optimizer].append(mean_losses)
    
    # Compute the mean across all files for each optimizer, ignoring NaNs
    mean_data = {}
    for optimizer, values in aggregated_data.items():
        mean_data[optimizer] = np.nanmean(values, axis=0)
    
    # Convert the mean data to a DataFrame
    mean_df = pd.DataFrame(mean_data).transpose()
    
    # Rename the columns to reflect the epochs
    mean_df.columns = [f'Epoch_{i+1}' for i in range(mean_df.shape[1])]
    
    # Print the resulting mean DataFrame
    print(mean_df)
    if less_discard:
        print("included datasets = ",datasets)
    
    # Save the result to a CSV file
    if less_discard:
        mean_df.to_csv(f'outputs/mean_training_loss_{num_epochs}epochs_exact.csv', index=True, sep="|")
    else:
        mean_df.to_csv(f'outputs/mean_training_loss_{num_epochs}epochs.csv', index=True, sep="|")

# Process files for num_epochs 5, 10, 50
for num_epochs in [5, 10, 50]:
    process_files(num_epochs, less_discard=True)

process_files(10, less_discard=False)

# ----------------------------
# Write the final rank to file.
# ----------------------------

# Initialize a dictionary to store the ranks
ranks = {}

for file_path in file_paths:
    df = pd.read_csv(file_path, delimiter='|')
    dataset_name = file_path.split('/')[1]

    # Get the columns for mean training loss
    mean_loss_columns = [col for col in df.columns if col.startswith('Mean_Training_Loss_epoch')]
    last_epoch_column = mean_loss_columns[-1]

    # Rank the optimizers based on the last epoch's mean training loss
    df['Rank'] = df[last_epoch_column].rank()

    # Store the ranks in the dictionary
    for index, row in df.iterrows():
        optimizer_name = row['Optimizer Name']
        rank = row['Rank']
        if optimizer_name not in ranks:
            ranks[optimizer_name] = {}
        ranks[optimizer_name][dataset_name] = rank

# Convert the ranks dictionary to a DataFrame
rank_df = pd.DataFrame(ranks).transpose()

# Calculate the average rank for each optimizer across all datasets
rank_df['Average Rank'] = rank_df.mean(axis=1)

# Save the resulting DataFrame to a CSV file
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
rank_df.to_csv(os.path.join(output_dir, 'mean_rank_optimizer_dataset.csv'), sep='|')

print("Ranking saved to outputs/mean_rank_optimizer_dataset.csv")