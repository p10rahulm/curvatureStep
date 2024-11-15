import pandas as pd
import numpy as np
import re
import os

import pandas as pd
import numpy as np
import re
import os

def parse_training_logs(log_file_path, dataset_name):
    # Regex patterns
    optimizer_re = re.compile(r'Running (\w+) - Run (\d+)/(\d+)')
    epoch_re = re.compile(r'Epoch (\d+)/(\d+) completed, Average Loss: ([\d.]+)')
    test_re = re.compile(r'Test set: Average loss: ([\d.]+), Accuracy: (\d+)/(\d+) \(([\d.]+)%\)')

    train_data = []
    test_data = []
    current_optimizer = None
    current_run = None

    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            optimizer_match = optimizer_re.search(line)
            if optimizer_match:
                current_optimizer = optimizer_match.group(1)
                current_run = optimizer_match.group(2)
                continue

            epoch_match = epoch_re.search(line)
            if epoch_match:
                epoch_number = epoch_match.group(1)
                avg_loss = epoch_match.group(3)
                train_data.append([current_optimizer, current_run, epoch_number, avg_loss])
                continue

            test_match = test_re.search(line)
            if test_match:
                avg_test_loss = test_match.group(1)
                accuracy = test_match.group(4)
                test_data.append([current_optimizer, current_run, avg_test_loss, accuracy])

    # Create and format train_full_logs
    train_df = pd.DataFrame(train_data, columns=[
        'Optimizer Name', 'Running loop number', 'Epoch number', 'Average Training Loss'])
    train_df['Optimizer Name'] = train_df['Optimizer Name'].fillna('')
    train_df['Running loop number'] = train_df['Running loop number'].fillna('')
    
    # Create and format test_full_logs
    test_df = pd.DataFrame(test_data, columns=[
        'Optimizer Name', 'Running loop number', 'Average Test set Loss', 'Average Test set accuracy'])
    
    # Convert numeric columns
    for df in [train_df, test_df]:
        for col in df.columns:
            if any(x in col.lower() for x in ['loss', 'accuracy', 'number']):
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create output directory
    output_dir = f'outputs/{dataset_name.lower()}'
    os.makedirs(output_dir, exist_ok=True)

    # Save full logs with specified format
    train_df.to_csv(f'{output_dir}/{dataset_name.lower()}_train_full_logs.csv', sep='|')
    test_df.to_csv(f'{output_dir}/{dataset_name.lower()}_test_full_logs.csv', sep='|')

    # Create train_grouped_logs
    train_grouped = train_df.groupby(['Optimizer Name', 'Epoch number'], as_index=False).agg(
        Mean_Training_Loss=('Average Training Loss', 'mean'),
        Std_Training_Loss=('Average Training Loss', 'std')
    )

    # Pivot and format for train_grouped_logs
    pivot_train_df = pd.pivot_table(
        train_grouped,
        index='Optimizer Name',
        columns='Epoch number',
        values=['Mean_Training_Loss', 'Std_Training_Loss']
    )

    # Flatten column names
    pivot_train_df.columns = [
        f'{"Mean_Training_Loss" if "mean" in col[0].lower() else "Std_Training_Loss"}_epoch{int(col[1])}'
        for col in pivot_train_df.columns
    ]

    # Sort columns to match required format
    mean_cols = sorted([col for col in pivot_train_df.columns if 'Mean' in col],
                      key=lambda x: int(x.split('epoch')[1]))
    std_cols = sorted([col for col in pivot_train_df.columns if 'Std' in col],
                     key=lambda x: int(x.split('epoch')[1]))
    pivot_train_df = pivot_train_df[mean_cols + std_cols].reset_index()

    # Create test_grouped_logs
    test_grouped = test_df.groupby('Optimizer Name', as_index=False).agg({
        'Average Test set Loss': ['mean', 'std'],
        'Average Test set accuracy': ['mean', 'std']
    })
    
    # Flatten column names for test_grouped
    test_grouped.columns = ['Optimizer Name', 
                          'Mean_Test_Set_Loss', 'Std_Test_Set_Loss',
                          'Mean_Test_Set_Accuracy', 'Std_Test_Set_Accuracy']

    # Save grouped logs
    pivot_train_df.to_csv(f'{output_dir}/{dataset_name.lower()}_train_grouped_logs.csv', 
                         index=False, sep='|')
    test_grouped.to_csv(f'{output_dir}/{dataset_name.lower()}_test_grouped_logs.csv', 
                       index=False, sep='|')

    return train_df, test_df, pivot_train_df, test_grouped


# Example usage for CIFAR-10 and MNIST
# parse_training_logs('outputs/cifar10_training_run_logs.txt', 'Cifar10')
# parse_training_logs('outputs/fashion_mnist_largebatch_training_run_logs.txt', 'FashionMNIST-largebatch')
parse_training_logs('logs/fashion_mnist_20241114_234656.txt', 'FashionMNIST')
# parse_training_logs('outputs/mnist_training_run_logs.txt', 'MNIST')
# parse_training_logs('outputs/imdb_training_run_logs.txt', 'IMDB')
# parse_training_logs('outputs/cola_training_run_logs.txt', 'CoLA')
# parse_training_logs('outputs/cola_bertpt_training_run_logs.txt', 'CoLABert')
# parse_training_logs('outputs/ag_news_training_run_logs.txt', 'AGNews')
# parse_training_logs('outputs/flowers102_training_run_logs.txt', 'Flowers102')
# parse_training_logs('outputs/stl10_cnn_training_run_logs.txt', 'STL10')
# parse_training_logs('outputs/stl10_training_run_logs.txt', 'STL10-Resnet')
# parse_training_logs('outputs/caltech101cnn_training_run_logs.txt', 'Caltech101')
# parse_training_logs('outputs/eurosat_training_run_logs.txt', 'EuroSAT')
# parse_training_logs('outputs/oxford_pet_collated.txt', 'OxfordPet')
# parse_training_logs('outputs/yelp_training_run_logs.txt', 'yelp')
# parse_training_logs('outputs/dbpedia_training_run_logs.txt', 'dbpedia')
# parse_training_logs('outputs/cifar100_training_run_logs.txt', 'CIFAR-100')
# parse_training_logs('outputs/caltech101_compiled_resnet.txt', 'Caltech101-resnet')
# parse_training_logs('outputs/reuters_training_run_logs.txt', 'reuters')
# parse_training_logs('outputs/reuters_largebatch_training_run_logs.txt', 'reuters-large-batch')
# parse_training_logs('outputs/sogou_training_run_logs.txt', 'sogou-news')
# parse_training_logs('outputs/arfull_training_run_logs.txt', 'amazon-review-full')
# parse_training_logs('outputs/arpolarity_training_run_logs.txt', 'amazon-review-polarity')


# # ----------------------------
# # Write the mean loss to file.
# # ----------------------------

# # List of file paths
# file_paths = [
#     "outputs/cifar-100/cifar-100_train_grouped_logs.csv",
#     "outputs/dbpedia/dbpedia_train_grouped_logs.csv",
#     "outputs/caltech101-resnet/caltech101-resnet_train_grouped_logs.csv",
#     "outputs/amazon-review-polarity/amazon-review-polarity_train_grouped_logs.csv",
#     "outputs/cifar10/cifar10_train_grouped_logs.csv",
#     "outputs/sogou-news/sogou-news_train_grouped_logs.csv",
#     "outputs/yelp/yelp_train_grouped_logs.csv",
#     "outputs/colabert/colabert_train_grouped_logs.csv",
#     "outputs/oxfordpet/oxfordpet_train_grouped_logs.csv",
#     "outputs/reuters-large-batch/reuters-large-batch_train_grouped_logs.csv",
#     "outputs/caltech101/caltech101_train_grouped_logs.csv",
#     "outputs/fashionmnist/fashionmnist_train_grouped_logs.csv",
#     "outputs/amazon-review-full/amazon-review-full_train_grouped_logs.csv",
#     "outputs/fashionmnist-largebatch/fashionmnist-largebatch_train_grouped_logs.csv",
#     "outputs/flowers102/flowers102_train_grouped_logs.csv",
#     "outputs/agnews/agnews_train_grouped_logs.csv",
#     "outputs/mnist/mnist_train_grouped_logs.csv",
#     "outputs/eurosat/eurosat_train_grouped_logs.csv",
#     "outputs/stl10-resnet/stl10-resnet_train_grouped_logs.csv",
#     "outputs/cola/cola_train_grouped_logs.csv",
#     "outputs/stl10/stl10_train_grouped_logs.csv",
#     "outputs/imdb/imdb_train_grouped_logs.csv",
#     "outputs/reuters/reuters_train_grouped_logs.csv"
# ]

# # Function to process files for a given num_epochs
# def process_files(num_epochs, less_discard=True):
#     # Dictionary to store aggregated data
#     aggregated_data = {}
#     datasets = set()
#     # Read each file and aggregate the data
#     for file_path in file_paths:
#         df = pd.read_csv(file_path, delimiter='|')
        
#         for index, row in df.iterrows():
#             optimizer = row['Optimizer Name']
#             mean_losses = row.filter(like='Mean_Training_Loss').values[:num_epochs]  # Restrict to first num_epochs
#             dataset_name = file_path.split('/')[1]
#             if dataset_name=="mnist":
#                 print(f"dataset={dataset_name}, len(mean_losses)={len(mean_losses)}")

#             # Skip if there are less than num_epochs
#             if len(mean_losses) < num_epochs:
#                 if less_discard:
#                     continue
#                 else:
#                     mean_losses = np.pad(mean_losses, (0, num_epochs - len(mean_losses)), constant_values=np.nan)
#             else:
#                 if less_discard:
                    
#                     datasets.add(dataset_name)
            
#             if optimizer not in aggregated_data:
#                 aggregated_data[optimizer] = []
            
#             aggregated_data[optimizer].append(mean_losses)
    
#     # Compute the mean across all files for each optimizer, ignoring NaNs
#     mean_data = {}
#     for optimizer, values in aggregated_data.items():
#         mean_data[optimizer] = np.nanmean(values, axis=0)
    
#     # Convert the mean data to a DataFrame
#     mean_df = pd.DataFrame(mean_data).transpose()
    
#     # Rename the columns to reflect the epochs
#     mean_df.columns = [f'Epoch_{i+1}' for i in range(mean_df.shape[1])]
    
#     # Print the resulting mean DataFrame
#     print(mean_df)
#     if less_discard:
#         print("included datasets = ",datasets)
    
#     # Save the result to a CSV file
#     if less_discard:
#         mean_df.to_csv(f'outputs/mean_training_loss_{num_epochs}epochs_exact.csv', index=True, sep="|")
#     else:
#         mean_df.to_csv(f'outputs/mean_training_loss_{num_epochs}epochs.csv', index=True, sep="|")

# # Process files for num_epochs 5, 10, 50
# for num_epochs in [5, 10, 50]:
#     process_files(num_epochs, less_discard=True)

# process_files(10, less_discard=False)

# # ----------------------------
# # Write the final rank to file.
# # ----------------------------

# # Initialize a dictionary to store the ranks
# ranks = {}

# for file_path in file_paths:
#     df = pd.read_csv(file_path, delimiter='|')
#     dataset_name = file_path.split('/')[1]

#     # Get the columns for mean training loss
#     mean_loss_columns = [col for col in df.columns if col.startswith('Mean_Training_Loss_epoch')]
#     last_epoch_column = mean_loss_columns[-1]

#     # Rank the optimizers based on the last epoch's mean training loss
#     df['Rank'] = df[last_epoch_column].rank()

#     # Store the ranks in the dictionary
#     for index, row in df.iterrows():
#         optimizer_name = row['Optimizer Name']
#         rank = row['Rank']
#         if optimizer_name not in ranks:
#             ranks[optimizer_name] = {}
#         ranks[optimizer_name][dataset_name] = rank

# # Convert the ranks dictionary to a DataFrame
# rank_df = pd.DataFrame(ranks).transpose()

# # Calculate the average rank for each optimizer across all datasets
# rank_df['Average Rank'] = rank_df.mean(axis=1)

# # Save the resulting DataFrame to a CSV file
# output_dir = 'outputs'
# os.makedirs(output_dir, exist_ok=True)
# rank_df.to_csv(os.path.join(output_dir, 'mean_rank_optimizer_dataset.csv'), sep='|')

# print("Ranking saved to outputs/mean_rank_optimizer_dataset.csv")