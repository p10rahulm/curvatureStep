import pandas as pd
import numpy as np
import re
import os



def parse_training_logs(log_file_path, dataset_name):
    # Regular expressions to extract information
    optimizer_re = re.compile(rf'Running {dataset_name} training with Optimizer = (\w+)')
    params_re = re.compile(r'params= ({.+})')
    loop_re = re.compile(r'Running Loop: (\d+)/(\d+)')
    epoch_re = re.compile(rf'Epoch (\d+)/(\d+) completed, Average Loss: ([\d.]+)')
    test_re = re.compile(r'Test set: Average loss: ([\d.]+), Accuracy: (\d+)/(\d+) \(([\d.]+)%\)')

    # Lists to store data for DataFrames
    train_data = []
    test_data = []

    # Variables to hold current state
    current_optimizer = None
    current_loop = None

    # Read the log file
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            optimizer_match = optimizer_re.search(line)
            if optimizer_match:
                current_optimizer = optimizer_match.group(1)
                continue

            loop_match = loop_re.search(line)
            if loop_match:
                current_loop = loop_match.group(1)
                continue

            epoch_match = epoch_re.search(line)
            if epoch_match:
                epoch_number = epoch_match.group(1)
                avg_loss = epoch_match.group(3)
                train_data.append([current_optimizer, current_loop, epoch_number, avg_loss])
                continue

            test_match = test_re.search(line)
            if test_match:
                avg_test_loss = test_match.group(1)
                accuracy = test_match.group(4)
                test_data.append([current_optimizer, current_loop, avg_test_loss, accuracy])
                continue

    # Create DataFrames
    train_df = pd.DataFrame(train_data, columns=[
        'Optimizer Name', 'Running loop number', 'Epoch number', 'Average Training Loss'])

    test_df = pd.DataFrame(test_data, columns=[
        'Optimizer Name', 'Running loop number', 'Average Test set Loss', 'Average Test set accuracy'])

    # Convert the relevant columns to numeric types
    train_df['Average Training Loss'] = pd.to_numeric(train_df['Average Training Loss'], errors='coerce')
    test_df['Average Test set Loss'] = pd.to_numeric(test_df['Average Test set Loss'], errors='coerce')
    test_df['Average Test set accuracy'] = pd.to_numeric(test_df['Average Test set accuracy'], errors='coerce')

    # Create output directory if it doesn't exist
    output_dir = f'outputs/{dataset_name.lower()}'
    os.makedirs(output_dir, exist_ok=True)

    # Save full logs to CSV files
    train_df.to_csv(f'{output_dir}/{dataset_name.lower()}_train_full_logs.csv', index=True, sep="|")
    test_df.to_csv(f'{output_dir}/{dataset_name.lower()}_test_full_logs.csv', index=True, sep="|")

    # Group by 'Optimizer Name', and 'Epoch number', and calculate mean and standard deviation
    train_grouped = train_df.groupby(['Optimizer Name', 'Epoch number']).agg(
        Mean_Training_Loss=('Average Training Loss', 'mean'),
        Std_Training_Loss=('Average Training Loss', 'std')
    ).reset_index()

    # Pivot the DataFrame to get Epoch number as columns
    pivot_train_df = train_grouped.pivot(index=['Optimizer Name'], columns='Epoch number')

    # Flatten the multi-level column index
    pivot_train_df.columns = [f'{stat}_epoch{int(epoch)}' for stat, epoch in pivot_train_df.columns]

    # Reset index to turn multi-index into columns
    pivot_train_df = pivot_train_df.reset_index()

    # Save the DataFrame to a CSV file
    pivot_train_df.to_csv(f'{output_dir}/{dataset_name.lower()}_train_grouped_logs.csv', index=False, sep="|")

    # Group by 'Optimizer Name' and calculate mean and standard deviation
    test_grouped = test_df.groupby(['Optimizer Name']).agg(
        Mean_Test_Set_Loss=('Average Test set Loss', 'mean'),
        Std_Test_Set_Loss=('Average Test set Loss', 'std'),
        Mean_Test_Set_Accuracy=('Average Test set accuracy', 'mean'),
        Std_Test_Set_Accuracy=('Average Test set accuracy', 'std')
    ).reset_index()

    # Save the DataFrame to a CSV file
    test_grouped.to_csv(f'{output_dir}/{dataset_name.lower()}_test_grouped_logs.csv', index=False, sep="|")

# Example usage for CIFAR-10 and MNIST
parse_training_logs('outputs/cifar10_training_run_logs.txt', 'Cifar10')
parse_training_logs('outputs/fashion_mnist_largebatch_training_run_logs.txt', 'FashionMNIST-largebatch')
parse_training_logs('outputs/fashion_mnist_training_run_logs.txt', 'FashionMNIST')
parse_training_logs('outputs/mnist_training_run_logs.txt', 'MNIST')
parse_training_logs('outputs/imdb_training_run_logs.txt', 'IMDB')
parse_training_logs('outputs/cola_training_run_logs.txt', 'CoLA')
parse_training_logs('outputs/cola_bertpt_training_run_logs.txt', 'CoLABert')
parse_training_logs('outputs/ag_news_training_run_logs.txt', 'AGNews')
parse_training_logs('outputs/flowers102_training_run_logs.txt', 'Flowers102')
parse_training_logs('outputs/stl10_cnn_training_run_logs.txt', 'STL10')
parse_training_logs('outputs/stl10_training_run_logs.txt', 'STL10-Resnet')
parse_training_logs('outputs/caltech101cnn_training_run_logs.txt', 'Caltech101')
parse_training_logs('outputs/eurosat_training_run_logs.txt', 'EuroSAT')
parse_training_logs('outputs/oxford_pet_collated.txt', 'OxfordPet')
parse_training_logs('outputs/yelp_training_run_logs.txt', 'yelp')
parse_training_logs('outputs/dbpedia_training_run_logs.txt', 'dbpedia')
parse_training_logs('outputs/cifar100_training_run_logs.txt', 'CIFAR-100')
parse_training_logs('outputs/caltech101_compiled_resnet.txt', 'Caltech101-resnet')
parse_training_logs('outputs/reuters_training_run_logs.txt', 'reuters')
parse_training_logs('outputs/reuters_largebatch_training_run_logs.txt', 'reuters-large-batch')
parse_training_logs('outputs/sogou_training_run_logs.txt', 'sogou-news')
parse_training_logs('outputs/arfull_training_run_logs.txt', 'amazon-review-full')
parse_training_logs('outputs/arpolarity_training_run_logs.txt', 'amazon-review-polarity')


# ----------------------------
# Write the mean loss to file.
# ----------------------------

# List of file paths
file_paths = [
    "outputs/cifar-100/cifar-100_train_grouped_logs.csv",
    "outputs/dbpedia/dbpedia_train_grouped_logs.csv",
    "outputs/caltech101-resnet/caltech101-resnet_train_grouped_logs.csv",
    "outputs/amazon-review-polarity/amazon-review-polarity_train_grouped_logs.csv",
    "outputs/cifar10/cifar10_train_grouped_logs.csv",
    "outputs/sogou-news/sogou-news_train_grouped_logs.csv",
    "outputs/yelp/yelp_train_grouped_logs.csv",
    "outputs/colabert/colabert_train_grouped_logs.csv",
    "outputs/oxfordpet/oxfordpet_train_grouped_logs.csv",
    "outputs/reuters-large-batch/reuters-large-batch_train_grouped_logs.csv",
    "outputs/caltech101/caltech101_train_grouped_logs.csv",
    "outputs/fashionmnist/fashionmnist_train_grouped_logs.csv",
    "outputs/amazon-review-full/amazon-review-full_train_grouped_logs.csv",
    "outputs/fashionmnist-largebatch/fashionmnist-largebatch_train_grouped_logs.csv",
    "outputs/flowers102/flowers102_train_grouped_logs.csv",
    "outputs/agnews/agnews_train_grouped_logs.csv",
    "outputs/mnist/mnist_train_grouped_logs.csv",
    "outputs/eurosat/eurosat_train_grouped_logs.csv",
    "outputs/stl10-resnet/stl10-resnet_train_grouped_logs.csv",
    "outputs/cola/cola_train_grouped_logs.csv",
    "outputs/stl10/stl10_train_grouped_logs.csv",
    "outputs/imdb/imdb_train_grouped_logs.csv",
    "outputs/reuters/reuters_train_grouped_logs.csv"
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
                print(len(mean_losses))

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