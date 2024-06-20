import pandas as pd
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
parse_training_logs('outputs/fashion_mnist_training_run_logs.txt', 'FashionMNIST')
parse_training_logs('outputs/mnist_training_run_logs.txt', 'MNIST')
parse_training_logs('outputs/imdb_training_run_logs.txt', 'IMDB')
parse_training_logs('outputs/cola_training_run_logs.txt', 'CoLA')
parse_training_logs('outputs/cola_bertpt_training_run_logs.txt', 'CoLABert')
parse_training_logs('outputs/ag_news_training_run_logs.txt', 'AGNews')
parse_training_logs('outputs/flowers102_training_run_logs.txt', 'Flowers102')
parse_training_logs('outputs/stl10_cnn_training_run_logs.txt', 'STL10')
parse_training_logs('outputs/caltech101cnn_training_run_logs.txt', 'Caltech101')
parse_training_logs('outputs/eurosat_training_run_logs.txt', 'EuroSAT')
parse_training_logs('outputs/oxford_pet_collated.txt', 'OxfordPet')
parse_training_logs('outputs/yelp_training_run_logs.txt', 'yelp')
parse_training_logs('outputs/dbpedia_training_run_logs.txt', 'dbpedia')