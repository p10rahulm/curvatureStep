import pandas as pd
import re

# Log file path
log_file_path = 'outputs/hyperparam_search_logs.txt'

# Regular expressions to extract information
optimizer_re = re.compile(r'Running Hyperparameter grid search with Optimizer = (\w+) and momentum=([\d.]+).')
params_re = re.compile(r'params= ({.+})')
loop_re = re.compile(r'Running Loop: (\d+)/(\d+)')
epoch_re = re.compile(r'Epoch (\d+)/(\d+) completed, Average Loss: ([\d.]+)')
test_re = re.compile(r'Test set: Average loss: ([\d.]+), Accuracy: (\d+)/(\d+) \(([\d.]+)%\)')

# Lists to store data for DataFrames
train_data = []
test_data = []

# Variables to hold current state
current_optimizer = None
current_momentum = None
current_loop = None

# Read the log file
with open(log_file_path, 'r') as log_file:
    for line in log_file:
        optimizer_match = optimizer_re.search(line)
        if optimizer_match:
            current_optimizer = optimizer_match.group(1)
            current_momentum = optimizer_match.group(2)
            continue

        loop_match = loop_re.search(line)
        if loop_match:
            current_loop = loop_match.group(1)
            continue

        epoch_match = epoch_re.search(line)
        if epoch_match:
            epoch_number = epoch_match.group(1)
            avg_loss = epoch_match.group(3)
            train_data.append([current_optimizer, current_momentum, current_loop, epoch_number, avg_loss])
            continue

        test_match = test_re.search(line)
        if test_match:
            avg_test_loss = test_match.group(1)
            accuracy = test_match.group(4)
            test_data.append([current_optimizer, current_momentum, current_loop, avg_test_loss, accuracy])
            continue

# Create DataFrames
train_df = pd.DataFrame(train_data, columns=[
    'Optimizer Name', 'Momentum', 'Running loop number', 'Epoch number', 'Average Training Loss'])

test_df = pd.DataFrame(test_data, columns=[
    'Optimizer Name', 'Momentum', 'Running loop number', 'Average Test set Loss', 'Average Test set accuracy'])

# Display DataFrames
print("Training Data:")
print(train_df)
print("\nTest Data:")
print(test_df)

# Convert the relevant columns to numeric types
train_df['Average Training Loss'] = pd.to_numeric(train_df['Average Training Loss'], errors='coerce')
test_df['Average Test set Loss'] = pd.to_numeric(test_df['Average Test set Loss'], errors='coerce')
test_df['Average Test set accuracy'] = pd.to_numeric(test_df['Average Test set accuracy'], errors='coerce')

# Save full logs to CSV files
train_df.to_csv('outputs/hyperparam_search_mnist_train_full_logs.txt', index=True,sep="|")
test_df.to_csv('outputs/hyperparam_search_mnist_test_full_logs.txt', index=True,sep="|")

# Group by 'Optimizer Name', 'Momentum', and 'Epoch number', and calculate mean and standard deviation
train_grouped = train_df.groupby(['Optimizer Name', 'Momentum', 'Epoch number']).agg(
    Mean_Training_Loss=('Average Training Loss', 'mean'),
    Std_Training_Loss=('Average Training Loss', 'std')
).reset_index()

# Pivot the DataFrame to get Epoch number as columns
pivot_df = train_grouped.pivot(index=['Optimizer Name', 'Momentum'], columns='Epoch number')

# Flatten the multi-level column index
pivot_df.columns = [f'{stat}_epoch{int(epoch)}' for stat, epoch in pivot_df.columns]

# Reset index to turn multi-index into columns
pivot_df = pivot_df.reset_index()

# Set the display format for floats to scientific notation
pd.options.display.float_format = '{:.4e}'.format

# Format the DataFrame columns to scientific notation for saving
pivot_df = pivot_df.applymap(lambda x: f"{x:.4e}" if isinstance(x, (float, int)) else x)

# Display the resulting DataFrame
print(pivot_df)

# Save the DataFrame to a CSV file
pivot_df.to_csv('outputs/hyperparam_search_mnist_train_grouped_logs.txt', index=False, sep="|")


# Group by 'Optimizer Name' and 'Momentum' and calculate mean and standard deviation
test_grouped = test_df.groupby(['Optimizer Name', 'Momentum']).agg(
    Mean_Test_Set_Loss=('Average Test set Loss', 'mean'),
    Std_Test_Set_Loss=('Average Test set Loss', 'std'),
    Mean_Test_Set_Accuracy=('Average Test set accuracy', 'mean'),
    Std_Test_Set_Accuracy=('Average Test set accuracy', 'std')
).reset_index()

# Set the display format for floats to scientific notation for displaying
pd.options.display.float_format = '{:.4e}'.format

# Format the DataFrame columns to scientific notation for saving
test_grouped['Mean_Test_Set_Loss'] = test_grouped['Mean_Test_Set_Loss'].apply(lambda x: f"{x:.4e}")
test_grouped['Std_Test_Set_Loss'] = test_grouped['Std_Test_Set_Loss'].apply(lambda x: f"{x:.4e}")
test_grouped['Mean_Test_Set_Accuracy'] = test_grouped['Mean_Test_Set_Accuracy'].apply(lambda x: f"{x:.4e}")
test_grouped['Std_Test_Set_Accuracy'] = test_grouped['Std_Test_Set_Accuracy'].apply(lambda x: f"{x:.4e}")

# Display the resulting DataFrame
print(test_grouped)

# Save the DataFrame to a CSV file
test_grouped.to_csv('outputs/hyperparam_search_mnist_test_grouped_logs.txt', index=False, sep="|")

# Pivot the test DataFrame to get Curvature as columns
pivot_test_df = test_grouped.pivot_table(index=['Optimizer Name', 'Momentum'], columns='Curvature', values=['Mean_Test_Set_Loss', 'Std_Test_Set_Loss', 'Mean_Test_Set_Accuracy', 'Std_Test_Set_Accuracy'])
# # Pivot the test DataFrame to get Curvature as columns
# pivot_test_df = test_grouped.pivot_table(index=['Optimizer Name', 'Momentum'], columns='Curvature', values=['Mean_Test_Set_Loss', 'Std_Test_Set_Loss', 'Mean_Test_Set_Accuracy', 'Std_Test_Set_Accuracy'])
# Flatten the multi-level column index
pivot_test_df.columns = [f'{stat}_{curvature}' for stat, curvature in pivot_test_df.columns]
# Reset index to turn multi-index into columns
pivot_test_df = pivot_test_df.reset_index()
# Set the display format for floats to scientific notation for displaying
pd.options.display.float_format = '{:.4e}'.format
# Format the DataFrame columns to scientific notation for saving
pivot_test_df = pivot_test_df.applymap(lambda x: f"{x:.4e}" if isinstance(x, (float, int)) else x)
# Display the resulting DataFrame
print(pivot_test_df)

# Save the DataFrame to a CSV file
pivot_test_df.to_csv('outputs/hyperparam_search_mnist_test_optgrouped_logs.txt', index=False, sep="|")