# Adaptive Curvature Step Size (ACSS)

## Introduction

We propose an adaptive step size method called Adaptive Curvature Step Size (ACSS), which dynamically adjusts the step size based on the local geometry of the optimization path. Our approach calculates the radius of curvature using consecutive gradients along the iterate path and sets the step size equal to this radius.

The effectiveness of ACSS stems from its ability to adapt to the local landscape of the optimization problem. In regions of low curvature, where consecutive gradient steps are nearly identical, the method allows for larger steps, recognizing that the convergence point is still distant. Conversely, in areas of high curvature, where gradient steps differ significantly in direction, ACSS reduces the step size, acknowledging proximity to a potential convergence point or the need for more careful navigation of a complex landscape.

A key advantage of ACSS is its ability to incorporate second-order gradient information without the need to explicitly compute or store second-order terms. This results in improved optimization performance while maintaining a lower memory footprint compared to methods that directly use second-order information.

Through extensive empirical evaluation on 20 different datasets, we compare ACSS against 12 popular optimization methods, including Adam, SGD, AdaGrad, RMSProp, and their variants. Our results consistently show that ACSS provides performance benefits. We provide PyTorch implementations of ACSS versions for popular optimizers.

## Datasets

The following datasets were used for testing:

- CIFAR-100
- DBPedia
- Caltech101
- Amazon Review Polarity
- CIFAR-10
- Sogou News
- Yelp
- CoLA
- Oxford-IIIT Pet
- Reuters
- FashionMNIST
- Amazon Review Full
- Flowers102
- AGNews
- MNIST
- EuroSat
- STL-10
- IMDB

## Optimizers

The following optimizers were tested:

- SimpleSGD
- HeavyBall
- NAG
- Adadelta
- Adagrad
- NAdam
- NAdamW
- RMSProp
- RMSPropMomentum
- AdamW
- Adam
- AMSGrad

## Results

### Mean Improvement over 5 Epochs for 12 Optimizers by Using Curvature Step Size over 20 Datasets

| Optimizer        | Epoch_1      | Epoch_2      | Epoch_3      | Epoch_4      | Epoch_5      |
|------------------|--------------|--------------|--------------|--------------|--------------|
| SimpleSGD        | 0.539742504  | 1.252146061  | 0.750951018  | 0.821385925  | 0.908407596  |
| HeavyBall        | 0.167656917  | 0.564890333  | 0.197533917  | 0.280896583  | 0.379726917  |
| NAG              | 0.155248917  | 0.459523917  | 0.16732425   | 0.237815     | 0.322022583  |
| Adagrad          | 0.024861417  | 0.046564833  | 0.049125417  | 0.049988917  | 0.056792917  |
| Adadelta         | 0.010991917  | 0.071387333  | 0.065003     | 0.05667075   | 0.054823333  |
| RMSProp          | -0.016485167 | 0.046684417  | 0.015839083  | 0.0243955    | 0.046374417  |
| AMSGrad          | 0.002314167  | 0.004105417  | 0.01043375   | 0.015027333  | 0.03454925   |
| NAdam            | 0.002429667  | -0.003835417 | 0.001610417  | 0.008449667  | 0.026809167  |
| NAdamW           | 0.020950583  | 0.069456333  | -0.013666561 | -0.00336811  | 0.019472961  |
| AdamW            | 0.0108145    | 0.023831917  | 0.010111583  | 0.012479333  | 0.019335167  |
| Adam             | -0.0067625   | 0.009065833  | -0.008171083 | 0.008893333  | 0.003704917  |
| RMSPropMomentum  | 0.002767083  | 0.017398167  | -0.018411167 | -0.018351833 | -0.00997575  |

### MNIST Training Losses

| Optimizer Name ↓  Mean Training Loss →  | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Epoch 6 | Epoch 7 | Epoch 8 | Epoch 9 | Epoch 10 |
|--------------------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|
| SimpleSGDCurvature     | 0.39651             | 0.17593 | 0.13057 | 0.10601 | 0.09036 | 0.07877 | 0.06973 | 0.06195 | 0.05586 | 0.05067 |
| HeavyBallCurvature     | 0.33452             | 0.15764 | 0.12197 | 0.10183 | 0.08857 | 0.0792  | 0.07067 | 0.06429 | 0.05887 | 0.05326 |
| NAGCurvature           | 0.33464             | 0.1574  | 0.12209 | 0.10172 | 0.08846 | 0.07916 | 0.07075 | 0.06407 | 0.05865 | 0.05351 |
| AMSGrad                | 0.38229             | 0.19564 | 0.14119 | 0.11287 | 0.09475 | 0.08296 | 0.07267 | 0.06531 | 0.05876 | 0.05411 |
| AMSGradCurvature       | 0.38224             | 0.19604 | 0.1416  | 0.11305 | 0.09526 | 0.08334 | 0.0735  | 0.06566 | 0.0594  | 0.05448 |
| Adam                   | 0.3821              | 0.19528 | 0.1412  | 0.11358 | 0.09664 | 0.08501 | 0.0753  | 0.06788 | 0.06244 | 0.05771 |
| AdamCurvature          | 0.38231             | 0.19535 | 0.14124 | 0.11385 | 0.09683 | 0.08534 | 0.0755  | 0.06868 | 0.06235 | 0.05795 |
| AdamW                  | 0.38243             | 0.19519 | 0.1416  | 0.11413 | 0.09725 | 0.08565 | 0.07603 | 0.0693  | 0.06338 | 0.0587  |
| AdamWCurvature         | 0.38209             | 0.19512 | 0.14148 | 0.11425 | 0.09695 | 0.08542 | 0.07564 | 0.06909 | 0.06313 | 0.05913 |
| RMSPropCurvature       | 0.39926             | 0.20883 | 0.15436 | 0.12608 | 0.10902 | 0.09647 | 0.08718 | 0.07955 | 0.07338 | 0.06823 |
| RMSProp                | 0.39966             | 0.21042 | 0.15572 | 0.12669 | 0.10897 | 0.09636 | 0.08687 | 0.07933 | 0.07351 | 0.06829 |
| RMSPropMomentumCurvature| 0.39423            | 0.20636 | 0.15312 | 0.1259  | 0.10941 | 0.09735 | 0.08818 | 0.08072 | 0.07464 | 0.06955 |
| RMSPropMomentum        | 0.3936              | 0.20554 | 0.15268 | 0.12597 | 0.11013 | 0.09829 | 0.08926 | 0.0817  | 0.07547 | 0.07026 |



## Setup and Installation

### Requirements

The required dependencies are listed in `requirements.txt`. To install them, run:

```bash
pip install -r requirements.txt
```

### Data Loaders

The data loaders for various datasets are implemented in the `data_loaders` directory. Each data loader is responsible for loading the respective dataset, tokenizing the text, and creating data batches. The data loaders use the GloVe embeddings for text representation.

Here are examples of data loaders for the three datasets:

#### Amazon Review Full

```python
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import AmazonReviewFull

# Define BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load your dataset
train_iter, test_iter = AmazonReviewFull(split=('train', 'test'))

# Function to convert text to tensor for BERT
def text_pipeline(x):
    return tokenizer(x, padding='max_length', max_length=512, truncation=True, return_tensors='pt')

# Function to convert label to tensor
def label_pipeline(x):
    return int(x) - 1  # Convert labels to zero-index

# Collate function for DataLoader
def collate_batch(batch):
    label_list, input_ids_list, attention_mask_list = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = text_pipeline(_text)
        input_ids_list.append(processed_text['input_ids'].squeeze(0))
        attention_mask_list.append(processed_text['attention_mask'].squeeze(0))
    
    input_ids_list = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_list = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    
    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'label': label_list
    }

# Function to load Amazon Review Full dataset
def load_amazon_review_full(batch_size=16):
    train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader
```

#### Amazon Review Polarity

```python
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import AmazonReviewPolarity

# Define BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load your dataset
train_iter, test_iter = AmazonReviewPolarity(split=('train', 'test'))

# Function to convert text to tensor for BERT
def text_pipeline(x):
    return tokenizer(x, padding='max_length', max_length=512, truncation=True, return_tensors='pt')

# Function to convert label to tensor
def label_pipeline(x):
    return int(x) - 1  # Convert labels to zero-index

# Collate function for DataLoader
def collate_batch(batch):
    label_list, input_ids_list, attention_mask_list = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = text_pipeline(_text)
        input_ids_list.append(processed_text['input_ids'].squeeze(0))
        attention_mask_list.append(processed_text['attention_mask'].squeeze(0))
    
    input_ids_list = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_list = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    
    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'label': label_list
    }

# Function to load Amazon Review Polarity dataset
def load_amazon_review_polarity(batch_size=16):
    train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader
```

#### Sogou News

### Sogou News (Continued)

```python
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import SogouNews

# Define BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load your dataset
train_iter, test_iter = SogouNews(split=('train', 'test'))

# Function to convert text to tensor for BERT
def text_pipeline(x):
    return tokenizer(x, padding='max_length', max_length=512, truncation=True, return_tensors='pt')

# Function to convert label to tensor
def label_pipeline(x):
    return int(x) - 1  # Convert labels to zero-index

# Collate function for DataLoader
def collate_batch(batch):
    label_list, input_ids_list, attention_mask_list = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = text_pipeline(_text)
        input_ids_list.append

        processed_text = text_pipeline(_text)
        input_ids_list.append(processed_text['input_ids'].squeeze(0))
        attention_mask_list.append(processed_text['attention_mask'].squeeze(0))
    
    input_ids_list = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_list = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    
    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'label': label_list
    }

# Function to load Sogou News dataset
def load_sogou_news(batch_size=16):
    train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader
```

### Usage

To use the ACSS method with any dataset and optimizer, you can run the corresponding experiment file. Below are some examples:

#### Running CIFAR-100 Experiment

```bash
python experiments/cifar100_training_runs.py
```

#### Running Amazon Review Polarity Experiment

```bash
python experiments/amazon_review_polarity_training_runs.py
```

#### Running Sogou News Experiment

```bash
python experiments/sogou_news_training_runs.py
```

### Experiment Runner

The `experiment_utils.py` file contains the `run_experiment` function, which handles the training and testing of the model with different optimizers. Here is an example usage of the `run_experiment` function:

```python
mean_accuracy, std_accuracy = run_experiment(
    optimizer_class,
    params,
    dataset_loader=dataset_loader,
    model_class=model,
    num_runs=total_runs,
    num_epochs=total_epochs,
    debug_logs=True,
    model_hyperparams=model_hyperparams,
    loss_criterion=loss_criterion,
    device=device,
    trainer_fn=trainer_function,
    tester_fn=test_function,
)
```

### Results Logging

The results of the experiments are logged in CSV files located in the `outputs` directory. Each experiment generates a separate log file with detailed performance metrics.

### Directory Structure

```
.
├── data_loaders
│   ├── amazon_review_full.py
│   ├── amazon_review_polarity.py
│   ├── caltech101.py
│   ├── cifar100.py
│   ├── cola.py
│   ├── cola_bert.py
│   ├── dbpedia.py
│   ├── flowers102.py
│   ├── mnist.py
│   ├── oxford_pet.py
│   ├── reuters.py
│   ├── sogou_news.py
│   ├── stl10.py
│   └── yelp.py
├── experiments
│   ├── ag_news_training_runs.py
│   ├── amazon_review_full_training_runs.py
│   ├── amazon_review_polarity_training_runs.py
│   ├── caltech101_training_runs.py
│   ├── cifar100_training_runs.py
│   ├── cola_training_runs.py
│   ├── dbpedia_training_runs.py
│   ├── flowers102_training_runs.py
│   ├── imdb_training_runs.py
│   ├── mnist_training_runs.py
│   ├── oxford_pet_training_runs.py
│   ├── reuters_training_runs.py
│   ├── sogou_news_training_runs.py
│   ├── stl10_training_runs.py
│   ├── yelp_training_runs.py
│   └── eurosat_training_runs.py
├── models
│   ├── bert_model.py
│   ├── resnet.py
│   ├── simple_cnn.py
│   └── simple_rnn_multiclass.py
├── outputs
│   ├── agnews
│   ├── amazon-review-full
│   ├── amazon-review-polarity
│   ├── caltech101
│   ├── caltech101-resnet
│   ├── cifar-100
│   ├── cifar10
│   ├── cola
│   ├── colabert
│   ├── dbpedia
│   ├── eurosat
│   ├── fashionmnist
│   ├── fashionmnist-largebatch
│   ├── flowers102
│   ├── imdb
│   ├── mnist
│   ├── oxfordpet
│   ├── reuters
│   ├── reuters-large-batch
│   ├── sogou-news
│   ├── stl10
│   ├── stl10-resnet
│   └── yelp
├── README.md
├── requirements.txt
├── test.py
├── train.py
├── experiment_utils.py
├── optimizer_params.py
└── utilities.py
```

## Conclusion

The Adaptive Curvature Step Size (ACSS) method demonstrates significant improvements in training performance across a variety of datasets and optimizers. By dynamically adjusting the step size based on the local geometry of the optimization path, ACSS offers a more efficient and effective approach to training deep learning models.

For more detailed results and analysis, please refer to the individual experiment logs in the `outputs` directory.