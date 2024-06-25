# Adaptive Curvature Step Size (ACSS) for Optimizers

## Overview

We propose an adaptive step size method called Adaptive Curvature Step Size (ACSS), which dynamically adjusts the step size based on the local geometry of the optimization path. Our approach calculates the radius of curvature using consecutive gradients along the iterate path and sets the step size equal to this radius.

The effectiveness of ACSS stems from its ability to adapt to the local landscape of the optimization problem. In regions of low curvature, where consecutive gradient steps are nearly identical, the method allows for larger steps, recognizing that the convergence point is still distant. Conversely, in areas of high curvature, where gradient steps differ significantly in direction, ACSS reduces the step size, acknowledging proximity to a potential convergence point or the need for more careful navigation of a complex landscape.

A key advantage of ACSS is its ability to incorporate second-order gradient information, without the need to explicitly compute or store second-order terms. This results in improved optimization performance while maintaining a lower memory footprint compared to methods that directly use second-order information.

Through extensive empirical evaluation on 20 different datasets, we compare ACSS against 12 popular optimization methods, including Adam, SGD, AdaGrad, RMSProp, and their variants. Our results consistently show that ACSS provides performance benefits. We provide PyTorch implementations of ACSS versions for popular optimizers.

## Directory Structure

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
│   ├── simple_rnn_multiclass.py
│   ├── simple_rnn.py
│   ├── simple_dqn.py
│   ├── simple_rnn_speech.py
│   ├── simple_cnn_template.py
│   ├── simple_nn.py
├── optimizers
│   ├── adadelta_curvature.py
│   ├── adadelta.py
│   ├── adagrad_curvature.py
│   ├── adagrad.py
│   ├── adam_curvature.py
│   ├── adam.py
│   ├── adamw_curvature.py
│   ├── adamw.py
│   ├── amsgrad_curvature.py
│   ├── amsgrad.py
│   ├── heavyball_curvature.py
│   ├── heavyball.py
│   ├── nadam_curvature.py
│   ├── nadam.py
│   ├── nadamw_curvature.py
│   ├── nadamw.py
│   ├── nag_curvature.py
│   ├── nag.py
│   ├── rmsprop_curvature.py
│   ├── rmsprop.py
│   ├── rmsprop_with_momentum_curvature.py
│   ├── rmsprop_with_momentum.py
│   ├── shampoo_curvature.py
│   ├── shampoo.py
│   ├── simplesgd_curvature.py
│   ├── simplesgd.py
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
│   ├── yelp
├── README.md
├── requirements.txt
├── test.py
├── train.py
├── experiment_utils.py
├── optimizer_params.py
└── utilities.py
```

## Setup

To set up the environment, install the required dependencies using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Training and Testing

### Training

The `train.py` script handles the training process for various models and datasets. It includes different training functions for standard models, language models, and BERT-based models.

### Testing

The `test.py` script evaluates the performance of the trained models. It supports standard models, language models, and BERT-based models, providing detailed metrics such as average loss and accuracy.

### Experiment Utilities

The `experiment_utils.py` script facilitates running experiments with different optimizers, models, and datasets. It manages the training and testing loops and aggregates results for analysis.

## Models

We provide implementations for various models, including BERT classifiers, ResNet, and simple CNNs and RNNs. The models are defined in the `models` directory.

## Optimizers

Our implementation includes standard optimizers and their ACSS-enhanced versions. The optimizers are defined in the `optimizers` directory.

### Available Optimizers

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

## Training Log Parsing

The `training_parsers.py` script processes raw training logs, extracting relevant metrics and saving them in a structured format for further analysis. It supports multiple datasets and provides functionality to aggregate results across different optimizers.

### Example Usage

1. **Parsing Training Logs:**
   - Reads log files and extracts optimizer names, training losses, and test set performances.
   - Supports logs generated from various datasets and optimizers.

2. **Creating DataFrames:**
   - Stores extracted data in Pandas DataFrames for easy manipulation.
   - Separates training and test data into different DataFrames.

3. **Aggregating Results:**
   - Groups the data by optimizer and epoch to calculate mean and standard deviation of training losses.
   - Pivots the training DataFrame to get a more analysis-friendly format with epochs as columns.

4. **Saving Results:**
   - Saves the full logs and aggregated results into separate CSV files for each dataset.
   - Saves the mean training losses across different epochs for easy comparison between optimizers.

## Results

### Mean Improvement Over 5 Epochs for 12 Optimizers Across 20 Datasets

| Optimizer       | Epoch_1    | Epoch_2    | Epoch_3    | Epoch_4    | Epoch_5    |
|-----------------|------------|------------|------------|------------|------------|
| SimpleSGD       | 0.539742504 | 1.252146061 | 0.750951018 | 0.821385925 | 0.908407596 |
| HeavyBall       | 0.167656917 | 0.564890333 | 0.197533917 | 0.280896583 | 0.379726917 |
| NAG             | 0.155248917 | 0.459523917 | 0.16732425  | 0.237815    | 0.322022583 |
| Adagrad         | 0.024861417 | 0.046564833 | 0.049125417 | 0.049988917 | 0.056792917 |
| Adadelta        | 0.010991917 | 0.071387333 | 0.065003    | 0.05667075  | 0.054823333 |
| RMSProp         | -0.016485167 | 0.046684417 | 0.015839083 | 0.0243955   | 0.046374417 |
| AMSGrad         | 0.002314167 | 0.004105417 | 0.01043375 | 0.015027333 | 0.03454925 |
| NAdam           | 0.002429667 | -0.003835417 | 0.001610417 | 0.008449667 | 0.026809167 |
| NAdamW          | 0.020950583 | 0.069456333 | -0.013666561 | -0.00336811 | 0.019472961 |
| AdamW           | 0.0108145   | 0.023831917 | 0.010111583 | 0.012479333 | 0.019335167 |
| Adam            | -0.0067625  | 0.009065833 | -0.008171083 | 0.008893333 | 0.003704917 |
| RMSPropMomentum | 0.002767083 | 0.017398167 | -0.018411167 | -0.018351833 | -0.00997575 |

### MNIST Training Losses

| Optimizer Name          | Mean_Training_Loss_epoch1 | Mean_Training_Loss_epoch2 | Mean_Training_Loss_epoch3 | Mean_Training_Loss_epoch4 | Mean_Training_Loss_epoch5 | Mean_Training_Loss_epoch6 | Mean_Training_Loss_epoch7 | Mean_Training_Loss_epoch8 | Mean_Training_Loss_epoch9 | Mean_Training_Loss_epoch10 |
|-------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|----------------------------|
| SimpleSGDCurvature      | 0.39651                   | 0.17593                   | 0.13057                   | 0.10601                   | 0.09036                   | 0.07877                   | 0.06973                   | 0.06195                   | 0.05586                   | 0.05067                    |
| HeavyBallCurvature      | 0.33452                   | 0.15764                   | 0.12197                   | 0.10183                   | 0.08857                   | 0.0792                    | 0.07067                   | 0.06429                   | 0.05887                   | 0.05326                    |
| NAGCurvature            | 0.33464                   | 0.1574                    | 0.12209                   | 0.10172                   | 0.08846                   | 0.07916                   | 0.07075                   | 0.06407                   | 0.05865                   | 0.05351                    |
| AMSGrad                 | 0.38229                   | 0.19564                   | 0.14119                   | 0.11287                   | 0.09475                   | 0.08296                   | 0.07267                   | 0.06531                   | 0.05876                   | 0.05411                    |
| AMSGradCurvature        | 0.38224                   | 0.19604                   | 0.1416                    | 0.11305                   | 0.09526                   | 0.08334                   | 0.0735                    | 0.06566                   | 0.0594                    | 0.05448                    |
| Adam                    | 0.3821                    | 0.19528                   | 0.1412                    | 0.11358                   | 0.09664                   | 0.08501                   | 0.0753                    | 0.06788                   | 0.06244                   | 0.05771                    |
| AdamCurvature           | 0.38231                   | 0.19535                   | 0.14124                   | 0.11385                   | 0.09683                   | 0.08534                   | 0.0755                    | 0.06868                   | 0.06235                   | 0.05795                    |
| AdamW                   | 0.38243                   | 0.19519                   | 0.1416                    | 0.11413                   | 0.09725                   | 0.08565                   | 0.07603                   | 0.0693                    | 0.06338                   | 0.0587                     |
| AdamWCurvature          | 0.38209                   | 0.19512                   | 0.14148                   | 0.11425                   | 0.09695                   | 0.08542                   | 0.07564                   | 0.06909                   | 0.06313                   | 0.05913                    |
| RMSPropCurvature        | 0.39926                   | 0.20883                   | 0.15436                   | 0.12608                   | 0.10902                   | 0.09647                   | 0.08718                   | 0.07955                   | 0.07338                   | 0.06823                    |
| RMSProp                 | 0.39966                   | 0.21042                   | 0.15572                   | 0.12669                   | 0.10897                   | 0.09636                   | 0.08687                   | 0.07933                   | 0.07351                   | 0.06829                    |
| RMSPropMomentumCurvature| 0.39423                   | 0.20636                   | 0.15312                   | 0.1259                    | 0.10941                   | 0.09735                   | 0.08818                   | 0.08072                   | 0.07464                   | 0.06955                    |
| RMSPropMomentum         | 0.3936                    | 0.20554                   | 0.15268                   | 0.12597                   | 0.11013                   | 0.09829                   | 0.08926                   | 0.0817                    | 0.07547                   | 0.07026                    |
| Shampoo                 | 1.51103                   | 0.3873                    | 0.28421                   | 0.23752                   | 0.20485                   | 0.18046                   | 0.16148                   | 0.14628                   | 0.13371                   | 0.12331                    |
| NAdam                   | 0.34123                   | 0.16887                   | 0.1374                    | 0.12632                   | 0.12249                   | 0.12197                   | 0.12092                   | 0.12473                   | 0.13025                   | 0.13143                    |
| NAdamCurvature          | 0.34136                   | 0.16884                   | 0.13723                   | 0.12647                   | 0.12195                   | 0.12215                   | 0.12292                   | 0.12607                   | 0.12846                   | 0.13282                    |
| NAdamW                  | 0.34081                   | 0.16921                   | 0.1374                    | 0.12618                   | 0.12287                   | 0.12161                   | 0.12179                   | 0.12504                   | 0.12858                   | 0.13328                    |
| NAdamWCurvature         | 0.34113                   | 0.16885                   | 0.13745                   | 0.12675                   | 0.12318                   | 0.12175                   | 0.12175                   | 0.12509                   | 0.12932                   | 0.13443                    |
| HeavyBall               | 0.77726                   | 0.36868                   | 0.32254                   | 0.29565                   | 0.27446                   | 0.25613                   | 0.23913                   | 0.22388                   | 0.21011                   | 0.1978                     |
| NAG                     | 0.77726                   | 0.36867                   | 0.32255                   | 0.29565                   | 0.27446                   | 0.25614                   | 0.23914                   | 0.22388                   | 0.21012                   | 0.19781                    |
| AdagradCurvature        | 0.6987                    | 0.4394                    | 0.39262                   | 0.36851                   | 0.35273                   | 0.3412                    | 0.33223                   | 0.32486                   | 0.31857                   | 0.31301                    |
| Adagrad | 0.69824 | 0.43924 | 0.39256 | 0.36847 | 0.35274 | 0.34125 | 0.33224 | 0.32488 | 0.31861 | 0.31304 |
| Adadelta | 2.12782 | 1.78443 | 1.49008 | 1.26083 | 1.08826 | 0.95834 | 0.85957 | 0.78315 | 0.7232 | 0.67529 |
| AdadeltaCurvature | 2.19944 | 1.99861 | 1.812 | 1.63747 | 1.48097 | 1.34431 | 1.22604 | 1.12458 | 1.03787 | 0.96392 |

## Setup and Installation

To set up the environment and run the experiments, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Training and Testing

### Training

The `train.py` script handles the training process for various models and datasets. It includes different training functions for standard models, language models, and BERT-based models.

### Testing

The `test.py` script evaluates the performance of the trained models. It supports standard models, language models, and BERT-based models, providing detailed metrics such as average loss and accuracy.

### Experiment Utilities

The `experiment_utils.py` script facilitates running experiments with different optimizers, models, and datasets. It manages the training and testing loops and aggregates results for analysis.

## Models

We provide implementations for various models, including BERT classifiers, ResNet, and simple CNNs and RNNs. The models are defined in the `models` directory.

### Available Models

- `bert_model.py`
- `resnet.py`
- `simple_cnn.py`
- `simple_rnn_multiclass.py`
- `simple_rnn.py`
- `simple_dqn.py`
- `simple_rnn_speech.py`
- `simple_cnn_template.py`
- `simple_nn.py`

## Optimizers

Our implementation includes standard optimizers and their ACSS-enhanced versions. The optimizers are defined in the `optimizers` directory.

### Available Optimizers

- `adadelta_curvature.py`
- `adadelta.py`
- `adagrad_curvature.py`
- `adagrad.py`
- `adam_curvature.py`
- `adam.py`
- `adamw_curvature.py`
- `adamw.py`
- `amsgrad_curvature.py`
- `amsgrad.py`
- `heavyball_curvature.py`
- `heavyball.py`
- `nadam_curvature.py`
- `nadam.py`
- `nadamw_curvature.py`
- `nadamw.py`
- `nag_curvature.py`
- `nag.py`
- `rmsprop_curvature.py`
- `rmsprop.py`
- `rmsprop_with_momentum_curvature.py`
- `rmsprop_with_momentum.py`
- `shampoo_curvature.py`
- `shampoo.py`
- `simplesgd_curvature.py`
- `simplesgd.py`

## Training Log Parsing

The `training_parsers.py` script processes raw training logs, extracting relevant metrics and saving them in a structured format for further analysis. It supports multiple datasets and provides functionality to aggregate results across different optimizers.

### Features

1. **Parsing Training Logs:**
   - Reads log files and extracts optimizer names, training losses, and test set performances.
   - Supports logs generated from various datasets and optimizers.

2. **Creating DataFrames:**
   - Stores extracted data in Pandas DataFrames for easy manipulation.
   - Separates training and test data into different DataFrames.

3. **Aggregating Results:**
   - Groups the data by optimizer and epoch to calculate mean and standard deviation of training losses.
   - Pivots the training DataFrame to get a more analysis-friendly format with epochs as columns.

4. **Saving Results:**
   - Saves the full logs and aggregated results into separate CSV files for each dataset.
   - Saves the mean training losses across different epochs for easy comparison between optimizers.

---

