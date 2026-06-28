# CIFAR-10 Machine Learning Project

This repository contains a compact end-to-end study of the CIFAR-10 image classification dataset using multiple modeling approaches. The project compares a simple convolutional neural network, a residual CNN, and several classical machine learning baselines, and it includes evaluation utilities for reporting accuracy, confusion matrices, and per-class performance.

## Project Overview

The CIFAR-10 dataset consists of 60,000 color images of size $32 \times 32$ pixels divided into 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The notebooks in this repository train and evaluate models for this task and save the best-performing checkpoints as HDF5 files.

## What Is Included

- A simple CNN implementation in [cnn_simple.ipynb](cnn_simple.ipynb)
- A residual CNN implementation in [cnn_resnet.ipynb](cnn_resnet.ipynb)
- A classical machine learning benchmark in [classical_ml.ipynb](classical_ml.ipynb)
- Evaluation helpers in [model_check.py](model_check.py)
- Pretrained model checkpoints:
  - [best_cifar10_cnn.h5](best_cifar10_cnn.h5)
  - [best_cifar10_resnet.h5](best_cifar10_resnet.h5)

## Repository Structure

```text
.
├── README.md
├── model_check.py
├── cnn_simple.ipynb
├── cnn_resnet.ipynb
├── classical_ml.ipynb
├── best_cifar10_cnn.h5
├── best_cifar10_resnet.h5
└── LICENSE
```

## Dataset

The notebooks use the CIFAR-10 dataset provided by Keras through `tensorflow.keras.datasets.cifar10`.

Typical preprocessing steps include:

- loading train and test splits
- normalizing pixel values to the range $[0, 1]$
- converting labels to one-hot vectors for neural network training
- resizing or reshaping inputs when required by specific models

## Models and Experiments

### 1. Simple CNN

The simple CNN notebook trains a compact convolutional neural network with stacked convolutional layers, pooling layers, and dense layers. This serves as a baseline deep learning model for CIFAR-10 classification.

### 2. Residual CNN

The residual CNN notebook uses a deeper architecture with residual blocks to improve feature learning and gradient flow. This model is more expressive than the simple CNN and generally achieves stronger performance.

### 3. Classical Machine Learning

The classical ML notebook evaluates several non-deep-learning classifiers such as:

- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine
- Multi-Layer Perceptron
- XGBoost

These experiments provide a traditional machine learning benchmark for comparison with the neural network approaches.

## Evaluation Utilities

The script [model_check.py](model_check.py) provides helper functions for:

- generating full classification reports
- printing confusion matrices
- computing per-class accuracy
- visualizing predictions and misclassifications

These utilities are especially useful after loading a trained model to inspect its behavior on the test set.

## Installation

This project is designed for Python 3.9+.

Install the required packages with:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter ipykernel
```

If you are using a GPU-enabled environment, TensorFlow will leverage it automatically when available.

## Running the Notebooks

Start Jupyter Notebook or JupyterLab from the repository root:

```bash
jupyter notebook
```

Then open one of the notebooks:

- [cnn_simple.ipynb](cnn_simple.ipynb)
- [cnn_resnet.ipynb](cnn_resnet.ipynb)
- [classical_ml.ipynb](classical_ml.ipynb)

## Reproducing the Training Workflow

1. Open the desired notebook.
2. Run the cells in order.
3. The notebook will download the CIFAR-10 dataset, train the model, and save the best checkpoint.
4. The final cells evaluate the saved model on the test split and print performance metrics.

## Loading a Saved Model

You can load the saved checkpoints with TensorFlow:

```python
import tensorflow as tf

model = tf.keras.models.load_model("best_cifar10_cnn.h5")
print(model.summary())
```

## Example Results

The notebooks report strong performance for the deep learning models, while the classical methods provide a useful baseline. Representative results observed in the training notebooks include:

- Simple CNN: approximately 89.67% test accuracy
- Residual CNN: approximately 91.77% test accuracy
- Classical ML baselines: accuracy varied by model, with SVM and MLP performing better than simpler methods

These values may vary slightly depending on hardware, random initialization, and training conditions.

## Notes

- The repository is intentionally lightweight and focused on experimentation and education.
- The project is well-suited for learning image classification, CNN architectures, transfer of concepts from classical machine learning to deep learning, and model evaluation.
- The saved model files can be used for quick inference or comparison without retraining.

## License

This project is distributed under the terms of the license included in [LICENSE](LICENSE).
