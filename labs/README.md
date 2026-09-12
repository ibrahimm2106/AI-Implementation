# AI / Machine Learning Practical Labs

[![Lab 02](https://img.shields.io/badge/Lab_02-Preprocessing-0A66C2)](original/lab02/)
[![Lab 07](https://img.shields.io/badge/Lab_07-Clustering-6F42C1)](original/lab07/)
[![Lab 08](https://img.shields.io/badge/Lab_08-Decision_Tree-2DA44E)](original/lab08/)
[![Lab 10](https://img.shields.io/badge/Lab_10-Neural_Network-FF6F00)](original/lab10/)

This section keeps GitHub-friendly notebook/source editions based on the uploaded practical work and provides cleaned standalone Python versions for portfolio review.

## Lab 02 — Data preprocessing

**Concepts:** previewing data, checking missingness, median/constant imputation, numerical scaling, categorical encoding and CSV export.

The uploaded notebook records the missing-value counts, removes missingness through imputation, scales the numerical features and exports a processed dataset. The public repository keeps the preprocessing logic and documented final output while avoiding a duplicate large processed dataset file.

Files:

- [`original/lab02/Lab_2_Preprocessing.ipynb`](original/lab02/Lab_2_Preprocessing.ipynb)
- [`original/lab02/Lab_2_Preprocessing.py`](original/lab02/Lab_2_Preprocessing.py)
- [`lab02_preprocessing.py`](lab02_preprocessing.py)

## Lab 07 — Clustering and hierarchical clustering

**Concepts:** Euclidean distance, K-means concepts, Ward linkage and dendrogram visualisation.

### Code

![Lab 07 code](../docs/images/code/lab07_clustering_code.svg)

### Ward-linkage dendrogram

![Lab 07 dendrogram](../docs/images/portfolio/lab07_dendrogram.svg)

Files:

- [`original/lab07/Lab_7_Clustering.ipynb`](original/lab07/Lab_7_Clustering.ipynb)
- [`original/lab07/Lab_7_Clustering.py`](original/lab07/Lab_7_Clustering.py)
- [`lab07_clustering.py`](lab07_clustering.py)

## Lab 08 — Decision Tree classification

**Concepts:** categorical encoding, `DecisionTreeClassifier`, sample predictions and interpretable tree rules.

The recorded uploaded-notebook output produced:

```text
Sample 1 Prediction: Hired
Sample 2 Prediction: Not Hired
```

It also printed readable decision-tree logic.

Files:

- [`original/lab08/Lab_8_Decision_Tree.ipynb`](original/lab08/Lab_8_Decision_Tree.ipynb)
- [`original/lab08/Lab_8_Decision_Tree.py`](original/lab08/Lab_8_Decision_Tree.py)
- [`lab08_decision_tree.py`](lab08_decision_tree.py)

> The `PastHires.csv` source dataset was not among the files supplied for this repository build, so the runnable script expects the dataset to be provided at runtime.

## Lab 10 — MNIST neural network

**Concepts:** TensorFlow/Keras, tensors, dense neural networks, ReLU, softmax, training/validation history, prediction, confusion matrices and hidden-layer representations.

The uploaded run trained for 10 epochs and recorded:

```text
Test Accuracy: 0.9695
Test Loss: 0.1034
Predicted Digit: 7
True Label: 7
```

### Code

![Lab 10 code](../docs/images/code/lab10_neural_network_code.svg)

### Training history

![Lab 10 training](../docs/images/portfolio/lab10_training_accuracy.svg)

### Example prediction

![MNIST 7](../docs/images/portfolio/lab10_prediction.svg)

Files:

- [`original/lab10/Lab_10_MNIST_Neural_Network.ipynb`](original/lab10/Lab_10_MNIST_Neural_Network.ipynb)
- [`original/lab10/Lab_10_MNIST_Neural_Network.py`](original/lab10/Lab_10_MNIST_Neural_Network.py)
- [`lab10_mnist_neural_network.py`](lab10_mnist_neural_network.py)

## Why both academic evidence and refactored scripts are included

The notebook/source editions show the learning process and recorded outputs. The standalone scripts remove machine-specific paths, make the examples easier to rerun and demonstrate the move from exploratory work toward maintainable software.
