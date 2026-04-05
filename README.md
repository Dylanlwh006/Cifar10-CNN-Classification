# CIFAR-10 CNN Classification

CNN-based CIFAR-10 image classification with systematic model improvement and performance analysis.

> Built to explore CNN architectures and training strategies for image classification.

## Overview
This project implements and evaluates convolutional neural networks for CIFAR-10 image classification using PyTorch.

A baseline CNN is first established, followed by a systematically improved model incorporating data augmentation, batch normalization, dropout, and learning rate scheduling.

The goal is to analyze how architectural and training choices impact model performance.

## Results
- Baseline test accuracy: **76.33%**
- Final test accuracy: **87.17%**
- Improvement: **+10.84 percentage points**

## Techniques Used
- Data augmentation
- Batch normalization
- Dropout
- Adam optimizer
- Learning rate scheduling

## Project Files
- `cifar10_cnn_classification.py` — main training and evaluation script
- `report.pdf` — detailed project report
- `images/` — plots and confusion matrix

## Sample Results

### Final Accuracy Curve
![Final Accuracy](images/Final%20Accuracy.png)

### Final Loss Curve
![Final Loss](images/Final%20Loss.png)

### Baseline Accuracy Curve
![Baseline Accuracy](images/Baseline%20Accuracy.png)

### Baseline Loss Curve
![Baseline Loss](images/Baseline%20Loss.png)

### Confusion Matrix
![Confusion Matrix](images/Confusion%20Matrix.png)

- Dataset: CIFAR-10 (50,000 training / 10,000 test images)

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
