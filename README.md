# Linear Regression Models: Batch, Stochastic, and Mini-Batch Gradient Descent

This repository contains an implementation of linear regression using three different optimization techniques: Batch Gradient Descent, Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent. The project demonstrates data preprocessing, exploratory data analysis (EDA), and model training from scratch using Python and NumPy, with visualizations using Matplotlib and Seaborn.

## Table of Contents
- [Linear Regression Models: Batch, Stochastic, and Mini-Batch Gradient Descent](#linear-regression-models-batch-stochastic-and-mini-batch-gradient-descent)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Dataset](#dataset)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Results \& Analysis](#results--analysis)
  - [References](#references)

## Overview
This project explores the performance and convergence of different gradient descent algorithms for linear regression. It includes:
- Data cleaning and preprocessing (including one-hot encoding for categorical variables)
- Exploratory Data Analysis (EDA) with visualizations
- Implementation of Batch, Stochastic, and Mini-Batch Gradient Descent from scratch
- Comparison of convergence rates and feature importances

## Features
- **Data Preprocessing:** Handles missing values and encodes categorical features.
- **EDA:** Plots histograms, boxplots, scatter plots, and correlation heatmaps.
- **Custom Linear Regression:** Implements three gradient descent variants without using scikit-learn.
- **Visualization:** Plots loss curves and feature importances.

## Requirements
Install the following Python packages:
```bash
pip install pandas numpy matplotlib seaborn
```

## Dataset
- The code expects a file named `dataset.csv` in the project root directory.
- The dataset should include both numerical and categorical columns, with a target variable named `Economic Loss`.
- Categorical columns are automatically detected and one-hot encoded.
- **Note:** The dataset is not included in this repository due to size constraints.

## Usage
1. **Place your dataset:** Ensure `dataset.csv` is in the project root.
2. **Open the notebook:**
   ```bash
   jupyter notebook notebook.ipynb
   ```
   or use JupyterLab or VSCode's notebook interface.
3. **Outputs:**
   - Cleaned dataset saved as `cleaned_dataset.csv`.
   - Visualizations and loss curves will be displayed.
   - Feature importance plots for each gradient descent method.

## Project Structure
```
├── notebook.ipynb       # Main script with all code and analysis
├── dataset.csv           # Input dataset (not included)
├── cleaned_dataset.csv   # Output after preprocessing
├── README.md             # Project documentation
├── LICENSE               # License file
```

## Results & Analysis
- The script compares the convergence and performance of Batch GD, SGD, and Mini-Batch SGD.
- Visualizations show the loss history and feature importances for each method.
- See the end of `i210442_a1.py` for detailed discussion and plots.

## References
- [Gradient Descent from Scratch: Batch, Stochastic, and Mini-Batch](https://medium.com/@jaleeladejumo/gradient-descent-from-scratch-batch-gradient-descent-stochastic-gradient-descent-and-mini-batch-def681187473)
- [Stochastic Gradient Descent in Python: A Complete Guide](https://medium.com/@dhirendrachoudhary_96193/stochastic-gradient-descent-in-python-a-complete-guide-for-ml-optimization-c140de6119dc)
- [pyimagesearch: Stochastic Gradient Descent (SGD) with Python](https://pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/)

---
**Author:** Hassaan Farooq Malik
