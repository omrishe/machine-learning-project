# Text Classification Analysis

## Overview

This project focuses on text classification using machine learning techniques its main purpose is to experiment and understand the various methods of cleaning data and trying different approaches.
It involves analyzing textual data to categorize it into predefined classes, which is essential in various applications like sentiment analysis, spam detection, and topic labeling.

## Contents

- `text-analysis.ipynb`: Jupyter Notebook containing the text analysis and classification process.
- `Xclean.csv`: Cleaned feature dataset.
- `Yclean.csv`: Corresponding labels for the feature dataset.
- `annotated_corpus_for_train.csv`: Annotated corpus used for training the model.
- `classification_results.csv`: Results from the classification process.
- `corpus_for_test.csv`: Corpus used for testing the model.

## Dependencies

To run the notebook and reproduce the analysis, ensure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `nltk` (Natural Language Toolkit)
- `matplotlib`
- `seaborn`
- `jupyter`

You can install these dependencies using `pip`:

pip install numpy pandas scikit-learn nltk matplotlib seaborn jupyter
## Usage
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/omrishe/machine-learning-project.git
cd machine-learning-project
Launch Jupyter Notebook:

jupyter notebook
Open and Run:

Open the text-analysis.ipynb notebook.
Execute the cells sequentially to perform the text classification analysis.
Methodology
The project follows these steps:

Data Preprocessing:

Loading and cleaning the textual data.
Tokenization and normalization.
Feature extraction using techniques like TF-IDF.
Model Training:

Splitting the data into training and testing sets.
Training machine learning models (e.g., Naive Bayes, SVM) on the training data.
Evaluation:

Assessing model performance using metrics like accuracy, precision, recall, and F1-score.
Analyzing the results and visualizing them for better understanding.
Results
The classification results are documented in the classification_results.csv file. Detailed analysis and visualizations are available in the Jupyter Notebook.

Author: Omri She
