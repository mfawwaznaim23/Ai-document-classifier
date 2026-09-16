# AI Document Classification and Search System

A machine learning and NLP project that classifies news articles into four categories and provides document search using TF-IDF and cosine similarity.

## Overview

This project uses the AG News dataset containing 120,000 labeled news articles. The system processes article titles and descriptions, converts the text into numerical features using TF-IDF, and trains a Logistic Regression classifier using scikit-learn.

## Categories

- World
- Sports
- Business
- Science/Technology

## Technologies Used

- Python
- Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook

## Machine Learning Workflow

1. Loaded and explored the dataset using Pandas.
2. Combined article titles and descriptions into a single text feature.
3. Split the dataset into training and testing sets.
4. Converted text into numerical features using TF-IDF vectorization.
5. Trained a Logistic Regression classification model using scikit-learn.
6. Evaluated the model using accuracy, precision, recall, and F1-score.
7. Built a function to classify new articles.
8. Implemented document search using TF-IDF and cosine similarity.
9. Visualized model performance using Matplotlib.
10. Performed error analysis on misclassified articles.

## Model Performance

The dataset was split into 96,000 training articles and 24,000 testing articles.

The Logistic Regression model achieved approximately **91.8% accuracy** on the unseen test data.

The F1-scores for the four categories were approximately:

- World: 0.92
- Sports: 0.97
- Business: 0.89
- Science/Technology: 0.90

## Document Search

The project also includes a document retrieval feature. A search query is converted into a TF-IDF vector and compared with the article vectors using cosine similarity.

The articles with the highest similarity scores are returned as the most relevant search results.

## Error Analysis

The model's incorrect predictions were examined to better understand its limitations. Some articles contained vocabulary related to multiple categories, which caused the classifier to confuse categories such as Business and Science/Technology or World and Sports.

## Results

The model achieved approximately **91.8% test accuracy** across four news categories.

The project demonstrates an end-to-end NLP workflow including data preprocessing, feature extraction, model training, prediction, evaluation, document retrieval, visualization, and error analysis.

## Dataset

This project uses the AG News Classification Dataset.

The dataset files are not included in this repository. To run the project, place the dataset files inside a local `data` folder.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Project

Open the Jupyter Notebook in VS Code or Jupyter Notebook and run the cells in order.