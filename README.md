# AI Document Classification and Search System

A machine learning and Natural Language Processing (NLP) project that classifies news articles into four categories and provides document search using TF-IDF and cosine similarity.

## Overview

This project uses the AG News dataset containing 120,000 labeled news articles. The system processes article titles and descriptions, converts text into numerical features using TF-IDF, and trains a Logistic Regression classifier using scikit-learn.

The project also uses 5-fold cross-validation and hyperparameter tuning with GridSearchCV to evaluate model consistency and optimize the Logistic Regression regularization parameter.

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

1. Loaded and explored 120,000 labeled news articles using Pandas.
2. Combined article titles and descriptions into a single text feature.
3. Split the dataset into 96,000 training articles and 24,000 testing articles.
4. Converted text into numerical features using TF-IDF vectorization.
5. Trained a Logistic Regression classification model using scikit-learn.
6. Evaluated the model using accuracy, precision, recall, and F1-score.
7. Built a function to classify new user-provided articles.
8. Implemented document retrieval using TF-IDF and cosine similarity.
9. Visualized model performance using Matplotlib.
10. Performed error analysis on misclassified articles.
11. Applied 5-fold cross-validation to evaluate model consistency across different subsets of the training data.
12. Used GridSearchCV to tune the Logistic Regression `C` hyperparameter.
13. Selected the best model and evaluated it on the untouched test set.
14. Compared the original and tuned model performance.

## Model Performance

The dataset was split into:

- Training samples: 96,000
- Testing samples: 24,000
- Original test accuracy: 91.8%
- Tuned test accuracy: 91.8%

The F1-scores on the test set were approximately:

- World: 0.92
- Sports: 0.97
- Business: 0.89
- Science/Technology: 0.90

## Cross-Validation

5-fold cross-validation was performed on the training data to evaluate how consistently the Logistic Regression model performed across different subsets of the data.

The model achieved:

- Average cross-validation accuracy: approximately 91.4%
- Cross-validation standard deviation: approximately 0.31 percentage points

The small variation between folds indicated that the model produced relatively consistent performance across the different training subsets.

## Hyperparameter Tuning

GridSearchCV was used to test different values of the Logistic Regression `C` hyperparameter:

- `C = 0.1`
- `C = 1`
- `C = 10`

The search used 5-fold cross-validation to compare the parameter values.

GridSearchCV selected:

- Best `C`: 1
- Best cross-validation accuracy: approximately 91.4%

Because `C = 1` was already the value used by the original Logistic Regression model, the tuned model achieved the same test accuracy of approximately 91.8%.

This showed that the original regularization setting was already the best-performing option among the tested values.

## Document Search

The project includes a document retrieval feature that allows a user to enter a search query.

The query is transformed using the trained TF-IDF vectorizer and compared with article vectors using cosine similarity. Articles with the highest similarity scores are returned as the most relevant results.

## Error Analysis

Incorrect predictions were examined to better understand the limitations of the classifier.

Some articles contained vocabulary associated with multiple categories, causing the model to confuse categories such as Business and Science/Technology or World and Sports.

This analysis helped identify situations where a TF-IDF-based model can struggle when different news categories contain overlapping vocabulary.

## Results

The final Logistic Regression classifier achieved approximately **91.8% accuracy on 24,000 unseen test articles**.

5-fold cross-validation produced an average accuracy of approximately **91.4%**, showing relatively consistent performance across different subsets of the training data.

GridSearchCV selected **C = 1** as the best regularization setting among the tested values. The tuned model maintained the original **91.8% test accuracy**.

The project demonstrates an end-to-end NLP and machine learning workflow including:

- Data preprocessing
- Train/test splitting
- TF-IDF feature extraction
- Logistic Regression classification
- Model evaluation
- Cross-validation
- Hyperparameter tuning
- Document retrieval
- Performance visualization
- Error analysis

## Dataset

This project uses the AG News Classification Dataset.

The dataset files are not included in this repository due to their size. To run the project, place the dataset files inside a local `data` folder.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Project

Open `document_classifier.ipynb` in VS Code or Jupyter Notebook and run the cells in order.
