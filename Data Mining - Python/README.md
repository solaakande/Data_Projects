# Data Mining & Informatics Workshop Solutions

## Overview
This project contains solutions to a series of workshops completed as part of the Data Mining & Informatics module.

The work progresses from basic Python programming to advanced machine learning techniques, including data preprocessing, classification, clustering, association rule mining, and sentiment analysis.

## Project Structure

### Workshop 1: Python Fundamentals
- Implemented basic Python tasks such as:
  - Capitalising words in a string  
  - Calculating the area of a circle  
  - Manipulating datasets using pandas  
- Created functions to:
  - Swap dataset columns  
  - Compare numerical values and generate new features  

### Workshop 2: Data Preprocessing
- Worked with the Adult dataset  
- Performed full preprocessing pipeline:
  - Dataset exploration (`info`, `head`)  
  - Handling missing values  
  - Encoding categorical data  
  - Normalising numerical features  

**Key Decisions**
- Used mean for numerical missing values  
- Used mode for categorical missing values  
- Applied MinMax scaling for standardisation  

### Workshop 3: Classification Models
- Built and compared four machine learning models:
  - Support Vector Machine (SVM)  
  - Decision Tree (DT)  
  - Random Forest (RF)  
  - K-Nearest Neighbours (KNN)  

**Steps**
- Cleaned and encoded dataset  
- Performed train/test split  
- Evaluated models using confusion matrices and metrics  

**Key Result**
- Random Forest achieved the highest accuracy  
- SVM also performed well after optimisation  

### Workshop 4: Clustering & PCA
- Applied clustering techniques:
  - K-Means  
  - Hierarchical Clustering  

**Findings**
- Optimal number of clusters: 3  
- Used 3D visualisations to interpret clusters  

- Applied Principal Component Analysis (PCA):
  - Reduced dimensionality  
  - Visualised income groups  

### Workshop 5: Association Rule Mining
- Used Apriori algorithm on retail datasets  
- Generated association rules for multiple countries  

**Insights**
- Identified strong purchase patterns  
- Found items frequently bought together  
- Evaluated rules using:
  - Support  
  - Confidence  
  - Lift  

### Workshop 6: Advanced Topics

#### Decision Trees
- Calculated entropy and information gain manually  
- Determined best split for building a decision tree  
- Constructed a simple decision rule  

#### K-Means Clustering (Manual)
- Performed clustering step-by-step:
  - Calculated distances  
  - Updated centroids  
- Converged after 2 iterations  

#### Sentiment Analysis
- Built text classification models on tweet data  
- Applied multiple models:
  - SVM  
  - Logistic Regression  
  - Random Forest  
  - Naive Bayes  
  - KNN  
  - Decision Tree  

**Results**
- SVM and Logistic Regression performed best  
- Naive Bayes had lower sensitivity  

## What I Did
- Implemented machine learning models from scratch using Python  
- Applied full data preprocessing pipelines  
- Compared multiple algorithms across tasks  
- Performed both supervised and unsupervised learning  
- Analysed model performance using evaluation metrics  

## Tools and Technologies
- Python  
- pandas, numpy  
- scikit-learn  
- matplotlib  

## Key Skills Demonstrated
- Data preprocessing and cleaning  
- Feature engineering  
- Classification modelling  
- Clustering techniques  
- Association rule mining  
- Dimensionality reduction (PCA)  
- Model evaluation and comparison  

## Limitations
- Some models had moderate accuracy depending on dataset  
- Performance depends on dataset quality  
- Limited hyperparameter tuning in some tasks  

## How to Run
Clone the repository:

```bash
git clone https://github.com/solaakande/Data_Projects.git
