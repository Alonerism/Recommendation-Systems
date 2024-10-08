# Recommendation Systems Repository

This repository contains two different approaches for building recommendation systems: one designed for handling large-scale datasets using PySpark and another built for normal-sized datasets using SVD and KNN algorithms in Python. Each folder provides implementations tailored to the respective dataset size and system requirements.

## Repository Structure

### 1. **pySpark**
The `pySpark` folder contains recommendation models built using Apache Spark, optimized for large-scale datasets. These models include a Jaccard similarity-based recommendation system and a popularity-based recommendation baseline, both of which are highly scalable and suitable for big data environments.

#### Folder Structure:
- **jaccardBaseLine.py**: Implements a Jaccard similarity-based recommendation system using MinHashLSH and performs pairwise correlation analysis for user similarity.
- **popularBaseLine.py**: Implements a popularity-based recommendation system that recommends the most popular items to users. It includes evaluation metrics like MAP, NDCG, and MRR for model assessment.
- **sparkALS**: This subfolder contains an advanced recommendation system using the Alternating Least Squares (ALS) algorithm. It includes code for data preprocessing, model training, and evaluation using ranking metrics.

For large datasets, please refer to the models in this folder for scalable processing and efficient recommendation system training.

### 2. **SVD_KNN**
The `SVD_KNN` folder contains a recommendation system designed for normal-sized datasets, implemented using the Python `surprise` library. This folder includes both SVD and KNN-based recommendation models that generate top-10 recommendations based on user-item ratings.

#### Folder Structure:
- **main.py**: Trains both SVD and KNN models, evaluates their performance using MAP@K, and generates top-10 recommendations for a specific user.
- **KNNBasic.py**: Implements a KNN recommendation model with hyperparameter tuning through grid search.
- **SVD.py**: Implements an SVD recommendation model with grid search for optimal parameter selection.
- **MAP_K.py**: Evaluates the SVD and KNN models using the MAP@K metric.
- **getRec.py**: Retrieves top-10 recommendations for a user and prints them with associated song names.

If you're working with smaller datasets and prefer a Python-based approach, this folder contains models optimized for that scale.

## Requirements
- For the `pySpark` folder:
  - **Apache Spark**
  - **PySpark**
- For the `SVD_KNN` folder:
  - **Python 3.7+**
  - `surprise` library
  - `pandas`

## How to Use
1. **For large-scale datasets**: Use the code in the `pySpark` folder, including the `sparkALS` subfolder, to preprocess, train, and evaluate your recommendation models efficiently with Apache Spark.
2. **For normal-sized datasets**: Explore the `SVD_KNN` folder for SVD and KNN-based recommendation models, which are better suited for smaller datasets and can be run directly in Python.

Each folder provides a README for further instructions on running the models and adjusting parameters.

## Note on Dataset Size
- **Large datasets**: Use the `pySpark` folder for efficient processing and model training.
- **Normal datasets**: Use the `SVD_KNN` folder for smaller datasets that can be handled with Python-based models.

