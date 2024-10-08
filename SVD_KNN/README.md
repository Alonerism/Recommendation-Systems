# SVD and KNN-Based Recommendation System

This folder contains a recommendation system implemented using two different algorithms: Singular Value Decomposition (SVD) and KNN. The system processes a user-item rating dataset and generates top-10 recommendations for users based on the trained models.

## Folder Structure
- **main.py**: Trains both SVD and KNN models, evaluates their performance using MAP@K, and generates top-10 recommendations for a specified user.
- **KNNBasic.py**: Implements a KNN recommendation model with grid search to tune hyperparameters.
- **SVD.py**: Implements an SVD recommendation model with grid search for optimal hyperparameter tuning.
- **MAP_K.py**: Evaluates the SVD and KNN models using the MAP@K metric to assess recommendation quality.
- **getRec.py**: Retrieves the top-10 recommendations for a given user and prints them alongside song names from a dataset.

## Requirements
- Python 3.7+
- `surprise` library for recommendation models
- `pandas` for data handling

## How to Use
1. **Train and Evaluate Models**: Run `main.py` to train both SVD and KNN models, evaluate their performance, and generate recommendations for a specified user.
2. **Generate Recommendations**: After training, top-10 song recommendations are printed, using the provided song titles dataset.

## Note on Dataset Size
This system is designed for "normal-sized" datasets. If you are working with very large datasets, please refer to the `pySpark` folder, which contains implementations optimized for large-scale data processing using Apache Spark.

This setup allows easy experimentation with both SVD and KNN models to generate recommendations based on user-item ratings.
