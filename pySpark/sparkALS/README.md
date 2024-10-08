# Spark ALS Recommendation System

This folder contains a recommendation system built using Apache Spark and the Alternating Least Squares (ALS) algorithm. The system processes large datasets, trains an ALS model with additional genre features, and evaluates model performance using various ranking metrics.

## Folder Structure
- **preproces.py**: Preprocesses the data by reading in large datasets, encoding genres, and preparing the data for model training.
- **trainALS.py**: Trains the ALS model, performs grid search for hyperparameter tuning, and evaluates model performance on a test set.
- **evaluate.py**: Evaluates the trained ALS model using metrics such as Precision@K, Recall@K, MAP@K, NDCG@K, and MRR.

## Requirements
- Apache Spark
- PySpark
- A dataset in CSV or Parquet format

## How to Use
1. **Preprocess the data**: Run `preproces.py` to prepare the dataset for model training.
2. **Train the model**: Use `trainALS.py` to train the ALS model with grid search for optimal parameters.
3. **Evaluate the model**: After training, use `evaluate.py` to compute ranking metrics for model evaluation.

This setup is designed to handle large-scale datasets and provides flexibility in adjusting resources for Spark clusters.
 