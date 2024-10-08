# PySpark Recommendation Baseline Models

This folder contains baseline recommendation models implemented using Apache Spark. These models provide foundational approaches for recommendation systems, utilizing methods such as Jaccard similarity and popularity-based recommendations.

## Folder Structure
- **jaccardBaseLine.py**: Implements a Jaccard similarity-based recommendation system. It uses MinHashLSH to identify similar users based on their movie preferences and calculates pairwise correlations and Jaccard distances.
- **popularBaseLine.py**: Implements a popularity-based recommendation system. The model recommends the most popular movies to all users and evaluates the results using MAP, NDCG, and MRR metrics.

Additionally, this folder includes the `SparkALS` subfolder, which contains a full recommendation system using the ALS algorithm. The code in `SparkALS` allows for training an ALS model with genre-based features and evaluating its performance.

## Requirements
- Apache Spark
- PySpark
- Datasets in CSV or Parquet format

## How to Use
1. **Jaccard Similarity Model**: Run `jaccardBaseLine.py` to identify similar users and evaluate correlations.
2. **Popularity Baseline Model**: Use `popularBaseLine.py` to recommend the most popular movies and compute evaluation metrics for the model's performance.
3. **Spark ALS Model**: Explore the `SparkALS` folder for an advanced recommendation system using the Alternating Least Squares (ALS) algorithm.

All scripts are designed to process large-scale datasets efficiently with Spark.
