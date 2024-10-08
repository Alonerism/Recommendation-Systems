from SVD import train_svd_model
from KNNBasic import train_knn_model
from MAP_K import evaluate_models
from getRec import get_top_10_recommendations, print_recommendations_with_names
import pandas as pd

# Define dataset path
data_path = 'RATINGS DATA'

# Train models
best_svd, trainset_svd, testset_svd = train_svd_model(data_path)
best_knn, trainset_knn, testset_knn = train_knn_model(data_path)

# Evaluate models (predictions should be generated in svd_model.py and knn_model.py)
evaluate_models(best_svd.test(testset_svd), best_knn.test(testset_knn))

# Generate recommendations for user 1
user_id = 1  # Replace with the actual user ID you want recommendations for
top_10_svd = get_top_10_recommendations(user_id, best_svd, trainset_svd)
top_10_knn = get_top_10_recommendations(user_id, best_knn, trainset_knn)

# Load song data for printing names
spotify_data = pd.read_csv('SONG TITLES DATA')

# Print recommendations
print_recommendations_with_names(top_10_svd, spotify_data)
print_recommendations_with_names(top_10_knn, spotify_data)