import numpy as np

# Function to calculate MAP@K for the model
def average_precision_at_k(predictions, k=10, threshold=3.5):
    user_est_true = {}
    for prediction in predictions:
        uid = prediction.uid
        true_r = prediction.r_ui
        est = prediction.est

        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))

    average_precisions = []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        relevant_ranks = [i + 1 for i, (_, true_r) in enumerate(user_ratings[:k]) if true_r >= threshold]
        if relevant_ranks:
            precisions = [len([r for r in relevant_ranks if r <= rank]) / rank for rank in relevant_ranks]
            average_precisions.append(np.mean(precisions))
        else:
            average_precisions.append(0.0)

    return np.mean(average_precisions)

# Function to evaluate SVD and KNN models
def evaluate_models(predictions_svd, predictions_knn):
    map_k_svd = average_precision_at_k(predictions_svd, k=10)
    print(f'MAP@K=10 for the best SVD model: {map_k_svd}')

    map_k_knn = average_precision_at_k(predictions_knn, k=10)
    print(f'MAP@K=10 for the best KNN model: {map_k_knn}')
