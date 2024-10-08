import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split, GridSearchCV

# Function to load and train the KNN model
def train_knn_model(data_path):
    # Load the star ratings dataset
    star_ratings_df = pd.read_csv(data_path, header=None)

    # Filter out users (rows) and items (columns) with no ratings
    star_ratings_df = star_ratings_df.dropna(how='all', axis=1)  # Drop items with no ratings
    star_ratings_df = star_ratings_df.dropna(how='all', axis=0)  # Drop users with no ratings

    # Prepare the data for Surprise
    reader = Reader(rating_scale=(0, 4))
    data = Dataset.load_from_df(star_ratings_df.stack().reset_index(name='rating'), reader)

    # Split the data into training and test sets
    trainset, testset = train_test_split(data, test_size=0.3, random_state=18630499)

    # Define parameter grid for KNNBasic
    param_grid_knn = {
        'k': [20, 40, 60],
        'sim_options': {
            'name': ['msd', 'pearson'],
            'user_based': [True, False]
        }
    }

    # Grid search for KNNBasic
    gs_knn = GridSearchCV(KNNBasic, param_grid_knn, measures=['rmse'], cv=5, n_jobs=-1)
    gs_knn.fit(data)

    best_knn = gs_knn.best_estimator['rmse']
    best_knn.fit(trainset)

    return best_knn, trainset, testset