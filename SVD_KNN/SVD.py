import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV

# Function to load and train the SVD model
def train_svd_model(data_path):
    # Load the star ratings dataset
    star_ratings_df = pd.read_csv(data_path, header=None)

    # Prepare the data for Surprise
    reader = Reader(rating_scale=(0, 4))
    data = Dataset.load_from_df(star_ratings_df.stack().reset_index(name='rating'), reader)

    # Split the data into training and test sets
    trainset, testset = train_test_split(data, test_size=0.3, random_state=18630499)

    # Define parameter grid for SVD
    param_grid_svd = {
        'n_epochs': [5, 10, 20],
        'lr_all': [0.002, 0.005, 0.01],
        'reg_all': [0.02, 0.05, 0.1]
    }

    # Grid search for SVD
    gs_svd = GridSearchCV(SVD, param_grid_svd, measures=['rmse'], cv=5, n_jobs=-1)
    gs_svd.fit(data)

    best_svd = gs_svd.best_estimator['rmse']
    best_svd.fit(trainset)

    return best_svd, trainset, testset
