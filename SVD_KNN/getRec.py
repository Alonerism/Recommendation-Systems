# Function to get top N recommendations for a user
def get_top_10_recommendations(user_id, model, trainset, n=10):
    inner_user_id = trainset.to_inner_uid(user_id)
    user_ratings = trainset.ur[inner_user_id]
    rated_items = {item_id for item_id, _ in user_ratings}

    all_items = set(range(trainset.n_items))
    unrated_items = all_items - rated_items

    predictions = [(item_id, model.predict(user_id, trainset.to_raw_iid(item_id)).est) for item_id in unrated_items]
    predictions.sort(key=lambda x: x[1], reverse=True)

    top_n = [(trainset.to_raw_iid(item_id), rating) for item_id, rating in predictions[:n]]
    return top_n

# Function to print top 10 recommendations with song names
def print_recommendations_with_names(recommendations, spotify_data):
    for song_id, predicted_rating in recommendations:
        song_name = spotify_data.loc[spotify_data['songNumber'] == song_id, 'track_name'].values[0]
        print(f"Song ID: {song_id}, Song Name: {song_name}, Predicted Rating: {predicted_rating:.2f}")
