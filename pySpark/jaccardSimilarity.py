from pyspark.sql import SparkSession
from pyspark.sql.functions import col, corr, expr, avg, collect_list
from pyspark.ml.feature import MinHashLSH, VectorAssembler, CountVectorizer
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import functions as F

def calculate_pairwise_correlations(spark, ratings, pairs):
    correlations = []
    for row in pairs.collect():
        user1_ratings = ratings.filter(col("userId") == row["User1"]).select("movieId", "rating").alias("ratings1")
        user2_ratings = ratings.filter(col("userId") == row["User2"]).select("movieId", "rating").alias("ratings2")
        joined_ratings = user1_ratings.join(user2_ratings, "movieId")
        correlation = joined_ratings.select(corr("ratings1.rating", "ratings2.rating").alias("correlation")).first()["correlation"]
        if correlation is not None:
            correlations.append(correlation)
    return sum(correlations) / len(correlations) if correlations else 0

def generate_random_pairs(users, count=100):
    # Assign unique sequential IDs to handle large user IDs and self-join issue
    users = users.withColumn("id", monotonically_increasing_id())
    return users.alias("u1").crossJoin(users.alias("u2")) \
                .filter("u1.id < u2.id") \
                .selectExpr("u1.userId as User1", "u2.userId as User2") \
                .limit(count)

def main(spark, userID):
    print('Dataframe loading and SQL query')
    print()
    ratings_path = 'hdfs:/user/abf386_nyu_edu/ml-latest/ratings.csv'
    ratings = spark.read.csv(ratings_path, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')

    movie_counts = ratings.groupBy("movieId").count()
    filtered_ratings = ratings.join(movie_counts, "movieId", "inner")

    user_movies = filtered_ratings.groupBy("userId").agg(collect_list("movieId").alias("movieIds"))

    # Convert movieIds to array<string>
    user_movies = user_movies.withColumn("movieIds", col("movieIds").cast("array<string>"))

    # Using CountVectorizer to transform movieIds into a vector
    cv = CountVectorizer(inputCol="movieIds", outputCol="features")
    cv_model = cv.fit(user_movies)
    user_features = cv_model.transform(user_movies)

    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=4)
    model = mh.fit(user_features)

    similar_users = model.approxSimilarityJoin(user_features, user_features, threshold=0.5, distCol="JaccardDistance") \
                         .filter("datasetA.userId < datasetB.userId") \
                         .select(col("datasetA.userId").alias("User1"), col("datasetB.userId").alias("User2"), "JaccardDistance") \
                         .orderBy("JaccardDistance") \
                         .limit(100)

    similar_users.show(100, False)

    average_correlation_top_pairs = calculate_pairwise_correlations(spark, ratings, similar_users)
    print(f"Average Correlation of Top 100 Pairs: {average_correlation_top_pairs}")
    print()

    users = ratings.select("userId").distinct()
    random_pairs = generate_random_pairs(users, 100)
    average_correlation_random_pairs = calculate_pairwise_correlations(spark, ratings, random_pairs)
    print(f"Average Correlation of 100 Random Pairs: {average_correlation_random_pairs}")
    print()

    # Calculate the average Jaccard distance for the top 100 pairs
    average_jaccard_distance_top_pairs = similar_users.select(avg("JaccardDistance")).first()[0]
    print(f"Average Jaccard Distance of Top 100 Pairs: {average_jaccard_distance_top_pairs}")
    print()

    # Calculate the Jaccard distances for the random pairs
    random_pairs_with_jaccard = model.approxSimilarityJoin(user_features, user_features, threshold=1.0, distCol="JaccardDistance") \
                                     .filter(expr("datasetA.userId < datasetB.userId")) \
                                     .select(col("datasetA.userId").alias("User1"), col("datasetB.userId").alias("User2"), "JaccardDistance") \
                                     .join(random_pairs, ["User1", "User2"], "inner") \
                                     .limit(100)

    # Calculate the average Jaccard distance for 100 random pairs
    average_jaccard_distance_random_pairs = random_pairs_with_jaccard.select(avg("JaccardDistance")).first()[0]
    print(f"Average Jaccard Distance of 100 Random Pairs: {average_jaccard_distance_random_pairs}")
    print()

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName('22') \
        .config('spark.executor.memory', '10g') \
        .config('spark.driver.memory', '10g') \
        .config('spark.sql.shuffle.partitions', '200') \
        .config('spark.executor.memoryOverhead', '1g') \
        .getOrCreate()
    userID = 'default_user'  # Directly setting userID here
    main(spark, userID)