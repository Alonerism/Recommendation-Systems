from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType

# Initialize Spark session
spark = SparkSession.builder \
    .appName('Popular Movies Baseline Model with Evaluation Metrics') \
    .config('spark.executor.memory', '10g') \
    .config('spark.driver.memory', '10g') \
    .config('spark.sql.shuffle.partitions', '200') \
    .config('spark.executor.memoryOverhead', '1g') \
    .getOrCreate()

# Load the ratings data
print("Loading ratings data...")
ratings = spark.read.format('csv').options(header='true', inferschema='true').load("LOAD CSV")

# Define a UDF to determine relevance based on ratings
def relevance_label_udf():
    return F.udf(lambda x: 1 if x >= 3.5 else 0, IntegerType())

# Calculate the most popular movies based on the count of ratings
popular_movies = ratings.groupBy("movieId").agg(
    F.count("rating").alias("rating_count"),
    F.avg("rating").alias("average_rating")
).orderBy(F.desc("rating_count"), F.desc("average_rating"))

# Select the top 'k' movies as recommendations for all users
k = 10
top_k_popular_movies = popular_movies.limit(k).select("movieId").rdd.flatMap(lambda x: x).collect()

# Simulate recommendations for all users
users = ratings.select("userId").distinct()
recommended_to_all = users.withColumn("recommended_movies", F.array([F.lit(x) for x in top_k_popular_movies]))

# Calculate relevance of each rating
ratings = ratings.withColumn("relevant", relevance_label_udf()(F.col("rating")))

# Join ratings with user recommendations to simulate user interaction with recommended movies
user_movie_interaction = ratings.join(recommended_to_all, "userId")
user_movie_interaction = user_movie_interaction.withColumn("is_recommended", F.when(F.col("movieId").isin(top_k_popular_movies), 1).otherwise(0))

# Window specification to rank items by user
window_spec = Window.partitionBy('userId').orderBy(F.col('rating').desc())

# MAP Calculation
ranked_interactions = user_movie_interaction.withColumn("rank", F.rank().over(window_spec))
precision_at_rank = ranked_interactions.withColumn("precision_at_rank", F.col("relevant") / F.col("rank"))
avg_precision = precision_at_rank.groupBy("userId").agg(F.avg("precision_at_rank").alias("AP"))
map_k = avg_precision.select(F.avg("AP").alias("MAP")).first()['MAP']

# NDCG Calculation
dcg = ranked_interactions.withColumn('dcg', F.col("relevant") / F.log2(F.col("rank") + 1))
dcg = dcg.groupBy("userId").agg(F.sum("dcg").alias("DCG"))
idcg = ranked_interactions.withColumn("idcg", 1 / F.log2(F.col("rank") + 1))
idcg = idcg.groupBy("userId").agg(F.sum("idcg").alias("IDCG"))
ndcg = dcg.join(idcg, "userId", "inner").withColumn("NDCG", F.col("DCG") / F.col("IDCG"))
ndcg_k = ndcg.select(F.avg("NDCG").alias("mean_NDCG")).first()["mean_NDCG"]

# MRR Calculation
first_relevant_rank = ranked_interactions.filter(F.col("relevant") == 1).groupBy("userId").agg(F.min("rank").alias("first_relevant_rank"))
mrr = first_relevant_rank.select(F.avg(1 / F.col("first_relevant_rank")).alias("MRR")).first()["MRR"]

# Print results
print(f"MAP@{k}: {map_k}, NDCG@{k}: {ndcg_k}, MRR: {mrr}")

# Stop the Spark session
spark.stop()