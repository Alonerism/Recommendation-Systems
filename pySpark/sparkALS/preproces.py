from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark session with adjusted resources
spark = SparkSession.builder \
        .appName('22') \
        .config('spark.executor.memory', '4g') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.cores', '2') \
        .config('spark.executor.instances', '4') \
        .config('spark.sql.shuffle.partitions', '200') \
        .config('spark.executor.memoryOverhead', '1g') \
        .getOrCreate()

# Function to read files
def readFiles(filename):
    return spark.read.format('csv').options(header='true', inferschema='true').load(filename)

# Load data
ratings_file = 'FILL IN' #LARGE
movies_file = 'FILL IN' #LARGE
print("Reading data files...")
ratings = readFiles(ratings_file).drop('timestamp')  # Drop the timestamp column
movies = readFiles(movies_file)

# Process genres and index them
print("Processing and indexing genres...")
genres = movies.select('movieId', F.explode(F.split(F.col('genres'), '\\|')).alias('genre'))
genreIndexer = StringIndexer(inputCol='genre', outputCol='genreIdx')
genre_model = genreIndexer.fit(genres)
indexed_genres = genre_model.transform(genres)

# OneHotEncode the indexed genres
encoder = OneHotEncoder(inputCol="genreIdx", outputCol="genreVec")
print("Fitting OneHotEncoder...")
encoder_model = encoder.fit(indexed_genres)
print("Applying OneHotEncoder...")
encoded_genres = encoder_model.transform(indexed_genres)

# Aggregate the genre vectors for each movie
print("Aggregating genre vectors...")
genre_vector = encoded_genres.groupBy("movieId").agg(F.max("genreVec").alias("genreVec"))

# Join the genre data back to the movies dataframe, and then with ratings
print("Joining genre data with movies and ratings...")
enhanced_movies = movies.join(genre_vector, "movieId", "left").select(movies["*"], genre_vector["genreVec"])

# Broadcast the enhanced movies data if it is small enough
broadcast_movies = spark.sparkContext.broadcast(enhanced_movies.collect())
broadcast_movies_df = spark.createDataFrame(broadcast_movies.value)

ratings_enhanced = ratings.join(broadcast_movies_df, "movieId", "inner") \
                          .select(ratings["userId"], ratings["movieId"], ratings["rating"], broadcast_movies_df["genreVec"])

# Split the data into manageable chunks
chunk_size = 5000  # Adjust based on memory and performance considerations
window = Window.partitionBy("userId").orderBy("userId")
user_ids_with_row_num = ratings_enhanced.select("userId").distinct().withColumn("row_num", F.row_number().over(window))
num_chunks = user_ids_with_row_num.count() // chunk_size + 1

# Function to process and write each chunk
def process_chunk(chunk_num, chunk_size):
    start_id = chunk_num * chunk_size
    end_id = start_id + chunk_size

    user_ids_chunk = user_ids_with_row_num.filter((F.col("row_num") > start_id) & (F.col("row_num") <= end_id)).select("userId")

    chunk = ratings_enhanced.join(user_ids_chunk, "userId")

    # Write chunk to Parquet
    chunk.write.mode('append').partitionBy("userId").parquet(f"FILL IN. ({chunk_num} Keep this)")

# Process and write each chunk
for chunk_num in range(num_chunks):
    process_chunk(chunk_num, chunk_size)

print("SUCCESS")

# Stop Spark session
print("Stopping Spark session...")
spark.stop()