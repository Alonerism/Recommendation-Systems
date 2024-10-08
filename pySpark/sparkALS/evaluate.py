from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window

# Initialize Spark session
spark = SparkSession.builder \
        .appName('ALS Model with Genre Features') \
        .config('spark.executor.memory', '10g') \
        .config('spark.driver.memory', '10g') \
        .config('spark.sql.shuffle.partitions', '200') \
        .config('spark.executor.memoryOverhead', '1g') \
        .getOrCreate()

def calculate_MAP(predictions, k):
    windowSpec = Window.partitionBy('userId').orderBy(F.col('prediction').desc())
    ranked_predictions = predictions.withColumn("rank", F.rank().over(windowSpec))
    relevant_predictions = ranked_predictions.filter(F.col('relevant') == 1)
    precision_at_rank = relevant_predictions.withColumn("precision_at_rank",
                                                        F.col('relevant') / F.col('rank'))
    avg_precision_per_user = precision_at_rank.groupBy('userId').agg(
        F.avg('precision_at_rank').alias('AP')
    )
    MAP = avg_precision_per_user.select(F.avg('AP').alias('MAP')).first()['MAP']
    return MAP

def calculate_NDCG(predictions, k):
    windowSpec = Window.partitionBy('userId').orderBy(F.col('prediction').desc())
    predictions = predictions.withColumn("rank", F.rank().over(windowSpec))
    top_k = predictions.filter(F.col('rank') <= k)
    DCG = top_k.withColumn('dcg', F.col('relevant') / F.log2(F.col('rank') + 2))
    DCG = DCG.groupBy('userId').agg(F.sum('dcg').alias('DCG'))
    ideal_ranks = predictions.orderBy(F.col('relevant').desc(), F.col('rank'))
    ideal_ranks = ideal_ranks.withColumn("ideal_rank", F.rank().over(windowSpec))
    top_k_ideal = ideal_ranks.filter(F.col('ideal_rank') <= k)
    IDCG = top_k_ideal.withColumn('idcg', F.col('relevant') / F.log2(F.col('ideal_rank') + 2))
    IDCG = IDCG.groupBy('userId').agg(F.sum('idcg').alias('IDCG'))
    NDCG = DCG.join(IDCG, 'userId', 'inner').withColumn('NDCG', F.col('DCG') / F.col('IDCG'))
    mean_NDCG = NDCG.select(F.avg('NDCG').alias('mean_NDCG')).first()['mean_NDCG']
    return mean_NDCG

def calculate_MRR(predictions):
    """
    Calculates the Mean Reciprocal Rank (MRR) for the given predictions DataFrame.
    """
    # Calculate relevance labels
    relevant_label = F.udf(lambda x: 1 if x >= 3.5 else 0, IntegerType())
    predictions = predictions.withColumn("relevant", relevant_label(F.col("rating")))
    
    # Window specification to rank items by predicted score
    windowSpec = Window.partitionBy('userId').orderBy(F.col('prediction').desc())
    ranked_predictions = predictions.withColumn("rank", F.rank().over(windowSpec))
    
    # Filter to get the first relevant item per user
    first_relevant_rank = ranked_predictions.filter(F.col('relevant') == 1) \
                                             .groupBy('userId') \
                                             .agg(F.min('rank').alias('first_relevant_rank'))
    
    # Calculate reciprocal rank
    reciprocal_ranks = first_relevant_rank.select(F.avg(1 / F.col('first_relevant_rank')).alias('MRR'))
    
    # Return the first (and only) value in the DataFrame
    return reciprocal_ranks.first()['MRR']

def ranking_metrics(predictions, k):
    print("Calculating relevance labels...")
    # Calculate relevance labels (1 if relevant, 0 otherwise)
    relevant_label = F.udf(lambda x: 1 if x >= 3.5 else 0, IntegerType())
    predictions = predictions.withColumn("relevant", relevant_label(F.col("rating")))

    # Window specification to rank items by predicted score for all predictions
    windowSpec = Window.partitionBy('userId').orderBy(F.col('prediction').desc())
    predictions = predictions.withColumn("rank", F.rank().over(windowSpec))

    # Filter to top k items
    top_k_predictions = predictions.filter(F.col("rank") <= k)

    # Precision at K
    print("Calculating Precision at K...")
    precision_at_k = top_k_predictions.groupBy("userId").agg(
        F.mean("relevant").alias("precision")
    ).select(F.mean("precision").alias("mean_precision")).first()["mean_precision"]

    # Recall at K
    print("Calculating Recall at K...")
    total_relevant = predictions.filter(F.col("relevant") == 1).groupBy("userId").agg(F.count("*").alias("total_relevant"))
    recall_at_k = top_k_predictions.filter(F.col("relevant") == 1).groupBy("userId").agg(F.count("*").alias("relevant_count"))
    recall_at_k = recall_at_k.join(total_relevant, "userId", "left_outer")
    recall_at_k = recall_at_k.withColumn("recall", F.col("relevant_count") / F.col("total_relevant"))
    recall_at_k = recall_at_k.select(F.mean("recall").alias("mean_recall")).first()["mean_recall"]

    # Mean Average Precision at K
    print("Calculating MAP at K...")
    map_at_k = calculate_MAP(predictions, k)

    # DCG and NDCG
    print("Calculating NDCG at K...")
    ndcg_at_k = calculate_NDCG(predictions, k)

    # Calculate MRR
    print("Calculating MRR...")
    mrr = calculate_MRR(predictions)

    return precision_at_k, recall_at_k, map_at_k, ndcg_at_k, mrr

# Load the predictions
print("Loading predictions...")
best_predictions = spark.read.parquet("FILL IN")

# Evaluate metrics
k = 10
print("Evaluating metrics...")
metrics = ranking_metrics(best_predictions, k)
print(f"Precision@{k}: {metrics[0]}, Recall@{k}: {metrics[1]}, MAP@{k}: {metrics[2]}, NDCG@{k}: {metrics[3]}, MRR: {metrics[4]}")

# Stop the Spark session
spark.stop()