from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline
import json

# Initialize Spark session with increased resources
spark = SparkSession.builder \
        .appName('ALS Model with Genre Features') \
        .config('spark.executor.memory', '16g') \
        .config('spark.executor.cores', '4') \
        .config('spark.executor.instances', '10') \
        .config('spark.driver.memory', '16g') \
        .config('spark.executor.memoryOverhead', '4g') \
        .config('spark.sql.shuffle.partitions', '800') \
        .config('spark.speculation', 'true') \
        .getOrCreate()

# Load pre-processed data from Parquet
print("Reading pre-processed data from Parquet...")
data = spark.read.parquet("FILL IN")

data.printSchema()  # This will show you what columns are available in your DataFrame

# No need to repartition again; data is already partitioned by userId
# Split the data into training and testing sets
train, test = data.randomSplit([0.8, 0.2], seed=42)

# Define parameter grid for model tuning
paramGrid = ParamGridBuilder() \
    .addGrid(ALS.rank, [10, 50, 100]) \
    .addGrid(ALS.maxIter, [5, 10, 20]) \
    .addGrid(ALS.regParam, [0.01, 0.05, 0.1]) \
    .build()

# Custom function to perform grid search and print parameters and RMSE
def custom_cross_validation(train, paramGrid, numFolds=3):
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    results = []

    for params in paramGrid:
        als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative=True, coldStartStrategy="drop")
        als.setParams(**{param.name: value for param, value in params.items()})
        
        print(f"Training with parameters: {params}")
        
        # Use TrainValidationSplit for parallel execution and early stopping
        tvs = TrainValidationSplit(estimator=als, 
                                   estimatorParamMaps=[params],
                                   evaluator=evaluator,
                                   trainRatio=0.8, 
                                   parallelism=4)  # Run up to 4 models in parallel

        model = tvs.fit(train)
        rmse = model.validationMetrics[0]
        print(f"RMSE: {rmse}")
        print()
        
        # Save results
        param_dict = {param.name: value for param, value in params.items()}
        results.append((param_dict, rmse))

    return results

# Perform custom cross-validation
results = custom_cross_validation(train, paramGrid)

# Find the best model parameters based on RMSE
best_rmse = float('inf')
best_params = None

for params, rmse in results:
    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params

# Print the best model parameters and RMSE
print(f"Best model parameters: {best_params}")
print(f"Best model RMSE: {best_rmse}")

# Re-train the best model using the best parameters
best_als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative=True, coldStartStrategy="drop")
for param, value in best_params.items():
    best_als = best_als.setParam(param, value)

best_model = best_als.fit(train)

# Evaluate the best model on the test set
print("Evaluating best model...")
predictions = best_model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
final_rmse = evaluator.evaluate(predictions)
print(f"Final RMSE on test data: {final_rmse}")

# Save predictions of the best model
print("Saving predictions of the best model...")
predictions.write.mode('overwrite').parquet("FILL IN")

# Stop Spark session
print("Stopping Spark session...")
spark.stop()