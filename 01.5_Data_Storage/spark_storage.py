from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import os

# 1. Create Spark Session
spark = SparkSession.builder \
    .appName("CSVtoCassandra") \
    .config("spark.cassandra.connection.host", "127.0.0.1") \
    .config("spark.cassandra.connection.port", "9042") \
    .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.4.0") \
    .getOrCreate()

df = spark.read.csv("2_data_combined.csv", header=True, inferSchema=True)

# 2. Read CSV files
def read_csv_files(directory):
    df = spark.read.option("header", "true").csv(directory)
    return df

raw_data_df = read_csv_files("2_data_combined.csv")
raw_data_df.show()

raw_data_df.printSchema()

# 3. Save data to Cassandra
def save_to_cassandra(df, keyspace, table):
    try:
        df.show(5)
        df.write \
            .format("org.apache.spark.sql.cassandra") \
            .option("keyspace", keyspace) \
            .option("table", table) \
            .mode("append") \
            .save()
    except Exception as e:
        print(f"Error saving data to Cassandra: {e}")

save_to_cassandra(raw_data_df, "space", "data")

spark.stop()
