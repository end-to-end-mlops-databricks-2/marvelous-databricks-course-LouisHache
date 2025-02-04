from pyspark.sql import SparkSession


def initialize_spark() -> SparkSession:
    builder = SparkSession.builder.appName("class-warfare-solver")
    session = builder.getOrCreate()
    return session