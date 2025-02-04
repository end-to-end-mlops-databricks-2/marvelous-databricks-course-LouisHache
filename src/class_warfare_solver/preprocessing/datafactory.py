from class_warfare_solver.utils.methods import initialize_spark
import pandas as pd
from pyspark.sql import SparkSession

class DataFactory():

    def __init__(self, spark:SparkSession):
        # spark doesn't like data not on dbfs
        self.class_bases = spark.createDataFrame(pd.read_csv("../../../data/Class Bases.csv"))
        self.class_growths = spark.createDataFrame(pd.read_csv("../../../data/Class Growths.csv"))
        self.character_bases = spark.createDataFrame(pd.read_csv("../../../data/Character Bases.csv"))
        self.character_growths = spark.createDataFrame(pd.read_csv("../../../data/Character Growths.csv"))

    def get_class_bases(self):
        self.class_bases.show()
        self.class_growths.show()
        self.character_bases.show()
        self.character_growths.show()
        pass

if __name__ == "__main__":
    spark = initialize_spark()

    df = DataFactory(spark)

    df.get_class_bases()