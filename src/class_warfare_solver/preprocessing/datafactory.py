from class_warfare_solver.utils.methods import initialize_spark
import pandas as pd
from pyspark.sql import SparkSession, functions as F

class DataFactory():

    def __init__(self, spark:SparkSession, databricks=False):
        # spark doesn't like data not on dbfs, will move to volumes later
        # Plus that way columns will be correct type instead of all str
        prefix = "../../../" if not databricks else ""
        self._class_bases = spark.createDataFrame(pd.read_csv(f"{prefix}data/Class Bases.csv"))
        self._class_growths = spark.createDataFrame(pd.read_csv(f"{prefix}data/Class Growths.csv"))
        self._character_bases = spark.createDataFrame(pd.read_csv(f"{prefix}data/Character Bases.csv"))
        self._character_growths = spark.createDataFrame(pd.read_csv(f"{prefix}data/Character Growths.csv"))

    def get_class_bases(self):
        class_bases = self._class_bases
        
        class_bases = (
            class_bases
            .withColumn("Class", F.trim(F.get(F.split(class_bases.Class, '[(*)]', 2), 0)))
            .withColumn("Class", F.split(class_bases.Class, '[/,]', 2))
            .withColumn("Class", F.explode(class_bases.Class))
            .withColumn("Class", F.trim(F.col('Class')))
            .where(F.col("Class").isNotNull() & (F.col("Class") != ""))
            .where(F.col("Game") != "Game")
            .dropDuplicates(["Game", "Class"])
            .fillna("0", subset=["Lck"])
            .withColumn("Mag", F.when(F.col("Mag").isNull(), F.col("Str")).otherwise(F.col("Mag")))
            .withColumn("Res", F.when(F.col("Res").isNull(), F.col("Def")).otherwise(F.col("Res")))
            .withColumn("Mov", F.trim(F.get(F.split(class_bases.Mov, '[»]', 2), 0)))
        )
        self._class_bases = class_bases.select([
            "Game", "Class", "HP", "Str", "Mag", "Skl", "Spd", "Lck", "Def", "Res", "Mov"
        ])
        self._class_bases.show()
        return self._class_bases
    
    def get_class_growths(self):
        class_growths = self._class_growths
        class_growths = (class_growths
            .withColumn("Class", F.trim(F.get(F.split(class_growths.Class, '[(*)]', 2), 0)))
            .withColumn("Class", F.split(class_growths.Class, '[/,]', 4))
            .withColumn("Class", F.explode(class_growths.Class))
            .withColumn("Class", F.trim(F.col('Class')))
            .where(F.col("Class").isNotNull() & (F.col("Class") != ""))
            .where(F.col("Game") != "Game")
            .dropDuplicates(["Game", "Class"])
            .withColumn("Mag", F.when(F.col("Mag").isNull() & F.col("Game").isin([2, 6, 7, 8]), F.col("Str")).otherwise(F.col("Mag")))
            .withColumn("Res", F.when(F.col("Res").isNull() & F.col("Game").isin([5]), F.col("Def")).otherwise(F.col("Res")))
            .fillna("0")
        )

        self._class_growths = class_growths.select([
            "Game", "Class", "HP", "Str", "Mag", "Skl", "Spd", "Lck", "Def", "Res"
        ])
        self._class_growths.show()
        return self._class_growths
    
    def get_character_bases(self):
        character_bases = self._character_bases

        character_bases = (
            character_bases
            .withColumn("Name", F.trim(F.get(F.split(character_bases.Name, '[ *]', 2), 0)))
            .where(F.col("Class").isNotNull() | (F.col("Game") == 13))
            .dropDuplicates(["Game", "Name"])
            .withColumn("Class", F.trim(F.get(F.split(character_bases.Class, '[ *]', 2), 0)))
            .where(F.col("Game") != "Game")
            .withColumn("Mag", F.when(F.col("Mag").isNull() & F.col("Game").isin([2, 6, 7, 8]), F.col("Str")).otherwise(F.col("Mag")))
            .withColumn("Res", F.when(F.col("Res").isNull() & F.col("Game").isin([5]), F.col("Def")).otherwise(F.col("Res")))
            .fillna("0", subset=["Mov"])
        )
        for col in ["HP", "Str", "Mag", "Skl", "Spd", "Lck", "Def", "Res", "Mov"]:
            character_bases = character_bases.withColumn(col, F.cast(int, F.get(F.split(F.col(col), '[+]', 2), 0)))

        self._character_bases = character_bases.select([
            "Game", "Name", "Class", "HP", "Str", "Mag", "Skl", "Spd", "Lck", "Def", "Res", "Mov"
        ])
        self._character_bases.show()
        return self._character_bases

if __name__ == "__main__":
    spark = initialize_spark()

    df = DataFactory(spark)

    df.get_character_bases()