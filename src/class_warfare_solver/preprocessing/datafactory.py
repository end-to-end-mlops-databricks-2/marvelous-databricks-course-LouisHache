import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


class DataFactory:
    def __init__(self, spark:SparkSession, databricks=False):
        self.spark = spark
        self.prefix = "../../../" if not databricks else ""
        self._is_data_cleaned = False

    def load_raw_tables(self):
        self._class_bases = self.spark.createDataFrame(pd.read_csv(f"{self.prefix}data/Class Bases.csv"))
        self._class_growths = self.spark.createDataFrame(pd.read_csv(f"{self.prefix}data/Class Growths.csv"))
        self._character_bases = self.spark.createDataFrame(pd.read_csv(f"{self.prefix}data/Character Bases.csv"))
        self._character_growths = self.spark.createDataFrame(pd.read_csv(f"{self.prefix}data/Character Growths.csv"))

        return None

    def search_for_copper(self):
        self._clean_class_growths()
        self._clean_class_bases()
        self._clean_character_bases()
        self._clean_character_growths()
        self._is_data_cleaned = True

        return None

    def find_gold(self):
        if not self._is_data_cleaned:
            raise ValueError(raw_errror)

        self._build_class_features()
        self._build_character_features()

        return None

    def split_and_save(self):
        # Game 13 is our test set since characters don't have attributed classes
        test_set = self.character_df.where(F.col("Game") == 13)

        games = list(range(1,15))
        games.remove(13)

        train_set = self.spark.createDataFrame([], self.character_df.schema)
        validation_set = self.spark.createDataFrame([], self.character_df.schema)

        # Tricky since we want the split to be equal within all games
        for game in games:
            game_characters = self.character_df.filter(F.col("Game") == game)

            split_A, split_B = game_characters.randomSplit([0.8, 0.2], seed=42069)

            train_set = train_set.union(split_A)
            validation_set = validation_set.union(split_B)


        catalog = os.getenv('VOLUME')
        schema = os.getenv('SCHEMA')

        train_set.write.mode("overwrite").saveAsTable(
            f"{catalog}.{schema}.train_set"
        )

        validation_set.write.mode("overwrite").saveAsTable(
            f"{catalog}.{schema}.validation_set"
        )

        test_set.write.mode("overwrite").saveAsTable(
            f"{catalog}.{schema}.test_set"
        )

        return None

    def _build_class_features(self):
        class_bases = self._class_bases
        class_growths = self._class_growths

        class_df = class_bases.join(class_growths, on=["Game", "Class"], how="inner")

        for col in class_df.columns:
            if col not in ["Game", "Class"]:
                class_df = class_df.withColumnRenamed(col, "class_" + col)

        # Vectorization will come later
        self.class_df = class_df
        return None

    def _build_character_features(self):
        character_bases = self._character_bases
        character_growths = self._character_growths

        character_df = character_bases.join(character_growths, on=["Game", "Name"], how="inner")

        # We need to remove the class stats from everyone to get their absolute personal stats
        character_df = character_df.join(self.class_df, ["Game", "Class"], "left")

        for col_name in character_df.columns:
            if col_name not in ["Game", "Class", "Name", "Mov"] and "class" not in col_name:
                character_df = character_df.withColumn(
                    col_name,
                    F.when(F.col("Game") != 13, F.cast(int, (F.col(col_name) - F.col("class_" + col_name)))).otherwise(
                        F.col(col_name)
                    ),
                )
        character_df = character_df.withColumn(
            "Mov", F.cast(int, F.when(F.col("Game").isin([4, 13]), 0).otherwise(F.col("Mov") - F.col("class_Mov")))
        )

        character_df = character_df.select(
            [
                "Game",
                "Name",
                "Class",
                "HP",
                "Str",
                "Mag",
                "Skl",
                "Spd",
                "Lck",
                "Def",
                "Res",
                "Mov",
                "gHP",
                "gStr",
                "gMag",
                "gSkl",
                "gSpd",
                "gLck",
                "gDef",
                "gRes",
            ]
        )

        # Vectorization will come later, maybe
        self.character_df = character_df
        return None

    def _clean_class_bases(self):
        class_bases = self._class_bases

        class_bases = (
            class_bases.withColumn("Class", F.trim(F.get(F.split(class_bases.Class, "[(*)]", 2), 0)))
            .withColumn("Class", F.split(class_bases.Class, "[/,]", 30))
            .withColumn("Class", F.explode(class_bases.Class))
            .withColumn("Class", F.trim(F.col("Class")))
            .where(F.col("Class").isNotNull() & (F.col("Class") != ""))
            .where(F.col("Game") != "Game")
            .dropDuplicates(["Game", "Class"])
            .fillna("0", subset=["Lck"])
            .withColumn("Mag", F.when(F.col("Mag").isNull(), F.col("Str")).otherwise(F.col("Mag")))
            .withColumn("Res", F.when(F.col("Res").isNull(), F.col("Def")).otherwise(F.col("Res")))
            .withColumn("Mov", F.trim(F.get(F.split(class_bases.Mov, "[»]", 2), 0)))
        )

        for col in ["HP", "Str", "Mag", "Skl", "Spd", "Lck", "Def", "Res", "Mov"]:
            class_bases = class_bases.withColumn(col, F.col(col).cast("int"))

        self._class_bases = class_bases.select(
            ["Game", "Class", "HP", "Str", "Mag", "Skl", "Spd", "Lck", "Def", "Res", "Mov"]
        )
        return None

    def _clean_class_growths(self):
        class_growths = self._class_growths

        class_growths = (
            class_growths.withColumn("Class", F.trim(F.get(F.split(class_growths.Class, "[(*)]", 2), 0)))
            .withColumn("Class", F.split(class_growths.Class, "[/,]", 30))
            .withColumn("Class", F.explode(class_growths.Class))
            .withColumn("Class", F.trim(F.col("Class")))
            .where(F.col("Class").isNotNull() & (F.col("Class") != ""))
            .where(F.col("Game") != "Game")
            .dropDuplicates(["Game", "Class"])
            .withColumn(
                "Mag",
                F.when(F.col("Mag").isNull() & F.col("Game").isin([2, 6, 7, 8]), F.col("Str")).otherwise(F.col("Mag")),
            )
            .withColumn(
                "Res", F.when(F.col("Res").isNull() & F.col("Game").isin([5]), F.col("Def")).otherwise(F.col("Res"))
            )
            .fillna("0")
        )

        for col in ["HP", "Str", "Mag", "Skl", "Spd", "Lck", "Def", "Res"]:
            class_growths = class_growths.withColumn("g" + col, F.col(col).cast("int"))

        self._class_growths = class_growths.select(
            ["Game", "Class", "gHP", "gStr", "gMag", "gSkl", "gSpd", "gLck", "gDef", "gRes"]
        )
        return None

    def _clean_character_bases(self):
        character_bases = self._character_bases

        character_bases = (
            character_bases.withColumn("Name", F.trim(F.get(F.split(character_bases.Name, "[ *]", 2), 0)))
            .where(F.col("Class").isNotNull() | (F.col("Game") == 13))
            .dropDuplicates(["Game", "Name"])
            .withColumn("Class", F.trim(F.get(F.split(character_bases.Class, "[*]", 2), 0)))
            .where(F.col("Game") != "Game")
            .withColumn(
                "Mag",
                F.when(F.col("Mag").isNull() & F.col("Game").isin([2, 6, 7, 8]), F.col("Str")).otherwise(F.col("Mag")),
            )
            .withColumn(
                "Res", F.when(F.col("Res").isNull() & F.col("Game").isin([5]), F.col("Def")).otherwise(F.col("Res"))
            )
            .fillna("0", subset=["Mov"])
        )
        for col in ["HP", "Str", "Mag", "Skl", "Spd", "Lck", "Def", "Res", "Mov"]:
            character_bases = character_bases.withColumn(col, F.get(F.split(F.col(col), "[+]", 2), 0).cast("int"))

        self._character_bases = character_bases.select(
            ["Game", "Name", "Class", "HP", "Str", "Mag", "Skl", "Spd", "Lck", "Def", "Res", "Mov"]
        )
        return None

    def _clean_character_growths(self):
        character_growths = self._character_growths

        character_growths = (
            character_growths.withColumn("Name", F.trim(F.get(F.split(character_growths.Name, "[(*]", 2), 0)))
            .dropDuplicates(["Game", "Name"])
            .where(F.col("Game") != "Game")
            .withColumn(
                "Mag",
                F.when(F.col("Mag").isNull() & F.col("Game").isin([2, 6, 7, 8]), F.col("Str")).otherwise(F.col("Mag")),
            )
            .withColumn(
                "Res", F.when(F.col("Res").isNull() & F.col("Game").isin([5]), F.col("Def")).otherwise(F.col("Res"))
            )
        )
        for col in ["HP", "Str", "Mag", "Skl", "Spd", "Lck", "Def", "Res"]:
            character_growths = character_growths.withColumn(
                "g" + col, F.when(F.col(col) == "–", 0).otherwise(F.col(col)).cast("int")
            )

        self._character_growths = character_growths.select(
            [
                "Game",
                "Name",
                "gHP",
                "gStr",
                "gMag",
                "gSkl",
                "gSpd",
                "gLck",
                "gDef",
                "gRes",
            ]
        )
        return None


raw_errror = """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@@@@@%#####%%@%%%%%%%%%%%%%%%%%%%%%%@@
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@@@@@%#####%%@%%%%%%%%%%%%%%%%%%%%%%@@
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@%%%%%@%%%%%%%%%%%%%%%@@@%####%@@@%%###%%@@@%%%%%%%%%%%%%%%%%%%%%@@
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#:...-+-.........+...=@+.......:*@%%###%%@@@%%%%%%%%%%%%%%%%%%%%%@@
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#:...-*=:......::*...=*...:=-...-%%%###%%%@@@%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@#:.. -%@@*:...=@@@+.:*+...:**...-#%%###%%%@%@%%%%%%%%%%%%%%%%%%%%%@
%%%@@@@@@%%%%%%%%@@%%%@@@@@%%@@%@%:.. -%%@#:...=@@@@@@@#.....-#@@@@%%####%@@%@%%%%%%%%%%%%%%%%%%%%%%
%#**######%%%%%###%%#*#%%%%@%%%%@#:.. -%#%*:...=@%*#@@@@#:......+@@%#####%@@%@%%%%%%%%%%%%%%%%%%%%%@
+++++++****##*++++++++++*##%%%##%#:...-#*#*:...=@+***#@@@@%+.....:#@%####%%@@%%%%%%%%%%%%%%%%%%%%%%%
+======++++**++========++*######%#:...-#*#*:...=@+===+%#:::-**....*@%####%%@@%%%%%%%%%%%%%%%%%%%%%%@
==-===+=+++++++=========+**###**##:...-#*#*:...=@+=-=-##...:*#-...+%%####%%%%%%%%%%%%%%%%%%%%%%%%%%@
=====+++++++++++++++===++++++++=*#:...-##%*:...=@***++#%-...::....*@%####%%@@%%%%%%%%%%%%%%%%%%%%%%%
=====+++++++++++++++==+===+=+=++*%-:::=%*##-:::=@**+++*@%*-....:+#@%####%%%@@%%%%%%%%%%%%%%%%%%%%%%%
====++++++++++++*+++=======+====+#######+*##%%%#%###**+**#%@@@@@%%***#%%%%%%@%%%%%%%%%%%%%%%%%%%%%%%
*+++++++********++++++++*++++++=+++++++**###**###*#***++++++****###**%%%%@@@%%%%%%%%%%%%%%%%%%%%%%%#
***********####**+++++***#*++++=+*##***########***+++++++++++****#%#**%@@@@@@%%%%%%%%%%%%%%%%%%%%%##
#####*******####******####+=+*+++##*++*****+**++++=====+++++++****###**%%%@@%@%%%%%%%%%%%%%%%%%%##**
*******+++****#*****#####*=+***+*#*++++==================+++++*****##***#%@@%@%%%%%%%%%%%%%%%%%%###*
*************##########%#++*###*+*++++============-=========+++*##*#****#%%@@@%%%%%%%%%%%%%%%%%%%###
######*****###########%%%#**#%%#*++++=======----==============+*#####***#%%@@%%%%%##***#%%%%%%%%%%%%
%%%%%%%%#%%%%%%%%%%%%%%%%%%######*++============----=========++++*###***%%@@%##*++=====+*#%%%%%%%%%%
@@@%%%%%%%%%%%%%%%%%%%%%%%%%#####+++==============-=======++++**#*+######%%%#*+==----===+*%%@@@@@@%%
@@@@%%%%%%%%%%%%%%%%%%%%%%%%*###*++++=====------=============+++*####*+*%%%%#*+=-----==++*#%@@@@@@@@
@@@%%%%%%%%%%%%%%%%%%%%%%%%#*###**+==+======-=======+**++++====++*#*++*=#%@@%#++===++*****#%@@@@@@@@
@@@@@%%%%%%%%%%%%%%%%%%%%%%%###*#*++=====+++=====++*####**+++====+**+=++#%@@@%##*******#%%%%@@@@@@@@
@@@@@@@%%%%%%%%%%%%%%%%%%%%%%#####+=+++**####*=--=++***#*+++=====++*+==+*%@@@@@%%###**#%%%%%%%%@%%%%
@@@@@@@@%%%%%%%%%%%@@@%%%%%%%%#*##*+++*#%#***+========++==========++++==*%@@@@%%##*##%%%@@%%%%%%%%%%
@@@@@@@@%%%%%%%%%@@@@@@@@@%%%%#***+++**+++++==============---======++===*%%@%%%####%%%%@@%%%%%%%%%%%
@@@@@@@@%%%%%%%%@@@@@@@@@@@%%%%#**+=================+++=---=======++=-==#%%@%%%%%%%%%@@%%%%%%%%%%%%%
@@@@@@@@@@@@%%%%@@@%@@@%@@@@%%%%**++=======++==--=====+++=========++==-:::--====+#%%%%%%%%%%%%%%%%%%
@@@@@@@@@@@@@@@@@%%%#####%%@@@@%#++++=====++++++++++++==+*+=======+++=-:::::::::....::--===*####%%%%
@@@@@@@@@@@@@@@%%%##*****##%%%@@%++++++=++********+++===+++++=====+++=-::::.....::::::::::-=**++*%%%
%%%%%%%%%%%%%%%%##**++++***##%%%%*++++++++**+++++=+===++++++++====++==-:::.......:..:::::::-+*+++#%%
%%%%%%%%%%%%%###***++++++****###%%#**+++++**+++******##%%%*+++==++++==::..........::.::::::-=++===*%
%%%%%%%%%%%###****++++++++++****##%%%#+++++**%@%%%##%##***+++++==+++==-:............::.::::::-=+===+
%%%%%%%%%%###*****+**+++**++*****##%%%*+++=++*#******+++*+=++==++++=++-:.............:::..:::::-===+
%%%%%%%%%###**********************###=:-++==++++++++++========+++++++=:..............::::......::---
%%%%%%%%%##**********#######******=.::::-+++==++====++=++=====+++++++=:...............:::::.........
@%%%%%%%###*********####%%%###*+:..::::::-++===++++++========+++*+++=:.................:::::.:::::..
@%%%%%%%###********#####%%%%%+...:::::--:::=+==++=+========+++*+++++::..................:::::..:::::
%%%%##%%##******###**#####%=...:::::::--::::-+++=++==++++++***+++++=:::.................::::::...:::
#####%%%%################=...:::::::::---::::-+*+++++++*******+++++=:...................:::::::..:::
*####%%%%%#############=:..:::::::::::---:::::-=+**#******+***++++=-::....................::::::...:
+*##%%%%%%%%%%####%##+:..:::::::---:::----:::::--=*#*********+++=-::...:...................::::::...
*###%%%%%%%%%%%%%%%#-......::::::--:::----:::::::--+##*******+=-::....:.....................::::::..
*####%%%%%%%%%%%%%*:......:::::::---:::----::::::---======-----:.....................::.......::::::
*#####%%%%%%%%%%%%=......::::::::--:--:----:::.::::------::::::......................:::.......:::-:
*######%%@%%%%%%##=:...::::::::::::::-::---::::.:::::-----:...........................::::.......:::
*####%%%%@@%%%%##*-::..::::::::::::::--:---:::::..::::::::::.......:...................::::.......::
*####%%%%@@%%%%##*-::::.:::::::::::::--::--:::::-::::::::..........::::................:::::........
######%%%%@@%%%##*-:::::::::::::::::---::---:-::::---:::............::::::.............:::::::......
#######%%%@@%%%##*-::-:::.::::::::::---::------:::::--::.............:::::::::.........::::::::::::.
##@%%%%%%@@%%@@%%%%+#%@%%@%*#####%#############%%####+%%%%@%#=..:*#######*-+######%%##############%=
##%.....:=:..+=...#@=......+%+..:*+..=+:..+*:..*@...%%-.....-%*.-*-......=@@*....:##...*#:..=%-..:@-
##%...-*#*:..++...#+...#+...#=..:=..:#*:..+*:...@...%+..:+:..=*:-*-..-*:..*@-.....=%-..+*...-#-..=@:
##%...=@@*...++...#=...#+...*=...:..+@*:..+*:...+...%=..:+:..=+:-*-..-*:..*@...-..-%=..++...:+:..+#:
*#%.....-*:..++...#=...%@%%%%=.....:*@*:..+*:.......%=..:+---+#==*-......*@#..:#..:#+..--....=:..#*-
==%...-#%*...++...#=...%*---#=......=@*:..+*:.:... .%=..:+:..=*==*-..-*:..#+..:%..:**.....-:.:...%=-
=+%...=%+#:..+=...#+...%+...#=..:=..:**:..+*:.:+....%=..:*=..=*-=*-..-#:..*-.......=#:....*=....-@=-
++%...=%+%=..::..-%#:..-:..-@=..:+-..=+:..+*:.:%:...%#...:...=*--*-..-#:..*...-%-..-#-...-#+....+#--
++@+++#%-*%#+===*%#%%*===+#@@#++*%#++***++#%*+*@#+++@@%+==**+#*:-**++*%*++#+++#@#++*%#+++*@%*+++#*--
*+++++++==++****=-:-=======--==--==+++====-------:---:-==----::..:::-------------------:::--------:-
********###*###*=::::----:::--------====---:::--::::::..::::::::::....::::::::::::::::::..:::--:::::
"""
