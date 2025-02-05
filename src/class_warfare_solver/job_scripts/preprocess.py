from dotenv import load_dotenv

from class_warfare_solver.preprocessing.datafactory import DataFactory
from class_warfare_solver.utils.methods import initialize_spark

load_dotenv()

spark = initialize_spark()

df = DataFactory(spark)

df.load_raw_tables()
df.search_for_copper()
df.find_gold()
df.split_and_save()

# df.class_df.show()
# df.character_df.show()
