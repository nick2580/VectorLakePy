import os
from VectorLakePy import VectorLakePy
import polars as pl


# =============================================================================
# CREATE DATASET BLOCK
# =============================================================================
store = VectorLakePy(
    api_key=os.environ['OPENAI-KEY'],
    base_path=r"C:\Users\Nikhil\Desktop\M.Tech Related Stuff\VectorLakePy\test_data"
)

sample_csv_file_location = r"C:\Users\Nikhil\Desktop\M.Tech Related Stuff\VectorLakePy\Test Data\customers-100000.csv"

df = pl.read_csv(sample_csv_file_location)

store.write_to_dataset(df, partition_col="Subscription Date")



# =============================================================================
# READ DATASET BLOCK (LOCAL)
# =============================================================================
store = VectorLakePy(
    api_key=os.environ['OPENAI-KEY'],
    base_path=r"C:\Users\Nikhil\Desktop\M.Tech Related Stuff\VectorLakePy\test_data"
)


df = store.read_dataset(
    pyarrow_options={"partitions": [("Subscription Date", "=", "2021-11-11")]}
)

print(df)