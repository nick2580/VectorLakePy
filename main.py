import os
from VectorLakePy import VectorLakePy
import polars as pl


# =============================================================================
# CREATE DATASET BLOCK
# =============================================================================
# store = VectorLakePy(
#     api_key=os.environ['OPENAI-KEY'],
#     base_path=r"C:\Users\Nikhil\Desktop\M.Tech Related Stuff\VectorLakePy\facebook"
# )

# sample_csv_file_location = r"C:\Users\Nikhil\Desktop\M.Tech Related Stuff\VectorLakePy\Test Data\facebook_reviews.csv"

# df = pl.read_csv(sample_csv_file_location)

# store.write_to_dataset(df, partition_col="appVersion")



# =============================================================================
# READ DATASET BLOCK (LOCAL)
# =============================================================================
store = VectorLakePy(
    api_key=os.environ['OPENAI-KEY'],
    base_path=r"C:\Users\Nikhil\Desktop\M.Tech Related Stuff\VectorLakePy\facebook"
)


store.partition_col = "appVersion"

df = store.read_dataset(
    pyarrow_options={"partitions": [("appVersion", "=", "450.0.0.42.110")]}
)


newdf = df.filter(pl.col("thumbsUpCount") > 10000)

print(newdf)

# embeddf = store.create_embeddings(newdf, ["content"], persist=True)

# print(embeddf)


