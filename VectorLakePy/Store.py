import polars as pl
import os
from openai import OpenAI


class VectorLakePy:
    def __init__(self, api_key, model="text-embedding-3-small", dataset_name="mydata", base_path="data"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.base_path = base_path
        self.dataset_name = dataset_name


    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding


    def create_dataset(self, data, dataset_name=None):
        """
        Creates a new dataset (parquet file) from a list of dictionaries or a dictionary with text data.

        Args:
            data (list or dict): List of dictionaries or a dictionary with text data.
            dataset_name (str, optional): Name of the dataset (parquet file name). Defaults to None.
        """
        df = self.create_dataframe(data)
        if dataset_name is None:
            dataset_name = self.dataset_name
        self._write_parquet(df, dataset_name)


    def create_dataframe(self, data):
        """
        Creates a DataFrame with text data.

        Args:
            data (list or dict): List of dictionaries or a dictionary with text data.
            text_column (str, optional): Name of the column containing text data. Defaults to "text".

        Returns:
            pl.DataFrame: DataFrame with text data.
        """
        if isinstance(data, dict):
            data = [data]

        df = pl.DataFrame(data)
        return df


    def create_embeddings(self, data_columns, dataset_name=None):
        """
        Creates a new columns from the data columns list containing embeddings for text entries in a specific column. New columns are added to the DataFrame with prefix "embedding_".

        Args:
            data_columns (list): List of column names containing text data for which embeddings are generated.
            dataset_name (str, optional): Name of the dataset (parquet file name). Defaults to None.

        Returns:
            None
        """
        df = self.read_dataset(dataset_name)

        new_columns_list = [pl.lit(None).alias("embedding_" + column) for column in data_columns]

        df = df.with_columns(new_columns_list)

        for column in data_columns:
            df = df.with_columns(
                pl.col(column).map_elements(lambda x: self.get_embedding(x), return_dtype=pl.List(pl.Float64)).alias("embedding_" + column)
            )
        if dataset_name is None:
            dataset_name = self.dataset_name
        self._write_parquet(df, dataset_name)


    # https://stackoverflow.com/questions/77162729/writing-dataframes-as-partitioned-parquet-object-in-polars-with-pyarrow
    def _write_parquet(self, df, dataset_name):
        """
        Writes a DataFrame to a parquet file.

        Args:
            df (pl.DataFrame): DataFrame to be written.
            dataset_name (str): Name of the dataset (parquet file name).
        """
        df.write_parquet(os.path.join(self.base_path, dataset_name + ".parquet"))


    def read_dataset(self, dataset_name=None):
        """
        Reads a DataFrame from a Vector Data Lake dataset (parquet file).

        Args:
            name (str): Name of the dataset (parquet file name).

        Returns:
            pl.DataFrame: DataFrame read from the parquet file.
        """
        if dataset_name is None:
            dataset_name = self.dataset_name
        return pl.read_parquet(os.path.join(self.base_path, dataset_name + ".parquet"))


    # TODO: Remove id column dependency and refactor this with uuid or some other unique identifier
    def update_row(self, row_index, column_name, new_text):
        """
        Updates text and embedding for a specific row in the DataFrame.

        Args:
            row_index (int): Index of the row to update.
            column_name (str): Name of the column containing text data.
            new_text (str): New text for the row.

        Returns:
            pl.DataFrame: Updated DataFrame with new embedding.
        """
        df = self.read_dataset()

        df = df.with_columns(
            pl.when(pl.col("id") == row_index).then(pl.lit(new_text)).otherwise(pl.col(column_name)).alias(column_name),
            pl.when(pl.col("id") == row_index).then(self.get_embedding(new_text)).otherwise(pl.col("embedding_" + column_name)).alias("embedding_" + column_name)
        )

        self._write_parquet(df, self.dataset_name)


