import unittest
import polars as pl
import pandas as pd
#import numpy as np
from splitter import CustomTimeSeriesSplit

# test_splitter.py

class TestCustomTimeSeriesSplit(unittest.TestCase):
    def setUp(self):
        self.train_length = 2
        self.valid_length = 1
        self.step_size = 1
        self.n_splits = 2
        self.df_date_col_name = "time"
        self.np_date_col_number = 2

        self.data_polars = pl.DataFrame({
            "a": range(10),
            "b": range(9, -1, -1),
            "time": [1, 2, 3, 4, 1, 2, 3, 1, 2, 1]
        })

        self.data_pandas = pd.DataFrame({
            "a": range(10),
            "b": range(9, -1, -1),
            "time": [1, 2, 3, 4, 1, 2, 3, 1, 2, 1]
        })

        self.data_numpy = self.data_pandas.to_numpy()


    def test_initialization(self):
        splitter = CustomTimeSeriesSplit(
            self.train_length, self.valid_length, self.step_size, self.n_splits, self.df_date_col_name, self.np_date_col_number
        )
        self.assertEqual(splitter.train_length, self.train_length)
        self.assertEqual(splitter.valid_length, self.valid_length)
        self.assertEqual(splitter.step_size, self.step_size)
        self.assertEqual(splitter.n_splits, self.n_splits)
        self.assertEqual(splitter.df_date_col_name, self.df_date_col_name)
        self.assertEqual(splitter.np_date_col_number, self.np_date_col_number)

    def test_split_polars(self):
        splitter = CustomTimeSeriesSplit(
            self.train_length, self.valid_length, self.step_size, self.n_splits, self.df_date_col_name
        )
        splits = list(splitter.split(self.data_polars))
        self.assertEqual(len(splits), self.n_splits)

    def test_split_pandas(self):
        splitter = CustomTimeSeriesSplit(
            self.train_length, self.valid_length, self.step_size, self.n_splits, self.df_date_col_name
        )
        splits = list(splitter.split(self.data_pandas))
        print(splits)
        self.assertEqual(len(splits), self.n_splits)

    def test_split_numpy(self):
        splitter = CustomTimeSeriesSplit(
            self.train_length, self.valid_length, self.step_size, self.n_splits, np_date_col_number=self.np_date_col_number
        )
        splits = list(splitter.split(self.data_numpy))
        self.assertEqual(len(splits), self.n_splits)

if __name__ == "__main__":
    unittest.main()