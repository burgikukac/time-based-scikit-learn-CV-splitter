import polars as pl
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union, Generator

class CustomTimeSeriesSplit:
    def __init__(self, 
                 train_length: int, 
                 valid_length: int, 
                 step_size: int, 
                 n_splits: int, 
                 df_date_col_name: Optional[str] = None, 
                 np_date_col_number: Optional[int] = None):
        """
        Initializes the custom time series splitter with the required parameters.

        In the examples I would use 'days' as the time unit, but you can use any time unit you want.
        Args:
            train_length (int): 'Time' value range for the training set = [train_end - train_start + 1] IN DAYS.
            valid_length (int): The same as above, but for the validation set.
            step_size (int): The number of 'time' values to move the sliding window forward after each split.
            n_splits (int): The total number of splits to generate.
            df_date_col_name (Optional[str]): The name of the date column in the dataframe.
            np_date_col_number (Optional[int]): Optional numpy column index for date column if df_date_col_name is not provided.
        """
        self.train_length = train_length
        self.valid_length = valid_length
        self.step_size = step_size
        self.n_splits = n_splits
        self.df_date_col_name = df_date_col_name
        self.np_date_col_number = np_date_col_number

    def get_n_splits(self) -> int:
        """
        Returns the number of splits.

        Returns:
            int: The number of splits.
        """
        return self.n_splits

    def split(self, data: Union[pl.DataFrame, pd.DataFrame, np.ndarray]) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates indices for training and validation splits based on the values of the `date_col_name` 
        (for dataframes) or the `np_date_col_number` (for NumPy arrays).

        Args:
            data (Union[pl.DataFrame, pd.DataFrame, np.ndarray]): The dataframe (Polars or Pandas) or NumPy array containing the data.

        Yields:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the train and validation indices for each split.
        """
        # Handle dataframe inputs
        if isinstance(data, (pl.DataFrame, pd.DataFrame)):
            # Extract values from dataframe using column name
            if self.df_date_col_name is None:
                raise ValueError("For dataframe input, 'df_date_col_name' must be provided.")
            if isinstance(data, pl.DataFrame):
                values = data[self.df_date_col_name].to_numpy()
            elif isinstance(data, pd.DataFrame):
                values = data[self.df_date_col_name].values
            else:
                raise TypeError(f"Unsupported dataframe type. Please provide a Polars or Pandas dataframe. Got: {type(data)}")
        
        # Handle NumPy array inputs
        elif isinstance(data, np.ndarray):
            # Use the np_date_col_number to extract the relevant column
            if self.np_date_col_number is None:
                raise ValueError("For NumPy array input, 'np_date_col_number' must be provided.")
            values = data[:, self.np_date_col_number]
        
        else:
            raise TypeError(f"Unsupported data type. Please provide a Polars, Pandas dataframe, or NumPy array. Got: {type(data)}")

        max_value = values.max()
        min_value = values.min()

        # Yielding folds based on continuous values in the date column
        for fold in range(self.n_splits):
            valid_end = max_value - fold * self.step_size
            valid_start = valid_end - self.valid_length + 1
            train_end = valid_start - 1
            train_start = train_end - self.train_length + 1

            # Stop if we don't have enough data left for a valid train set
            if train_start < min_value:
                raise ValueError("Not enough data left for a valid train set.")

            # Get indices for train and validation sets based on the continuous date column
            train_idx = np.where((values >= train_start) & (values <= train_end))[0]
            valid_idx = np.where((values >= valid_start) & (values <= valid_end))[0]

            yield train_idx, valid_idx

# Example usage (Polars DataFrame):
# df = pl.DataFrame({"time_or_index_column": [1, 2, 3, ..., N]})
# custom_split = CustomTimeSeriesSplit(train_length=100, valid_length=50, step_size=25, n_splits=5, df_date_col_name="time_or_index_column")
# print("Number of splits:", custom_split.get_n_splits())
# for train_idx, valid_idx in custom_split.split(df):
#     print(train_idx, valid_idx)

# Example usage (Pandas DataFrame):
# df = pd.DataFrame({"time_or_index_column": [1, 2, 3, ..., N]})
# for train_idx, valid_idx in custom_split.split(df):
#     print(train_idx, valid_idx)

# Example usage (NumPy array):
# np_array = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
# custom_split = CustomTimeSeriesSplit(train_length=2, valid_length=1, step_size=1, n_splits=3, np_date_col_number=0)
# for train_idx, valid_idx in custom_split.split(np_array):
#     print(train_idx, valid_idx)
