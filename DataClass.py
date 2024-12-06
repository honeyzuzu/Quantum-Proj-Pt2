import pandas as pd
import numpy as np
from qiskit_finance.data_providers import BaseDataProvider
from decimal import Decimal, getcontext


#Class to handle the data that will be used in the quantum algorithm

import pandas as pd
import numpy as np
from qiskit_finance.data_providers import BaseDataProvider
from decimal import Decimal, getcontext


class DataClass(BaseDataProvider):
    """
    DataClass is a child class of parent class BaseDataProvider from Qiskit Finance.
    Storing data in this form will be beneficial as it allows usage of further Qiskit Finance functions.
    """
    def __init__(self, start, end, file_path=None, data=None):
        self._file_path = file_path
        self._start = pd.to_datetime(start)
        self._end = pd.to_datetime(end)
        self._tickers = []


        if self._data.empty and self._file_path is None:
            raise ValueError("Either file_path or data must be provided")

    def load_data(self) -> pd.DataFrame:
        try:
            # Read data from the provided Excel file
            df = pd.read_excel(self._file_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            df.sort_values("Date")
            df.reset_index(drop=True)

            # Filter the DataFrame to only include the specified date range
            df = df[(df.index >= self._start) & (df.index <= self._end)]
            # Set the tickers to the column headers
            self._tickers = df.columns.tolist()
            return df
        except Exception as e:
            raise IOError(f"Error loading data from {self._file_path}: {e}")

    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        return np.log(1 + prices).dropna()

    def run(self) -> None:
        # Load data from the Excel file
        df = self.load_data()
        # Calculate log returns for all tickers
        log_returns = df.apply(self.calculate_log_returns, axis=0)
        # Drop rows with NaN values (e.g., first row due to shift)
        self._data = log_returns.dropna()
        self.get_mean_vector()
        self.get_covariance_matrix()
        self.get_stddev()
        self.get_correlation()
        self.get_volatility()
        self.get_sharpe_ratio()

    def get_mean_vector(self) -> np.ndarray:
        self._mean_vector = self._data.mean().values
        return self._mean_vector