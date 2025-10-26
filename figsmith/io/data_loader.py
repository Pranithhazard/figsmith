"""
Data loader for .dat files
Converts data to numpy arrays or pandas DataFrames
"""

import numpy as np
import pandas as pd
from pathlib import Path


class DataLoader:
    """Load and manage data from .dat files"""

    def __init__(self, filepath=None):
        """
        Initialize data loader.

        Parameters
        ----------
        filepath : str, optional
            Path to .dat file to load
        """
        self.filepath = filepath
        self.data = None
        self.columns = []
        self.df = None

        if filepath:
            self.load(filepath)

    def _set_dataframe(self, df):
        """
        Centralized helper to store a DataFrame on the loader and sync metadata.
        """
        if df is None:
            raise ValueError("DataFrame cannot be None")

        # Always work with a shallow copy and string column names
        working = df.copy()
        working.columns = [str(col) for col in working.columns]

        self.df = working
        self.columns = list(working.columns)
        self.data = working.to_numpy()
        self.filepath = None
        return self

    def load(self, filepath, delimiter=None, header='infer', **kwargs):
        """
        Load data from .dat file.

        Parameters
        ----------
        filepath : str
            Path to .dat file
        delimiter : str, optional
            Column delimiter (auto-detected if None)
        header : int or 'infer', optional
            Row number to use as column names
        **kwargs : dict
            Additional arguments passed to pd.read_csv

        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        self.filepath = Path(filepath)

        # Try to auto-detect delimiter
        if delimiter is None:
            with open(filepath, 'r') as f:
                first_line = f.readline()
                # Check for comma-separated values
                if ',' in first_line and '\t' not in first_line:
                    delimiter = ','
                else:
                    # Use flexible whitespace delimiter (handles tabs, spaces, mixed)
                    # This works for both tab-separated and space-separated files
                    delimiter = r'\s+'

        try:
            # Load with pandas using sep parameter (delimiter is deprecated)
            self.df = pd.read_csv(
                filepath,
                sep=delimiter,
                header=header,
                engine='python' if delimiter == r'\s+' else 'c',
                **kwargs
            )

            # Store column names
            self.columns = list(self.df.columns)

            # Also store as numpy array
            self.data = self.df.to_numpy()

            return self.df

        except Exception as e:
            raise ValueError(f"Failed to load {filepath}: {e}")

    def get_column(self, column_name):
        """
        Get data from a specific column.

        Parameters
        ----------
        column_name : str or int
            Column name or index

        Returns
        -------
        np.ndarray
            Column data
        """
        if self.df is None:
            raise ValueError("No data loaded")

        if isinstance(column_name, int):
            return self.df.iloc[:, column_name].values
        else:
            return self.df[column_name].values

    def get_columns(self, column_names):
        """
        Get data from multiple columns.

        Parameters
        ----------
        column_names : list
            List of column names or indices

        Returns
        -------
        np.ndarray
            Array with shape (n_rows, n_columns)
        """
        if self.df is None:
            raise ValueError("No data loaded")

        return np.column_stack([self.get_column(col) for col in column_names])

    def save(self, filepath, delimiter='\t', header=True):
        """
        Save current data to .dat file.

        Parameters
        ----------
        filepath : str
            Output file path
        delimiter : str, optional
            Column delimiter
        header : bool, optional
            Include column names
        """
        if self.df is None:
            raise ValueError("No data to save")

        self.df.to_csv(
            filepath,
            sep=delimiter,
            index=False,
            header=header
        )

    @staticmethod
    def create_sample_data(n_points=100):
        """
        Create sample data for testing.

        Parameters
        ----------
        n_points : int
            Number of data points

        Returns
        -------
        pd.DataFrame
            Sample data
        """
        x = np.linspace(0, 2*np.pi, n_points)

        data = {
            'x': x,
            'sin': np.sin(x),
            'cos': np.cos(x),
            'tan': np.tan(x),
            'tanh': np.tanh(x)
        }

        return pd.DataFrame(data)

    # ===== In-memory ingestion helpers =====
    @classmethod
    def from_dataframe(cls, df):
        """
        Build a DataLoader from an existing pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Source data (will be shallow-copied)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        loader = cls()
        return loader._set_dataframe(df)

    @classmethod
    def from_numpy(cls, arr, columns=None):
        """
        Build a DataLoader from a NumPy array.

        Parameters
        ----------
        arr : np.ndarray or array-like
            Input array (1D or 2D). 1D arrays are treated as a single column.
        columns : list[str], optional
            Column names to assign. If omitted, generates col_0, col_1, ...
        """
        if arr is None:
            raise ValueError("arr cannot be None")

        np_arr = np.asarray(arr)
        if np_arr.ndim == 1:
            np_arr = np_arr.reshape(-1, 1)
        elif np_arr.ndim != 2:
            raise ValueError("NumPy data must be 1D or 2D")

        n_cols = np_arr.shape[1]

        if columns is not None:
            if len(columns) != n_cols:
                raise ValueError("Length of columns must match array width")
            col_names = [str(col) for col in columns]
        else:
            col_names = [f"col_{i}" for i in range(n_cols)]

        df = pd.DataFrame(np_arr, columns=col_names)
        loader = cls()
        return loader._set_dataframe(df)

    @classmethod
    def from_dict(cls, data_dict):
        """
        Build a DataLoader from a mapping of column names to array-like values.

        Parameters
        ----------
        data_dict : dict[str, array-like]
            Mapping of columns to equally sized vectors.
        """
        if not isinstance(data_dict, dict):
            raise TypeError("data_dict must be a dict of column -> data")
        if not data_dict:
            raise ValueError("data_dict cannot be empty")

        ordered_keys = sorted(data_dict.keys())

        lengths = set()
        normalized = {}
        for key in ordered_keys:
            value = np.asarray(data_dict[key])
            if value.ndim > 1:
                value = value.reshape(len(value), -1)
                if value.shape[1] != 1:
                    raise ValueError(f"Column '{key}' must be 1D")
                value = value[:, 0]
            normalized[key] = value
            lengths.add(len(value))

        if len(lengths) != 1:
            raise ValueError("All columns must have the same length")

        df = pd.DataFrame({str(k): normalized[k] for k in ordered_keys})
        loader = cls()
        return loader._set_dataframe(df)

    @staticmethod
    def create_field_data(nx=50, ny=50):
        """
        Create sample field data for contour plots.

        Parameters
        ----------
        nx, ny : int
            Grid resolution

        Returns
        -------
        pd.DataFrame
            Field data with x, y, and field values
        """
        x = np.linspace(-2, 2, nx)
        y = np.linspace(-2, 2, ny)
        X, Y = np.meshgrid(x, y)

        # Create some interesting fields
        temperature = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1*(X**2 + Y**2))
        velocity_magnitude = np.sqrt(X**2 + Y**2)
        vorticity = 2 * np.sin(X) * np.sin(Y)

        # Flatten for dataframe
        data = {
            'x': X.flatten(),
            'y': Y.flatten(),
            'temperature': temperature.flatten(),
            'velocity': velocity_magnitude.flatten(),
            'vorticity': vorticity.flatten()
        }

        return pd.DataFrame(data)
