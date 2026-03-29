import pandas as pd
from abc import ABC, abstractmethod


class ItemIdBuilderBase(ABC):

    @abstractmethod
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def is_index_empty(self) -> bool:
        pass

    @abstractmethod
    def build_new_ids(self, series: pd.Series) -> None:
        pass

    @abstractmethod
    def map_to_ids(self, series: pd.Series) -> pd.Series:
        pass

    @abstractmethod
    def get_input_column_name(self) -> str:
        pass

    @abstractmethod
    def get_output_column_name(self) -> str:
        pass