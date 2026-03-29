from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict


class ItemIndexBuilderServiceBase(ABC):

    item_to_index: Dict[str, int]
    index_to_item: Dict[int, str]
    itemIdColName: str
    itemNameColName: str
    indexIdColName: str

    @abstractmethod
    def build(self, series: pd.Series) -> None:
        pass

    # ------------------------------------------------------------ #

    @abstractmethod
    def to_index(self, series: pd.Series) -> pd.Series:
        pass

    # ------------------------------------------------------------ #

    @abstractmethod
    def to_item(self, series: pd.Series) -> pd.Series:
        pass

    # ------------------------------------------------------------ #

    @abstractmethod
    def contains(self, item: str) -> bool:
        pass

    # ------------------------------------------------------------ #

    @abstractmethod
    def size(self) -> int:
        pass

    # ------------------------------------------------------------ #

    @abstractmethod
    def get_mapping(self) -> Dict[str, int]:
        pass
