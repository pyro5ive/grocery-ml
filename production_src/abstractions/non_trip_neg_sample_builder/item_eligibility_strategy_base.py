import pandas as pd
from abc import ABC, abstractmethod

class ItemEligibilityStrategyBase(ABC):
    @abstractmethod

    def build_item_calendar(
        self,
        df: pd.DataFrame,
        itemIdColName: str,
        itemNameColName: str,
        dateColName: str,
        negStartDate: pd.Timestamp,
        negEndDate: pd.Timestamp
    ) -> pd.DataFrame:
        raise NotImplementedError()
#--------------------------#
