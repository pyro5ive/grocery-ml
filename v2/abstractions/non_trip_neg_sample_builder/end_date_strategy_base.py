import pandas as pd
from abc import ABC, abstractmethod

class EndDateStrategyBase(ABC):

    def resolve_end_date( self, df: pd.DataFrame, dateColName: str) -> pd.Timestamp:
        raise NotImplementedError()
    #==================================================#
