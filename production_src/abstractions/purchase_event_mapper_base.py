import pandas as pd
from abc import ABC, abstractmethod


class PurchaseEventMapperBase(ABC):

    @abstractmethod
    def to_domain_model(self, wireDf: pd.DataFrame) -> pd.DataFrame:
        """
        Map a wire/source DataFrame into the PurchaseEvent domain shape.
        Must return columns: vendor, source, date, item, qty
        """
        pass
