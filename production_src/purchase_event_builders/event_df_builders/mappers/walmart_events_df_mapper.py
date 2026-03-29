import pandas as pd
from abstractions.purchase_event_mapper_base import PurchaseEventMapperBase
from models.purchase_event import PurchaseEvent


class WalMartReceiptToPurchaseEventMapper(PurchaseEventMapperBase):

    vendor: str = "walmart"

    #--------------------------#
    def to_domain_model(self, wireDf: pd.DataFrame) -> pd.DataFrame:
        """
        Map Walmart receipt wire DataFrame to PurchaseEvent domain shape.
        Assumes wireDf already contains per-file `source` values.
        """
        return pd.DataFrame({
            PurchaseEvent.VENDOR: self.vendor,
            PurchaseEvent.SOURCE: wireDf["source"],
            PurchaseEvent.DATE:   pd.to_datetime(wireDf["date"]).dt.date,
            PurchaseEvent.ITEM:   wireDf["item"],
            PurchaseEvent.QTY:    wireDf["qty"]
        })
    #--------------------------#
