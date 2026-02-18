import pandas as pd

from abstractions.purchase_event_mapper_base import PurchaseEventMapperBase
from models.purchase_event import PurchaseEvent


class WinnDixieReceiptToPurchaseEventMapper(PurchaseEventMapperBase):

    vendor: str = "winndixie"

    def to_domain_model(self, wireDf: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            PurchaseEvent.VENDOR: self.vendor,
            PurchaseEvent.SOURCE: wireDf[PurchaseEvent.SOURCE],
            PurchaseEvent.DATE:   pd.to_datetime(wireDf[PurchaseEvent.DATE]),
            PurchaseEvent.ITEM:   wireDf[PurchaseEvent.ITEM],
            PurchaseEvent.QTY:    wireDf[PurchaseEvent.QTY]
        })
    #--------------------------#
