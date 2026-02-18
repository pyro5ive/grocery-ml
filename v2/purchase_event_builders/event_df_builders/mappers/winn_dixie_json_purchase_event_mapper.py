import pandas as pd

from abstractions.purchase_event_mapper_base import PurchaseEventMapperBase
from models.purchase_event import PurchaseEvent


class WinnDixieJsonToPurchaseEventMapper(PurchaseEventMapperBase):

    vendor: str = "winndixie"
    source: str = "winndixie_app_json"

    from models.purchase_event import PurchaseEvent

    def to_domain_model(self, wireDf: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            PurchaseEvent.VENDOR: self.vendor,
            PurchaseEvent.SOURCE: self.source,
            PurchaseEvent.DATE: pd.to_datetime(wireDf["transactionDateTime"]).dt.date,
            PurchaseEvent.ITEM: wireDf["description"],
            PurchaseEvent.QTY: 1
        })

    #===============================================================================#
