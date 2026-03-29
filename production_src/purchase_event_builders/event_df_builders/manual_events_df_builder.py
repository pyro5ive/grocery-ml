import logging
import pandas as pd
from abstractions.event_df_builder_base import EventDfBuilderBase
from models.purchase_event import PurchaseEvent


#======================================================#
class ManualEntryEventsDfBuilder(EventDfBuilderBase):
    """
    Builds an events DataFrame from a manually maintained CSV file.
    Expects a fixed schema defined by the PurchaseEvent domain model.
    """

    sourceKey: str = "manual"

    expectedColumns: list[str] = [
        PurchaseEvent.SOURCE,
        PurchaseEvent.VENDOR,
        PurchaseEvent.DATE,
        PurchaseEvent.ITEM,
        PurchaseEvent.QTY
    ]

    logger: logging.Logger

    #======================================================#
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            "ManualEntryEventsDfBuilder initialized sourceKey=%s",
            self.sourceKey
        )

    #======================================================#
    def build_df(self, sourcePaths: dict) -> pd.DataFrame:
        if self.sourceKey not in sourcePaths:
            raise KeyError(
                f"sourcePaths missing required key '{self.sourceKey}'"
            )

        csvPath: str = sourcePaths[self.sourceKey]
        self.logger.info("build_df(): start csvPath=%s", csvPath)

        df: pd.DataFrame = pd.read_csv(csvPath)

        self._validate_columns(df)

        df[PurchaseEvent.DATE] = pd.to_datetime(df[PurchaseEvent.DATE])
        df[PurchaseEvent.QTY] = pd.to_numeric(df[PurchaseEvent.QTY])

        self.logger.info("build_df(): done rows=%s", len(df))
        return df

    #======================================================#
    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing: list[str] = [
            c for c in self.expectedColumns if c not in df.columns
        ]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} missing expected columns: {missing}"
            )
