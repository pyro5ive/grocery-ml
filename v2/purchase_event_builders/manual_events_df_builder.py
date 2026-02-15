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

    expectedColumns: list[str] = [
        PurchaseEvent.SOURCE,
        PurchaseEvent.VENDOR,
        PurchaseEvent.DATE,
        PurchaseEvent.ITEM,
        PurchaseEvent.QTY
    ]

    csvPath: str
    logger: logging.Logger

    #======================================================#
    def __init__(self, csvPath: str):
        """
        :param csvPath: Path to the manually maintained events CSV file.
        :type csvPath: str
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.csvPath = csvPath
        self.logger.info("ManualEntryEventsDfBuilder initialized csvPath=%s", self.csvPath)

    #======================================================#
    def build_df(self) -> pd.DataFrame:
        """
        Build the manual entry events DataFrame from the configured CSV file.

        :returns: DataFrame of manual entry purchase events.
        :rtype: pd.DataFrame
        :raises ValueError: If the CSV file is missing expected columns.
        """
        self.logger.info("build_df(): start csvPath=%s", self.csvPath)

        df: pd.DataFrame = pd.read_csv(self.csvPath)

        self._validate_columns(df)

        df[PurchaseEvent.DATE] = pd.to_datetime(df[PurchaseEvent.DATE])
        df[PurchaseEvent.QTY] = pd.to_numeric(df[PurchaseEvent.QTY])

        self.logger.info("build_df(): done rows=%s", len(df))
        return df

    #======================================================#
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that all expected columns are present in the DataFrame.

        :param df: DataFrame loaded from the CSV file.
        :type df: pd.DataFrame
        :raises ValueError: If any expected columns are missing.
        """
        missing: list[str] = [c for c in self.expectedColumns if c not in df.columns]
        if missing:
            raise ValueError(f"{self.__class__.__name__} missing expected columns: {missing}")