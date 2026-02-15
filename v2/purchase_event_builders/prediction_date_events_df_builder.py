import logging
import pandas as pd
from typing import List
from datetime import datetime
from abstractions.event_df_builder_base import EventDfBuilderBase


#======================================================#
class PredictionDateEventsDfBuilder(EventDfBuilderBase):
    """
    Builds a prediction date event DataFrame containing one row per item
    for the given prediction date.
    Used to construct the input context for prediction time inference.
    """

    COL_DATE: str = "date"
    COL_SOURCE: str = "source"
    COL_ITEM: str = "item"
    SOURCE_VALUE: str = "_prediction_date_df_builder_"
    VENDOR: str = "prediction_date_df_builder_"

    logger: logging.Logger

    #======================================================#
    def __init__(self) -> None:
        """
        Initializes the event DataFrame builder.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("PredictionDateEventsDfBuilder initialized")

    #======================================================#
    def build_df(self, prediction_date: datetime, item_list: List[str]) -> pd.DataFrame:
        """
        Build a prediction date event DataFrame with one row per item.

        :param prediction_date: The date for which the event DataFrame is being built.
        :type prediction_date: datetime
        :param item_list: List of item name strings to include in the DataFrame.
        :type item_list: List[str]
        :returns: DataFrame with one row per item for the prediction date.
        :rtype: pd.DataFrame
        """
        self.logger.info("build_df(): start date=%s items=%s", prediction_date, len(item_list))

        df: pd.DataFrame = pd.DataFrame({
            self.COL_DATE:   [prediction_date] * len(item_list),
            self.COL_SOURCE: [self.SOURCE_VALUE] * len(item_list),
            self.COL_ITEM:   item_list
        })

        self.logger.info("build_df(): done rows=%s", len(df))
        return df