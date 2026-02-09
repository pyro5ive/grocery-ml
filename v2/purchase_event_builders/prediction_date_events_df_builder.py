import logging
import pandas as pd
from typing import List
from datetime import datetime


class PredictionDateEventsDfBuilder:

    COL_DATE = "date"
    COL_SOURCE = "source"
    COL_ITEM = "item"
    SOURCE_VALUE = "_prediction_date_df_builder_"
    VENDOR = "prediction_date_df_builder_"

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
    #============================================================================#

    def build_df(self, prediction_date: datetime, item_list: List[str]) -> pd.DataFrame:

        self.logger.info("Building prediction df | date=%s | items=%s", prediction_date, len(item_list))
        df = pd.DataFrame({
            self.COL_DATE: [prediction_date] * len(item_list),
            self.COL_SOURCE: [self.SOURCE_VALUE] * len(item_list),
            self.COL_ITEM: item_list
        })

        self.logger.info("Prediction df built | rows=%s", len(df))

        return df
    #============================================================================#