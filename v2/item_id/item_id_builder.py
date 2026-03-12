import pandas as pd
from abstractions.item_id_builder_base import ItemIdBuilderBase
from abstractions.services.item_id_index_service_base import ItemIndexBuilderServiceBase


import logging

class ItemIdBuilder(ItemIdBuilderBase):

    def __init__(self, indexService: ItemIndexBuilderServiceBase):
        self.indexService = indexService
        self.itemNameColName = "item"
        self.itemIdColName = "itemId"

        self.logger = logging.getLogger(self.__class__.__name__)


    # ------------------------------------------------------------ #
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        inputCol = self.get_input_column_name()
        outputCol = self.get_output_column_name()

        if inputCol not in df.columns:
            raise ValueError(f"missing column '{inputCol}'")

        series = df[inputCol]

        if self.is_index_empty():
            self.build_new_ids(series)

        df[outputCol] = self.map_to_ids(series)

        df = self._reorder_item_columns(df)

        return df

    # ------------------------------------------------------------ #
    def _reorder_item_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce column order:
        index 3 -> item
        index 4 -> itemId
        """
        if self.itemNameColName not in df.columns:
            raise ValueError(f"missing column '{self.itemNameColName}'")

        if self.itemIdColName not in df.columns:
            raise ValueError(f"missing column '{self.itemIdColName}'")

        cols = list(df.columns)

        cols.remove(self.itemNameColName)
        cols.remove(self.itemIdColName)

        cols.insert(3, self.itemNameColName)
        cols.insert(4, self.itemIdColName)

        return df.reindex(columns=cols)

    # ------------------------------------------------------------ #
    def is_index_empty(self) -> bool:
        return self.indexService.is_empty()

    # ------------------------------------------------------------ #
    def build_new_ids(self, series: pd.Series) -> None:
        self.indexService.build(series)

    # ------------------------------------------------------------ #
    def map_to_ids(self, series: pd.Series) -> pd.Series:
        return self.indexService.to_index(series)

    # ------------------------------------------------------------ #
    def get_input_column_name(self) -> str:
        return self.itemNameColName

    # ------------------------------------------------------------ #
    def get_output_column_name(self) -> str:
        return self.itemIdColName