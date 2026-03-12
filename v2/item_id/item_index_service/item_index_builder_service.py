import pandas as pd
from abstractions.services.item_id_index_service_base import ItemIndexBuilderServiceBase

import logging

class ItemIndexBuilderService(ItemIndexBuilderServiceBase):

    def __init__(self):
        self.item_to_index: dict[str, int] = {}
        self.index_to_item: dict[int, str] = {}

        self.logger = logging.getLogger(self.__class__.__name__)
    # ------------------------------------------------------------ #
    def is_empty(self) -> bool:
        return len(self.item_to_index) == 0

    # ------------------------------------------------------------ #
    def size(self) -> int:
        return len(self.item_to_index)

    # ------------------------------------------------------------ #
    def contains(self, item: str) -> bool:
        return item in self.item_to_index

    # ------------------------------------------------------------ #
    def build(self, series: pd.Series) -> None:
        for item in series.dropna().unique():
            if item not in self.item_to_index:
                idx = len(self.item_to_index)
                self.item_to_index[item] = idx
                self.index_to_item[idx] = item

    # ------------------------------------------------------------ #
    def to_index(self, series: pd.Series) -> pd.Series:
        result = series.map(self.item_to_index)
        if result.isna().any():
            missing = series[result.isna()].unique().tolist()
            self.logger.warning(f"Unknown items: {missing[:10]}")
        return result.astype(int)

    # ------------------------------------------------------------ #
    def to_item(self, series: pd.Series) -> pd.Series:
        result = series.map(self.index_to_item)
        if result.isna().any():
            missing = series[result.isna()].unique().tolist()
            self.logger.warning(f"Unknown items: {missing[:10]}")
        return result

    # ------------------------------------------------------------ #
    def get_mapping(self) -> dict[str, int]:
        return dict(self.item_to_index)