import punq
import logging
from abstractions.services.item_id_index_service_base import ItemIndexBuilderServiceBase
import pandas as pd

class ItemIndexBuilderService(ItemIndexBuilderServiceBase):

    item_to_index: dict[str, int]
    index_to_item: dict[str, str]
    logger: logging.Logger
    itemIdColName: str
    itemNameColName: str
    indexIdColName: str

    def __init__(self,   existing_mapping: dict[str, int] | None = None ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.itemIdColName = "itemId";
        self.itemNameColName = "item";
        if existing_mapping is not None:
            self.item_to_index: dict[str, int] = dict(existing_mapping)
            self.index_to_item: dict[int, str] = {v: k for k, v in existing_mapping.items()}
            self.logger.info("initialized from existing mapping size=%s", len(self.item_to_index))
        else:
            self.item_to_index = {}
            self.index_to_item = {}
            self.logger.info("initialized empty")
    # ------------------------------------------------------------ #

    def build(self, series: pd.Series) -> None:
        self.logger.info("build(): start rows=%s", len(series))

        unique_items = series.dropna().unique()
        added_count = 0

        for item in unique_items:
            if item not in self.item_to_index:
                new_index = len(self.item_to_index)
                self.item_to_index[item] = new_index
                self.index_to_item[new_index] = item
                added_count += 1

        self.logger.info(
            "build(): done added=%s total_size=%s",
            added_count,
            len(self.item_to_index)
        )

    # ------------------------------------------------------------ #

    def to_index(self, series: pd.Series) -> pd.Series:
        self.logger.info("to_index(): start rows=%s", len(series))

        result = series.map(self.item_to_index)

        if result.isna().any():
            missing_items = series[result.isna()].unique().tolist()
            preview = missing_items[:10]
            self.logger.error(
                "to_index(): unknown items count=%s preview=%s",
                len(missing_items),
                preview
            )
            raise ValueError("Unknown items encountered during indexing")

        self.logger.info("to_index(): done")
        return result.astype(int)

    # ------------------------------------------------------------ #

    def to_item(self, series: pd.Series) -> pd.Series:
        self.logger.info("to_item(): start rows=%s", len(series))

        result = series.map(self.index_to_item)

        if result.isna().any():
            missing_indices = series[result.isna()].unique().tolist()
            preview = missing_indices[:10]
            self.logger.error(
                "to_item(): unknown indices count=%s preview=%s",
                len(missing_indices),
                preview
            )
            raise ValueError("Unknown indices encountered during reverse lookup")

        self.logger.info("to_item(): done")
        return result

    # ------------------------------------------------------------ #

    def contains(self, item: str) -> bool:
        return item in self.item_to_index

    # ------------------------------------------------------------ #

    def size(self) -> int:
        return len(self.item_to_index)

    # ------------------------------------------------------------ #

    def get_mapping(self) -> dict[str, int]:
        return dict(self.item_to_index)
