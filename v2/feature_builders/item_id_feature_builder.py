import logging
import pandas as pd
from abstractions.feature_builder_base import FeatureBuilderBase
from abstractions.services.item_id_index_service_base import ItemIndexBuilderServiceBase


#======================================================#
class ItemIdFeatureBuilder(FeatureBuilderBase):
    """
    Feature builder that derives an integer item ID column from an item name column.
    Delegates item-to-index mapping state to ItemIndexBuilderServiceBase.
    State is managed by the injected service, allowing it to survive the train/predict cycle.
    """

    itemNameColName: str
    itemIdColName: str
    indexBuilder: ItemIndexBuilderServiceBase
    logger: logging.Logger

    #======================================================#
    def __init__(self, indexBuilder, itemNameColName: str, itemIdColName: str):
        """
        :param indexBuilder: Injected index builder service that owns the item mapping state.
        :type indexBuilder: ItemIndexBuilderServiceBase
        :param itemNameColName: DataFrame column name containing item name strings.
        :type itemNameColName: str
        :param itemIdColName: DataFrame column name to write integer item IDs into.
        :type itemIdColName: str
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.indexBuilder = indexBuilder;
        self.itemNameColName = itemNameColName
        self.itemIdColName = itemIdColName
        self.logger.info(
            "ItemIdFeatureBuilder initialized itemNameColName=%s itemIdColName=%s",
            self.itemNameColName,
            self.itemIdColName
        )

    #======================================================#
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the item ID feature column from the item name column.
        Builds the index mapping if not yet initialized, otherwise maps existing.
        Drops rows containing unseen items with a warning.

        :param df: Input DataFrame containing the item name column.
        :type df: pd.DataFrame
        :returns: DataFrame with item ID column added.
        :rtype: pd.DataFrame
        :raises ValueError: If the item name column is missing from the DataFrame.
        :raises RuntimeError: If unexpected NaNs are detected after mapping.
        """
        self.logger.info("build(): start rows=%s", len(df))

        if self.itemNameColName not in df.columns:
            raise ValueError(f"build(): df missing required column '{self.itemNameColName}'")

        if self.indexBuilder.size() == 0:
            return self._build_item_ids(df)

        return self._map_existing_item_ids(df)

    #======================================================#
    def get_feature_names_in(self) -> list[str]:
        """
        Return the input column names this builder requires.

        :returns: List containing the item name column name.
        :rtype: list[str]
        """
        return [self.itemNameColName]

    #======================================================#
    def get_feature_names_out(self) -> list[str]:
        """
        Return the output column names this builder produces.

        :returns: List containing the item ID column name.
        :rtype: list[str]
        """
        return [self.itemIdColName]

    #======================================================#
    def _build_item_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a new index mapping from the item name column and apply it to the DataFrame.

        :param df: Input DataFrame containing the item name column.
        :type df: pd.DataFrame
        :returns: DataFrame with item ID column added.
        :rtype: pd.DataFrame
        :raises RuntimeError: If NaNs are detected after mapping.
        """
        self.logger.info("_build_item_ids(): start rows=%s", len(df))

        itemSeries: pd.Series = df[self.itemNameColName]
        self.indexBuilder.build(itemSeries)

        df[self.itemIdColName] = self.indexBuilder.to_index(itemSeries)

        if df[self.itemIdColName].isna().any():
            nan_count: int = int(df[self.itemIdColName].isna().sum())
            self.logger.error("_build_item_ids(): NaNs detected after mapping count=%s", nan_count)
            raise RuntimeError("itemId mapping produced NaNs during build")

        df.reset_index(drop=True, inplace=True)
        self.logger.info("_build_item_ids(): done mapping_size=%s", self.indexBuilder.size())
        return df

    #======================================================#
    def _map_existing_item_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map item name column to existing index mapping, dropping unseen items with a warning.

        :param df: Input DataFrame containing the item name column.
        :type df: pd.DataFrame
        :returns: DataFrame with item ID column added, unseen items dropped.
        :rtype: pd.DataFrame
        :raises RuntimeError: If unexpected NaNs are detected after mapping.
        """
        self.logger.info("_map_existing_item_ids(): start rows=%s", len(df))

        itemSeries: pd.Series = df[self.itemNameColName]
        unseen_mask: pd.Series = ~itemSeries.apply(self.indexBuilder.contains)
        dropped_count: int = int(unseen_mask.sum())

        if dropped_count > 0:
            unseen_items: list[str] = itemSeries[unseen_mask].dropna().unique().tolist()
            self.logger.warning(
                "_map_existing_item_ids(): dropping unseen items rows_dropped=%s unique_items=%s preview=%s",
                dropped_count, len(unseen_items), unseen_items[:10]
            )
            df = df.loc[~unseen_mask].copy()

        if len(df) == 0:
            self.logger.warning("_map_existing_item_ids(): all rows dropped due to unseen items")
            return df.reset_index(drop=True)

        df[self.itemIdColName] = self.indexBuilder.to_index(df[self.itemNameColName])

        if df[self.itemIdColName].isna().any():
            nan_count: int = int(df[self.itemIdColName].isna().sum())
            self.logger.error("_map_existing_item_ids(): unexpected NaNs after mapping count=%s", nan_count)
            raise RuntimeError("Unexpected NaNs after itemId mapping")

        df.reset_index(drop=True, inplace=True)
        self.logger.info("_map_existing_item_ids(): done rows=%s", len(df))
        return df