import logging
import pandas as pd
from abstractions.df_filter_base import DfFilterBase


#======================================================#
class SameTripQtyCombiner(DfFilterBase):
    """
    Filters a DataFrame by summing quantities for duplicate date/itemId combinations.
    Preserves all other columns by keeping the first occurrence of each date/itemId pair.
    """

    dateCol: str = "date"
    itemIdCol: str = "itemId"
    qtyCol: str = "qty"

    logger: logging.Logger

    #======================================================#
    def __init__(self):
        """
        Initializes the filter.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("SameTripQtyCombiner initialized")

    #======================================================#
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sum quantities for duplicate date/itemId combinations and merge
        back with remaining columns.

        :param df: Input DataFrame containing date, itemId and qty columns.
        :type df: pd.DataFrame
        :returns: DataFrame with quantities summed per date/itemId combination.
        :rtype: pd.DataFrame
        """
        self.logger.info("filter(): start rows=%s", len(df))

        qtySummed: pd.DataFrame = df.groupby(
            [self.dateCol, self.itemIdCol], as_index=False
        )[self.qtyCol].sum()

        otherCols: pd.DataFrame = (
            df.drop(columns=[self.qtyCol])
              .drop_duplicates(subset=[self.dateCol, self.itemIdCol])
        )

        df = otherCols.merge(qtySummed, on=[self.dateCol, self.itemIdCol], how="left")

        self.logger.info("filter(): done rows=%s", len(df))
        return df