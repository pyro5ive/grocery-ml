import logging
import pandas as pd
import os
from abstractions.event_df_builder_base import EventDfBuilderBase
from purchase_event_builders.wallmart_rcpt_parser import WallmartRecptParser


#======================================================#
class WalMartEventsDfBuilder(EventDfBuilderBase):
    """
    Builds an events DataFrame from Walmart receipt CSV files.
    Parses all CSV files from the injected data source path.
    Cleans product descriptions and filters non-food items.
    """

    purchaseEventsDf: pd.DataFrame
    recptParser: WallmartRecptParser
    dataSourcePath: dict
    logger: logging.Logger

    #======================================================#
    def __init__(self, recptParser: WallmartRecptParser, dataSourcePath: dict):
        """
        :param recptParser: Injected Walmart receipt parser.
        :type recptParser: WallmartRecptParser
        :param dataSourcePath: Dictionary of named data source paths keyed by vendor name.
        :type dataSourcePath: dict
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.purchaseEventsDf = None
        self.recptParser = recptParser
        self.dataSourcePath = dataSourcePath
        self.logger.info("WalMartEventsDfBuilder initialized")

    #======================================================#
    def build_df(self) -> pd.DataFrame:
        """
        Build the Walmart events DataFrame by parsing all receipt CSV files
        from the walmart data source path.

        :returns: DataFrame of all parsed Walmart purchase events.
        :rtype: pd.DataFrame
        """
        self.logger.info("build_df(): start")

        purchaseEventsDf: pd.DataFrame = self._build_wall_mart_df(self.dataSourcePath.get("walmart"))

        self.logger.info("build_df(): done rows=%s", len(purchaseEventsDf))
        return purchaseEventsDf

    #======================================================#
    def _build_wall_mart_df(self, folderPath: str) -> pd.DataFrame:
        """
        Import all Walmart receipt CSV files from a folder.
        Cleans product descriptions and filters non-food items.
        Adds a source column set to the CSV filename.

        :param folderPath: Folder path containing Walmart receipt CSV files.
        :type folderPath: str
        :returns: DataFrame of parsed and cleaned Walmart purchase rows.
        :rtype: pd.DataFrame
        """
        self.logger.info("_build_wall_mart_df(): folderPath=%s", folderPath)

        if folderPath is None:
            self.logger.warning("_build_wall_mart_df(): no folderPath provided")
            return pd.DataFrame()

        dataframes: list[pd.DataFrame] = []

        for fileName in os.listdir(folderPath):
            if fileName.lower().endswith(".csv"):
                filePath: str = os.path.join(folderPath, fileName)
                dataframe: pd.DataFrame = pd.read_csv(filePath)
                dataframe["source"] = fileName
                dataframes.append(dataframe)

        if len(dataframes) == 0:
            self.logger.warning("_build_wall_mart_df(): no CSV files found folderPath=%s", folderPath)
            return pd.DataFrame()

        df: pd.DataFrame = pd.concat(dataframes, ignore_index=True)

        df["Product Description"] = (
            df["Product Description"]
            .str.replace("Great Value", "", regex=False)
            .str.replace("Freshness Guaranteed", "", regex=False)
            .str.strip()
        )

        df = df[
            ~df["Product Description"].str.contains("Mainstays", case=False, na=False)
            & ~df["Product Description"].str.contains("Sizes", case=False, na=False)
            & ~df["Product Description"].str.contains("Pen+Gear", case=False, na=False, regex=False)
            & ~df["Product Description"].str.contains("Athletic", case=False, na=False)
        ]

        df = df.rename(columns={
            "Order Date":          "date",
            "Product Description": "item",
            "Product Quantity":    "qty"
        })

        df["date"] = pd.to_datetime(df["date"])

        self.logger.info("_build_wall_mart_df(): done rows=%s", len(df))
        return df