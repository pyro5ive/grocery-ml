import logging
import pandas as pd
import os

from abstractions.event_df_builder_base import EventDfBuilderBase
from abstractions.purchase_event_mapper_base import PurchaseEventMapperBase
from purchase_event_builders.event_df_builders.wallmart_rcpt_parser import WallmartRecptParser

#======================================================#
class WalMartEventsDfBuilder(EventDfBuilderBase):
    """
    Builds an events DataFrame from Walmart receipt CSV files.
    Parses all CSV files from the injected data source path.
    """

    sourceKey: str = "walmart"

    logger: logging.Logger
    recptParser: WallmartRecptParser
    mapper: PurchaseEventMapperBase

    #======================================================#
    def __init__(
        self,
        recptParser: WallmartRecptParser,
        mapper: PurchaseEventMapperBase
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.recptParser = recptParser
        self.mapper = mapper

        self.logger.info(
            "WalMartEventsDfBuilder initialized sourceKey=%s mapper=%s",
            self.sourceKey,
            mapper.__class__.__name__
        )

    #======================================================#
    def build_df(self, sourcePaths: dict) -> pd.DataFrame:
        self.logger.info("build_df(): start")

        if self.sourceKey not in sourcePaths:
            raise KeyError(
                f"sourcePaths missing required key '{self.sourceKey}'"
            )

        wireDf: pd.DataFrame = self._build_wall_mart_df(
            sourcePaths[self.sourceKey]
        )

        if wireDf.empty:
            return wireDf

        domainDf: pd.DataFrame = self.mapper.to_domain_model(wireDf)

        self.logger.info("build_df(): done rows=%s", len(domainDf))
        return domainDf

    #======================================================#
    def _build_wall_mart_df(self, folderPath: str) -> pd.DataFrame:
        self.logger.info("_build_wall_mart_df(): folderPath=%s", folderPath)

        if folderPath is None:
            self.logger.warning("_build_wall_mart_df(): no folderPath provided")
            return pd.DataFrame()

        dataframes: list[pd.DataFrame] = []

        for fileName in os.listdir(folderPath):
            if fileName.lower().endswith(".csv"):
                filePath: str = os.path.join(folderPath, fileName)
                df: pd.DataFrame = pd.read_csv(filePath)
                df["source"] = fileName
                dataframes.append(df)

        if not dataframes:
            self.logger.warning(
                "_build_wall_mart_df(): no CSV files found folderPath=%s",
                folderPath
            )
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
            & ~df["Product Description"].str.contains("Pen+Gear", case=False, na=False)
            & ~df["Product Description"].str.contains("Athletic", case=False, na=False)
        ]

        df = df.rename(columns={
            "Order Date":           "date",
            "Product Description":  "item",
            "Product Quantity":     "qty"
        })

        df["date"] = pd.to_datetime(df["date"])

        self.logger.info("_build_wall_mart_df(): done rows=%s", len(df))
        return df
