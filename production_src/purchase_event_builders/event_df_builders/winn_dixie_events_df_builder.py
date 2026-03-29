import logging
import pandas as pd
from pathlib import Path
from abstractions.event_df_builder_base import EventDfBuilderBase
from abstractions.purchase_event_mapper_base import PurchaseEventMapperBase
from .winn_dixie_recpt_parser import WinnDixieRecptParser
# from purchase_event_builders.event_df_builders.mappers.winn_dixie_events_df_mapper import  WinnDixieReceiptToPurchaseEventMapper


#======================================================#
class WinnDixieEventsDfBuilder(EventDfBuilderBase):

    vendorName: str = "winndixie"
    primarySourceKey: str = "winndixie"
    additionalSourceKey: str = "winndixieAdditional"

    logger: logging.Logger
    recptParser: WinnDixieRecptParser
    mapper: PurchaseEventMapperBase

    #======================================================#
    def __init__(
        self,
        recptParser: WinnDixieRecptParser,
        mapper: PurchaseEventMapperBase
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.recptParser = recptParser
        self.mapper = mapper
        self.logger.info("WinnDixieEventsDfBuilder initialized")

    #======================================================#
    def build_df(self, sourcePaths: dict) -> pd.DataFrame:
        self.logger.info("build_df(): start")

        winndixiePath: str = sourcePaths[self.primarySourceKey]
        winndixieAdditionalPath: str = sourcePaths[self.additionalSourceKey]

        winndixieDf = self._build_winn_dixie_df(winndixiePath)
        winndixieAdditionalDf = self._build_winn_dixie_additional_text_rcpts_df(
            winndixieAdditionalPath
        )

        dfs: list[pd.DataFrame] = []

        if not winndixieDf.empty:
            dfs.append(winndixieDf)

        if not winndixieAdditionalDf.empty:
            dfs.append(winndixieAdditionalDf)

        if len(dfs) == 0:
            self.logger.warning("build_df(): no data found")
            return pd.DataFrame()

        wireDf: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        domainDf: pd.DataFrame = self.mapper.to_domain_model(wireDf)

        self.logger.info(
            "build_df(): done rows=%s cols=%s",
            len(domainDf),
            len(domainDf.columns)
        )

        return domainDf

    ###########################################
    def _build_winn_dixie_additional_text_rcpts_df(self, folderPath):

        rows = []
        self.logger.info("_build_winn_dixie_additional_text_rcpts_df folderPath: %s", folderPath)

        if folderPath is None:
            self.logger.warning("No folderPath provided for winndixieAdditional")
            return pd.DataFrame()

        for p in Path(folderPath).glob("*.txt"):
            self.logger.debug("Parsing additional receipt file: %s", p)

            result = self.recptParser.parse(p.read_text(encoding="utf-8", errors="ignore"))

            for r in result["items"]:
                rows.append({
                    "vendor": self.vendorName,
                    "source": p.name,
                    "date": result["date"],
                    "time": result["time"],
                    "item": r["item"],
                    "qty": r["qty"],
                })

        winndixie_df = pd.DataFrame(rows)

        if winndixie_df.empty:
            self.logger.warning("WinnDixieEventsDfBuilder produced empty dataframe. No receipts found Path: %s", folderPath);
            # raise Exception("WinnDixieEventsDfBuilder produced empty dataframe. No receipts found.")

        winndixie_df["date"] = pd.to_datetime(winndixie_df["date"])
        winndixie_df["time"] = winndixie_df["time"].astype(str)

        winndixie_df = WinnDixieRecptParser.remove_duplicate_receipt_files(winndixie_df)
        winndixie_df = winndixie_df.sort_values(by=["date", "time"]).reset_index(drop=True)
        winndixie_df = winndixie_df.drop(columns=["time"])

        self.logger.info("Additional receipts processed rows=%s", len(winndixie_df))

        return winndixie_df

    ###########################################################################################
    def _build_winn_dixie_df(self, path):

        self.logger.info("_build_winn_dixie_df path: %s", path)

        rows = []

        if path is None:
            self.logger.warning("No path provided for winndixie")
            return pd.DataFrame()

        for p in Path(path).glob("*.txt"):
            self.logger.debug("Parsing receipt file: %s", p)

            result = self.recptParser.parse(p.read_text(encoding="utf-8", errors="ignore"))

            for r in result["items"]:
                rows.append({
                    "vendor": self.vendorName,
                    "source": p.name,
                    "date": result["date"],
                    "time": result["time"],
                    "item": r["item"],
                    "qty": r["qty"],
                })

        winndixie_df = pd.DataFrame(rows)

        if winndixie_df.empty:
            self.logger.warning("WinnDixieEventsDfBuilder produced empty dataframe. No receipts found Path: %s",  path);
            return pd.DataFrame()
            # raise Exception("WinnDixieEventsDfBuilder produced empty dataframe. No receipts found.")

        winndixie_df["date"] = pd.to_datetime(winndixie_df["date"])
        winndixie_df["time"] = winndixie_df["time"].astype(str)

        winndixie_df = WinnDixieRecptParser.remove_duplicate_receipt_files(winndixie_df)
        winndixie_df = winndixie_df.sort_values(by=["date", "time"]).reset_index(drop=True)
        winndixie_df = winndixie_df.drop(columns=["time"])

        self.logger.info("Primary receipts processed rows=%s", len(winndixie_df))

        return winndixie_df


