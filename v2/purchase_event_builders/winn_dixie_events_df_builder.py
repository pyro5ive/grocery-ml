import logging
import pandas as pd
from pathlib import Path
from abstractions.event_df_builder_base import EventDfBuilderBase
from .winn_dixie_recpt_parser import WinnDixieRecptParser


#======================================================#
class WinnDixieEventsDfBuilder(EventDfBuilderBase):
    """
    Builds an events DataFrame from WinnDixie receipt files.
    Parses both primary and additional text receipt files from injected data source paths.
    Combines results into a single normalized DataFrame sorted by date.
    """

    vendorName: str = "winndixie"

    dataSources: dict
    recptParser: WinnDixieRecptParser
    logger: logging.Logger

    #======================================================#
    def __init__(self, dataSources: dict):
        """
        :param dataSources: Dictionary of named data source paths keyed by vendor name.
        :type dataSources: dict
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataSources = dataSources
        self.recptParser = WinnDixieRecptParser()
        self.logger.info("WinnDixieEventsDfBuilder initialized")

    #======================================================#
    def build_df(self) -> pd.DataFrame:
        """
        Build the WinnDixie events DataFrame by parsing all receipt files
        from primary and additional data source paths.

        :returns: Combined DataFrame of all parsed receipt events sorted by date.
        :rtype: pd.DataFrame
        """
        self.logger.info("build_df(): start")

        winndixiePath: str = self.dataSources.get("winndixie")
        winndixieAdditionalPath: str = self.dataSources.get("winndixieAdditional")

        self.logger.info("build_df(): winndixiePath=%s", winndixiePath)
        self.logger.info("build_df(): winndixieAdditionalPath=%s", winndixieAdditionalPath)

        winndixieDf: pd.DataFrame = self._build_winn_dixie_df(winndixiePath)
        winndixieAdditionalDf: pd.DataFrame = self._build_winn_dixie_additional_text_rcpts_df(winndixieAdditionalPath)

        dfs: list[pd.DataFrame] = []

        if winndixieDf is not None and len(winndixieDf) > 0:
            dfs.append(winndixieDf)

        if winndixieAdditionalDf is not None and len(winndixieAdditionalDf) > 0:
            dfs.append(winndixieAdditionalDf)

        if len(dfs) == 0:
            self.logger.warning("build_df(): no data found returning empty DataFrame")
            return pd.DataFrame()

        resultDf: pd.DataFrame = pd.concat(dfs, ignore_index=True)

        self.logger.info("build_df(): done rows=%s cols=%s", len(resultDf), len(resultDf.columns))
        return resultDf

    #======================================================#
    def _build_winn_dixie_df(self, path: str) -> pd.DataFrame:
        """
        Parse primary WinnDixie receipt text files from the given folder path.

        :param path: Folder path containing primary receipt text files.
        :type path: str
        :returns: DataFrame of parsed receipt rows, or empty DataFrame if path is None or no files found.
        :rtype: pd.DataFrame
        """
        self.logger.info("_build_winn_dixie_df(): path=%s", path)

        if path is None:
            self.logger.warning("_build_winn_dixie_df(): no path provided")
            return pd.DataFrame()

        rows: list = []

        for p in Path(path).glob("*.txt"):
            self.logger.debug("_build_winn_dixie_df(): parsing file=%s", p)
            result: dict = self.recptParser.parse(p.read_text(encoding="utf-8", errors="ignore"))

            for r in result["items"]:
                rows.append({
                    "vendor": self.vendorName,
                    "source": p.name,
                    "date":   result["date"],
                    "time":   result["time"],
                    "item":   r["item"],
                    "qty":    r["qty"]
                })

        winndixieDf: pd.DataFrame = pd.DataFrame(rows)

        if winndixieDf.empty:
            self.logger.warning("_build_winn_dixie_df(): empty DataFrame path=%s", path)
            return pd.DataFrame()

        winndixieDf["date"] = pd.to_datetime(winndixieDf["date"])
        winndixieDf["time"] = winndixieDf["time"].astype(str)
        winndixieDf = WinnDixieRecptParser.remove_duplicate_receipt_files(winndixieDf)
        winndixieDf = winndixieDf.sort_values(by=["date", "time"]).reset_index(drop=True)
        winndixieDf = winndixieDf.drop(columns=["time"])

        self.logger.info("_build_winn_dixie_df(): done rows=%s", len(winndixieDf))
        return winndixieDf

    #======================================================#
    def _build_winn_dixie_additional_text_rcpts_df(self, folderPath: str) -> pd.DataFrame:
        """
        Parse additional WinnDixie receipt text files from the given folder path.

        :param folderPath: Folder path containing additional receipt text files.
        :type folderPath: str
        :returns: DataFrame of parsed receipt rows, or empty DataFrame if path is None or no files found.
        :rtype: pd.DataFrame
        """
        self.logger.info("_build_winn_dixie_additional_text_rcpts_df(): folderPath=%s", folderPath)

        if folderPath is None:
            self.logger.warning("_build_winn_dixie_additional_text_rcpts_df(): no folderPath provided")
            return pd.DataFrame()

        rows: list = []

        for p in Path(folderPath).glob("*.txt"):
            self.logger.debug("_build_winn_dixie_additional_text_rcpts_df(): parsing file=%s", p)
            result: dict = self.recptParser.parse(p.read_text(encoding="utf-8", errors="ignore"))

            for r in result["items"]:
                rows.append({
                    "vendor": self.vendorName,
                    "source": p.name,
                    "date":   result["date"],
                    "time":   result["time"],
                    "item":   r["item"],
                    "qty":    r["qty"]
                })

        winndixieDf: pd.DataFrame = pd.DataFrame(rows)

        if winndixieDf.empty:
            self.logger.warning("_build_winn_dixie_additional_text_rcpts_df(): empty DataFrame folderPath=%s", folderPath)
            return pd.DataFrame()

        winndixieDf["date"] = pd.to_datetime(winndixieDf["date"])
        winndixieDf["time"] = winndixieDf["time"].astype(str)
        winndixieDf = WinnDixieRecptParser.remove_duplicate_receipt_files(winndixieDf)
        winndixieDf = winndixieDf.sort_values(by=["date", "time"]).reset_index(drop=True)
        winndixieDf = winndixieDf.drop(columns=["time"])

        self.logger.info("_build_winn_dixie_additional_text_rcpts_df(): done rows=%s", len(winndixieDf))
        return winndixieDf