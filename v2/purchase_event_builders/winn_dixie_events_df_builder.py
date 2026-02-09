import logging
import pandas as pd
from pathlib import Path

from .winn_dixie_recpt_parser import WinnDixieRecptParser

class WinnDixieEventsDfBuilder:

    vendorName = "winndixie"

    def __init__(this, dataSources):
        this.data_sources = dataSources
        this.logger = logging.getLogger(this.__class__.__name__)
        this.recptParser = WinnDixieRecptParser()

    ###########################################
    def build_df(this):
        this.logger.info("WinDixie Events DF Builder started")

        winndixiePath = this.data_sources.get("winndixie")
        winndixieAdditionalPath = this.data_sources.get("winndixieAdditional")

        this.logger.info("winndixie path param: %s", winndixiePath)
        this.logger.info("winndixieAdditional path param: %s", winndixieAdditionalPath)

        winndixieDf = this._build_winn_dixie_df(winndixiePath)
        winndixieAdditionalDf = this._build_winn_dixie_additional_text_rcpts_df(winndixieAdditionalPath)

        dfs: list[pd.DataFrame] = []

        if winndixieDf is not None and len(winndixieDf) > 0:
            dfs.append(winndixieDf)

        if winndixieAdditionalDf is not None and len(winndixieAdditionalDf) > 0:
            dfs.append(winndixieAdditionalDf)

        if len(dfs) == 0:
            winndixieDf = pd.DataFrame()
        else:
            winndixieDf = pd.concat(dfs, ignore_index=True)

        this.logger.info("WinDixie Events DF Builder finished rows=%s cols=%s",
                         len(winndixieDf), len(winndixieDf.columns))

        return winndixieDf

    ###########################################
    def _build_winn_dixie_additional_text_rcpts_df(this, folderPath):

        rows = []
        this.logger.info("_build_winn_dixie_additional_text_rcpts_df folderPath: %s", folderPath)

        if folderPath is None:
            this.logger.warning("No folderPath provided for winndixieAdditional")
            return pd.DataFrame()

        for p in Path(folderPath).glob("*.txt"):
            this.logger.debug("Parsing additional receipt file: %s", p)

            result = this.recptParser.parse(p.read_text(encoding="utf-8", errors="ignore"))

            for r in result["items"]:
                rows.append({
                    "vendor": this.vendorName,
                    "source": p.name,
                    "date": result["date"],
                    "time": result["time"],
                    "item": r["item"],
                    "qty": r["qty"],
                })

        winndixie_df = pd.DataFrame(rows)

        if winndixie_df.empty:
            this.logger.warning("WinnDixieEventsDfBuilder produced empty dataframe. No receipts found Path: %s", folderPath);
            # raise Exception("WinnDixieEventsDfBuilder produced empty dataframe. No receipts found.")

        winndixie_df["date"] = pd.to_datetime(winndixie_df["date"])
        winndixie_df["time"] = winndixie_df["time"].astype(str)

        winndixie_df = WinnDixieRecptParser.remove_duplicate_receipt_files(winndixie_df)
        winndixie_df = winndixie_df.sort_values(by=["date", "time"]).reset_index(drop=True)
        winndixie_df = winndixie_df.drop(columns=["time"])

        this.logger.info("Additional receipts processed rows=%s", len(winndixie_df))

        return winndixie_df

    ###########################################################################################
    def _build_winn_dixie_df(this, path):

        this.logger.info("_build_winn_dixie_df path: %s", path)

        rows = []

        if path is None:
            this.logger.warning("No path provided for winndixie")
            return pd.DataFrame()

        for p in Path(path).glob("*.txt"):
            this.logger.debug("Parsing receipt file: %s", p)

            result = this.recptParser.parse(p.read_text(encoding="utf-8", errors="ignore"))

            for r in result["items"]:
                rows.append({
                    "vendor": this.vendorName,
                    "source": p.name,
                    "date": result["date"],
                    "time": result["time"],
                    "item": r["item"],
                    "qty": r["qty"],
                })

        winndixie_df = pd.DataFrame(rows)

        if winndixie_df.empty:
            this.logger.warning("WinnDixieEventsDfBuilder produced empty dataframe. No receipts found Path: %s",  path);
            return pd.DataFrame()
            # raise Exception("WinnDixieEventsDfBuilder produced empty dataframe. No receipts found.")

        winndixie_df["date"] = pd.to_datetime(winndixie_df["date"])
        winndixie_df["time"] = winndixie_df["time"].astype(str)

        winndixie_df = WinnDixieRecptParser.remove_duplicate_receipt_files(winndixie_df)
        winndixie_df = winndixie_df.sort_values(by=["date", "time"]).reset_index(drop=True)
        winndixie_df = winndixie_df.drop(columns=["time"])

        this.logger.info("Primary receipts processed rows=%s", len(winndixie_df))

        return winndixie_df
