import logging
import pandas as pd
from pathlib import Path

from .winn_dixie_recpt_parser import WinnDixieRecptParser

class WinnDixieEventsDfBuilder:

    vendorName = "winndixie";
    
    def __init__(this, dataSources):
        this.data_sources = dataSources
        this.logger = logging.getLogger(this.__class__.__name__)
        this.recptParser = WinnDixieRecptParser()
    ###########################################
    def build_df(this):
        this.logger.info("WinDixie Events DF Builder started");
    
        winndixieDf = this._build_winn_dixie_df(this.data_sources.get("winndixie"))
        winndixieAdditionalDf = this._build_winn_dixie_additional_text_rcpts_df(this.data_sources.get("winndixieAdditional"))
        dfs: list[pd.DataFrame] = []
        
        if winndixieDf is not None and len(winndixieDf) > 0: dfs.append(winndixieDf);
        if winndixieAdditionalDf is not None and len(winndixieAdditionalDf) > 0: dfs.append(winndixieAdditionalDf)
        if len(dfs) == 0:
            winndixieDf = pd.DataFrame()
        else:
            winndixieDf = pd.concat(dfs, ignore_index=True)
        
        this.logger.info("WinDixie Events DF Builder finished");
        return winndixieDf;
    ###########################################
    def _build_winn_dixie_additional_text_rcpts_df(this, folderPath):

        rows = []
        for p in Path(folderPath).glob("*.txt"):
            result = this.recptParser.parse(p.read_text(encoding="utf-8", errors="ignore"))
            for r in result["items"]:
                rows.append({
                    "vendor": this.vendorName,
                    "source": p.name,
                    "date": result["date"],
                    "time": result["time"],
                    #"manager": result["manager"],
                    #"cashier_raw": result["cashier"],
                    "item": r["item"],
                    "qty": r["qty"],
                    #"reg": r["reg"],
                    #"youPay": r["youPay"],
                    #"reportedItemsSold": result["reported"],
                    #"rowsMatchReported": result["validation"]["rowsMatchReported"],
                    #"qtyMatchReported": result["validation"]["qtyMatchReported"],
                })
    
        additional_rcpts_df = pd.DataFrame(rows)
        if additional_rcpts_df.empty:
            raise Exception("WinnDixieEventsDfBuilder produced empty dataframe. No receipts found.")
        
        additional_rcpts_df["date"] = pd.to_datetime(additional_rcpts_df["date"])
        additional_rcpts_df["time"] = additional_rcpts_df["time"].astype(str)
        
        additional_rcpts_df = WinnDixieRecptParser.remove_duplicate_receipt_files(additional_rcpts_df)
        additional_rcpts_df = additional_rcpts_df.sort_values(by=["date", "time"]).reset_index(drop=True)
        additional_rcpts_df = additional_rcpts_df.drop(columns=["time"])
        return additional_rcpts_df;
    ###########################################################################################
    def _build_winn_dixie_df(this, path):
        # recptParser = WinnDixieRecptParser()
        rows = []
        for p in Path(path).glob("*.txt"):
            result = this.recptParser.parse(p.read_text(encoding="utf-8", errors="ignore"))
            for r in result["items"]:
                rows.append({
                    "vendor": this.vendorName,
                    "source": p.name,
                    "date": result["date"],
                    "time": result["time"],
                    #"manager": result["manager"],
                    #"cashier_raw": result["cashier"],
                    "item": r["item"],
                    "qty": r["qty"],
                    #"reg": r["reg"],
                    #"youPay": r["youPay"],
                    #"reportedItemsSold": result["reported"],
                    #"rowsMatchReported": result["validation"]["rowsMatchReported"],
                    #"qtyMatchReported": result["validation"]["qtyMatchReported"],
                })
    
        winndixie_df = pd.DataFrame(rows)

        if winndixie_df.empty:
            raise Exception("WinnDixieEventsDfBuilder produced empty dataframe. No receipts found.")
        
        winndixie_df["date"] = pd.to_datetime(winndixie_df["date"])
        winndixie_df["time"] = winndixie_df["time"].astype(str)
        
        winndixie_df = WinnDixieRecptParser.remove_duplicate_receipt_files(winndixie_df)
        
        # winndixie_df["source"] = "winndixie-{";
        winndixie_df = winndixie_df.sort_values(by=["date", "time"]).reset_index(drop=True)
        winndixie_df = winndixie_df.drop(columns=["time"])
        return winndixie_df;
    ###########################################################################################