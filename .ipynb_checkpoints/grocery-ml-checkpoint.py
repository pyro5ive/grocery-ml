

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime
import asyncio
import json

import gc
import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from dataset_utils import DatasetUtils
from temporal_features import TemporalFeatures
from holiday_features import HolidayFeatures
from wallmart_rcpt_parser import WallmartRecptParser
from winn_dixie_recpt_parser import WinnDixieRecptParser 
from hidden_layer_param_builder import HiddenLayerParamSetBuilder
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class GroceryML:

    
    def BuildCombinedDataset(self):


        winndixie_df = self.BuildWinnDixie();
        wallmaert_df = self.BuildWallMart();
        weather_df = self.BuildWeather();
        
        self.combined_df = pd.concat([winndixie_df, wallmaert_df], ignore_index=True)
         self.combined_df["item"] = (self.combined_df["item"]
                .str.replace(r"^\s*[-–—]\s*", "", regex=True)
                .str.strip()
        )
        self.CreateItemIds();
        self.Canonicalize();
        
        self.combined_df = self.combined_df.merge(weather_df, on="date", how="left")
                
        self.BuildTripLevelFeatures();
        
        self.combined_df["freq7_over30"], self.combined_df["freq30_over365"]  
            = TemporalFeatures.compute_freq_ratios(self.combined_df["freq_7"], self.combined_df["freq_30"], self.combined_df["freq_365"])

        #     combined_df = pd.concat(
        #     [winndixie_df, wallmart_df[["date", "item", "source"]]],
        #     ignore_index=True
        # )
        self.combined_df["item_due_ratio"] = compute_due_ratio(combined_df)
        # remove - 
       

        
        freq_windows = [7, 15, 30, 90, 365]
        max_w = max(freq_windows)
        # initialize columns
        for w in freq_windows:
            self.combined_df[f"freq_{w}_feat"] = np.nan
        self.combined_df = (self.combined_df.groupby("itemId", group_keys=False).apply(fill_freq))
        # ============================================================
        # MERGE HABIT FEATURES
        # ============================================================
        habit_df = build_habit_features(combined_df)
        
        self.combined_df = self.combined_df.merge(habit_df, on="itemId",how="left")
        
        self.combined_df[["habitFrequency_feat", "habitSpan_feat", "habitDecay_feat"]] = (
            self.combined_df[["habitFrequency_feat", "habitSpan_feat", "habitDecay_feat"]].fillna(0.0)
        )
        
        
    ###########################################################################################
    def build_habit_features(self,df, tau_days=120):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
    
        total_trips = df["date"].nunique()
        timeline_days = (df["date"].max() - df["date"].min()).days or 1
    
        rows = []
    
        for itemId, g in df.groupby("itemId"):
            buys = g[g["didBuy_target"] == 1]["date"]
    
            if len(buys) == 0:
                rows.append({
                    "itemId": itemId,
                    "habitFrequency_feat": 0.0,
                    "habitSpan_feat": 0.0,
                    "habitDecay_feat": 0.0,
                })
                continue
    
            first = buys.min()
            last = buys.max()
    
            habitFrequency = len(buys) / total_trips
            habitSpan = (last - first).days / timeline_days
            days_since_last = (df["date"].max() - last).days
            habitDecay = np.exp(-days_since_last / tau_days)
    
            rows.append({
                "itemId": itemId,
                "habitFrequency_feat": habitFrequency,
                "habitSpan_feat": habitSpan,
                "habitDecay_feat": habitDecay,
            })
    
        return pd.DataFrame(rows)
    ###########################################################################################
    def compute_due_ratio(df, cap=3.0):
        ratio = df["daysSinceLastPurchase_feat"] / df["avgDaysBetweenPurchases_feat"]
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
        return ratio.clip(0, cap)
    ###########################################################################################

    def compute_due_score(df,itemId=None,use_sigmoid=True,normalize=False, weights=None):
        
        if weights is None:
            weights = {
                "daysSinceLastPurchase_feat": 1.5,
                "freq_30_feat": 1.0,
                "freq_90_feat": 0.5
            }
    
        # --------------------------------------------------------
        # Optional itemId filter
        # --------------------------------------------------------
        if itemId is not None:
            df = df[df["itemId"] == itemId].copy()
        else:
            df = df.copy()
    
        # --------------------------------------------------------
        # RAW linear score (pre-normalization)
        # --------------------------------------------------------
        df["due_score_raw_feat"] = (
            weights["daysSinceLastPurchase_feat"] * df["daysSinceLastPurchase_feat"]
          + weights["freq_30_feat"]              * df["freq_30_feat"]
          + weights["freq_90_feat"]              * df["freq_90_feat"]
        )
    
        # --------------------------------------------------------
        # Final due_score
        # --------------------------------------------------------
        if use_sigmoid:
            df["due_score"] = 1 / (1 + np.exp(-df["due_score_raw"]))
    
        elif normalize:
            mean = df["due_score_raw"].mean()
            std  = df["due_score_raw"].std() or 1.0
            df["due_score"] = (df["due_score_raw"] - mean) / std
    
        else:
            df["due_score"] = df["due_score_raw"]
    
        return df
    ###########################################################################################

    def BuildTripLevelFeatures(self):
        # 1. Build grouped table (one row per trip date)

        grouped = ( self.combined_df[["date"]]
            .drop_duplicates()
            .sort_values("date")
            .reset_index(drop=True)
        )      
        
        grouped["daysSinceLastTrip_feat"] = TemporalFeatures.ComputeDaysSinceLastTrip(grouped)
        grouped["avgDaysBetweenTrips_feat"] = TemporalFeatures.ComputeAvgDaysBetweenTrips(grouped)
        
        # 3. Holiday / School features
        grouped["daysUntilNextHoliday_feat"] = grouped["date"].apply(HolidayFeatures.ComputeDaysUntilNextHoliday)
        grouped["daysSinceLastHoliday_feat"] = grouped["date"].apply(HolidayFeatures.ComputeDaysSinceLastHoliday)
        grouped["holidayProximityIndex_feat"] = grouped["date"].apply(HolidayFeatures.ComputeHolidayProximityIndex)
        grouped["daysUntilSchoolStart_feat"] = grouped["date"].apply(HolidayFeatures.ComputeDaysUntilSchoolStart)
        grouped["daysUntilSchoolEnd_feat"]   = grouped["date"].apply(HolidayFeatures.ComputeDaysUntilSchoolEnd)
        grouped["schoolSeasonIndex_feat"]    = grouped["date"].apply(HolidayFeatures.ComputeSchoolSeasonIndex)
               
        grouped = TemporalFeatures.CreateDateFeatures(grouped)
        
        # merge in weather
        # grouped = grouped.merge(df_weather, on="date", how="left")
        
        self.combined_df = self.combined_df.merge(grouped, on="date", how="left")
    ###########################################################################################

    def BackFillitems(self):
        # 1. Mark actual purchases in the raw receipt rows
        self.combined_df["didBuy"] = 1
        # 2. Build complete grid
        all_items = self.combined_df["itemId"].unique()
        all_dates = self.combined_df["date"].unique()
        
        full = (
            pd.MultiIndex.from_product(
                [all_dates, all_items], 
                names=["date", "itemId"]
            ).to_frame(index=False)
        )
        
        # 3. Merge raw purchases onto the full grid
        df_full = full.merge(
            self.combined_df[["date", "itemId", "item", "source", "didBuy"]],
            on=["date", "itemId"],
            how="left"
        )
        
        # 4. Fill missing purchases with didBuy=0
        df_full["didBuy"] = df_full["didBuy"].fillna(0).astype(int)
        
        # 5. NOW REPLACE combined_df with df_full
        self.combined_df = df_full.copy()
    ###########################################################################################



    def CreateItemIds(self):
        self.combined_df, self.id_to_item = DatasetUtils.CreateItemId(self.combined_df)
    ###########################################################################################

    def Canonicalize(self):
        milk_patterns = ["know-and-love-milk", "kandl-milk", "prairie-farm-milk","kleinpeter-milk", "kl-milk", "Milk, Fat Free,", "Fat-Free Milk"]
        self.CanonicalizeItems(self.combined_df, milk_patterns, "milk")
        #
        bread_patterns = ["bunny-bread","se-grocers-bread","seg-sandwich-bread", "seg-white-bread"]
        self.CanonicalizeItems(self.combined_df, bread_patterns, "bread")
        #
        cheese_patterns = ["dandw-cheese", "kraft-cheese", "se-grocers-cheese", "know-and-love-cheese"]
        self.CanonicalizeItems(self.combined_df, cheese_patterns, "cheese")
        #
        mayo_patterns = ["blue-plate-mayo", "blue-plate-mynnase"]
        self.CanonicalizeItems(self.combined_df, mayo_patterns, "mayo")
        #
        chicken_patterns = ["chicken-cutlet", "chicken-leg", "chicken-thigh", "chicken-thighs"]
        self.CanonicalizeItems(self.combined_df, chicken_patterns, "chicken")
        #
        yogurt_patterns = ["chobani-yogrt-flip", "chobani-yogurt"]
        self.CanonicalizeItems(self.combined_df, yogurt_patterns, "yogurt")
        #
        coke_patterns = ["coca-cola", "coca-cola-cola", "cocacola-soda", "coke", "cola"]
        self.CanonicalizeItems(self.combined_df, coke_patterns, "coke")
        #
        hugbi_patterns = ["hugbi-pies", "-hugbi-pies"]
        self.CanonicalizeItems(self.combined_df, hugbi_patterns, "hugbi-pies")
        #
        ceralPaterns  = ["cereal"]
        self.CanonicalizeItems(self.combined_df, ceralPaterns, "ceral")
        #
        minute_maid_patterns = ["minute-maid-drink", "minute-maid-drinks", "minute-maid-lmnade"]
        self.CanonicalizeItems(self.combined_df, minute_maid_patterns, "minute-maid-drink")
        #
        eggs_pattern = ["egglands-best-egg", "egglands-best-eggs", "eggs"]
        self.CanonicalizeItems(self.combined_df, eggs_pattern, "eggs")
    ###########################################################################################
    def BuildWallMart(self):
        wallmart_raw = WallmartRecptParser.ImportWallMart("./walmart")
        ## rename cols
        wallmart_df = wallmart_raw[["Order Date","Product Description", "source"]].copy()
        wallmart_df = wallmart_df.rename(columns={
            "Order Date": "date",
            "Product Description": "item"
        })
        
        wallmart_df["date"] = pd.to_datetime(wallmart_df["date"])
        return wallmart_df;
    ###########################################################################################
    def BuildWinnDixie(self):
        recptParser  = WinnDixieRecptParser();
            for p in Path("winndixie rcpts/StevePhone2/pdf/text").glob("*.txt"):
            result = recptParser.parse(p.read_text(encoding="utf-8", errors="ignore"))
            for r in result["items"]:
                rows.append({
                    "source": p.name,
                    "date": result["date"],
                    "time": result["time"],
                    #"manager": result["manager"],
                    #"cashier": result["cashier"],
                    "item": r["item"]
                    #"qty": r["qty"],
                    #"reg": r["reg"],
                    #"youPay": r["youPay"],
                    #"reportedItemsSold": result["reported"],
                    #"rowsMatchReported": result["validation"]["rowsMatchReported"],
                    #"qtyMatchReported": result["validation"]["qtyMatchReported"],
                })
    
        winndixie_df = pd.DataFrame(rows)
        
        winndixie_df["date"] = pd.to_datetime(winndixie_df["date"])
        winndixie_df["time"] = winndixie_df["time"].astype(str)
        
        winndixie_df = WinnDixieRecptParser.remove_duplicate_receipt_files(winndixie_df)
        
        winndixie_df = winndixie_df.sort_values(by=["date", "time"]).reset_index(drop=True)
        winndixie_df = winndixie_df.drop(columns=["time"])
        return winndixie_df;
    ###########################################################################################

    def BuildWeather(self):
        # --- WEATHER PREP ---
        weatherCols=["datetime", "temp", "humidity", "feelslike", "dew", "precip"]
        df_weather = pd.read_csv("datasets/VisualCrossing-70062 2000-01-01 to 2025-12-14.csv", usecols=weatherCols)
        df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
        df_weather = df_weather.set_index("datetime").sort_index()

        df_weather["temp_5day_avg_feat"] = df_weather["temp"].rolling(5, min_periods=1).mean()
        df_weather["feelsLike_5day_avg_feat"] = df_weather["feelslike"].rolling(5, min_periods=1).mean()
        df_weather["dew_5day_avg_feat"] = df_weather["dew"].rolling(5, min_periods=1).mean()
        df_weather["humidity_5day_avg_feat"] = df_weather["humidity"].rolling(5, min_periods=1).mean()
        df_weather["precip_5day_avg_feat"] = df_weather["precip"].rolling(5, min_periods=1).mean()

        df_weather = df_weather.drop(columns=["temp", "humidity", "feelslike", "dew", "precip"])

        # convert index to date for merging
        df_weather["date"] = df_weather.index.date
        df_weather["date"] = pd.to_datetime(df_weather["date"])
        df_weather = df_weather.set_index("date")
        return df_weather;
    ###########################################################################################

    def export_df_to_excel_table(df, file_path, sheet_name="Data"):
        """
        Export a pandas DataFrame to an Excel file as a proper Excel Table
        with no duplicated header rows.
        """
        from openpyxl import load_workbook
        from openpyxl.worksheet.table import Table, TableStyleInfo
    
        df.to_excel(file_path, sheet_name=sheet_name, index=False)
    
        workbook = load_workbook(file_path)
        worksheet = workbook[sheet_name]
    
        end_row = worksheet.max_row
        end_col = worksheet.max_column
        end_col_letter = worksheet.cell(row=1, column=end_col).column_letter
    
        table_ref = f"A1:{end_col_letter}{end_row}"
        table = Table(displayName="DataTable", ref=table_ref)
    
        style = TableStyleInfo(
            name="TableStyleMedium9",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False
        )
    
        table.tableStyleInfo = style
        worksheet.add_table(table)
    
        workbook.save(file_path)
    ###########################################################################################
    def CanonicalizeItems(self, df, patterns, canonical_name):
        """
        For each pattern in `patterns`, find rows where `item` contains the pattern
        and replace df['item'] with `canonical_name`.
        """
        for p in patterns:
            mask = df["item"].str.contains(p, case=False, na=False)
            df.loc[mask, "item"] = canonical_name
