class GroceryMLEncoder:

    BuildEncodedDf(self, combined_df):
      

        freq_cols = [c for c in combined_df.columns if c.startswith("freq_")]
        weather_cols = [c for c in combined_df.columns if c.endswith("_5day_avg")]
        holiday_cols = [c for c in combined_df.columns if "holiday" in c.lower()]
        school_cols = [c for c in combined_df.columns if "school" in c.lower()]
        
        daysSince_purchase_cols = [c for c in combined_df.columns if "days" in c.lower() and "purchase" in c.lower()]
        daysSince_trip_cols     = [c for c in combined_df.columns if "days" in c.lower() and "trip" in c.lower()]
        days_cols = daysSince_purchase_cols + daysSince_trip_cols

        habit_cols = ["habitFrequency", "habitSpan", "habitDecay"]
        
        self.encoded_df = combined_df.copy()
        
        self.encoded_df = normalizeAndDropCols(encoded_df, ["item_due_ratio"])
        self.encoded_df = normalizeAndDropCols(encoded_df, freq_cols)
        self.encoded_df = normalizeAndDropCols(encoded_df, weather_cols)
        self.encoded_df = normalizeAndDropCols(encoded_df, holiday_cols)
        self.encoded_df = normalizeAndDropCols(encoded_df, school_cols)
        self.encoded_df = normalizeAndDropCols(encoded_df, days_cols)
        self.encoded_df = normalizeAndDropCols(encoded_df, habit_cols)



    EncodeCycicalFeatues(self):
    
        # ---------- CYCLICAL FEATURES ----------
        self.encoded_df["dow_sin"], self.encoded_df["dow_cos"] = TemporalFeatures.encode_sin_cos( self.encoded_df["dow"], 7.0)
        self.encoded_df["month_sin"], self.encoded_df["month_cos"] = TemporalFeatures.encode_sin_cos(self.encoded_df["month"], 12.0)
        self.encoded_df["doy_sin"], self.encoded_df["doy_cos"] = TemporalFeatures.encode_sin_cos(self.encoded_df["doy"], 365.0)
        self.encoded_df = self.encoded_df.drop(columns=["dow", "month", "doy"], errors="ignore")
    
        # ---------- NON-CYCLIC TIME FEATURES ----------
        nonCycCols = ["year", "day", "quarter"]
        self.encoded_df = normalizeAndDropCols(encoded_df, nonCycCols)
    
        # ---------- DROP NON-MODEL COLS ----------
        cols_to_drop = ["source", "item", "date"]
        encoded_df = encoded_df.drop(columns=cols_to_drop, errors="ignore")

    