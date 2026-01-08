other code
  #def drop_rare_items(self):
        # self.combined_df = self.drop_rare_items_with_zero_freq(self.combined_df, "freq_7_feat")
        # self.combined_df = self.drop_rare_items_with_zero_freq(self.combined_df, "freq_15_feat")
        # self.combined_df = self.drop_rare_items_with_zero_freq(self.combined_df, "freq_30_feat")
        # self.combined_df = self.drop_rare_items_with_zero_freq(self.combined_df, "freq_90_feat")



   def build_habit_frequency_for_training(self):
        print("build_habit_frequency_for_training()");
        df = self.combined_df[self.combined_df["didBuy_target"] == 1]
        latest_trip_date = df["date"].max()
        freq_map = self.compute_habit_frequency_map(df, latest_trip_date)
        self.combined_df["itemPurchaseHabitFrequency_feat"] = self.combined_df["itemId"].map(freq_map)
    ############################################################################################
    def recompute_habit_frequency_for_prediction_time(self, prediction_date: datetime):
        print("recompute_habit_frequency_for_prediction_time()");
        df = self.combined_df[self.combined_df["didBuy_target"] == 1]
        latest_trip_date = prediction_date
        freq_map = self.compute_habit_frequency_map(df, latest_trip_date)
        return freq_map
    ############################################################################################
    def compute_habit_frequency_map(self, filtered_df: pd.DataFrame, ref_date: datetime):
        print("compute_habit_frequency_map()");
        oldest_date = filtered_df["date"].min()
        days_span = (ref_date - oldest_date).days
        if days_span <= 0:
            return {}

        counts = filtered_df.groupby("itemId")["date"].count()
        freq_map = (counts / days_span).to_dict()
        return freq_map
    #############################################################################################


  # self.build_habit_frequency_for_training();
        # self.groceryMLCore.build_freq_ratios()
     
    # def build_habit_features(self, df, tau_days=120):
    #     df = df.copy()
    #     df["date"] = pd.to_datetime(df["date"])
    
    #     total_trips = df["date"].nunique()
    #     timeline_days = (df["date"].max() - df["date"].min()).days or 1
    
    #     rows = []
    
    #     for itemId, g in df.groupby("itemId"):
    #         buys = g[g["didBuy_target"] == 1]["date"]
    
    #         if len(buys) == 0:
    #             rows.append({
    #                 "itemId": itemId,
    #                 "habitFrequency_feat": 0.0,
    #                 "habitSpan_feat": 0.0,
    #                 "habitDecay_feat": 0.0,
    #             })
    #             continue
    
    #         first = buys.min()
    #         last = buys.max()
    
    #         habitFrequency = len(buys) / total_trips
    #         habitSpan = (last - first).days / timeline_days
    #         days_since_last = (df["date"].max() - last).days
    #         habitDecay = np.exp(-days_since_last / tau_days)
    
    #         rows.append({
    #             "itemId": itemId,
    #             "habitFrequency_feat": habitFrequency,
    #             "habitSpan_feat": habitSpan,
    #             "habitDecay_feat": habitDecay,
    #         })
    
    #     return pd.DataFrame(rows)
    # ###########################################################################################     
        
    # def compute_due_score(self, df, itemId=None, use_sigmoid=True, normalize=False, weights=None):
        
    #     if weights is None:
    #         weights = {
    #             "daysSinceThisItemLastPurchased_log_feat": 1.5,
    #             "freq_30_feat": 1.0,
    #             "freq_90_feat": 0.5
    #         }
    
    #     # --------------------------------------------------------
    #     # Optional itemId filter
    #     # --------------------------------------------------------
    #     if itemId is not None:
    #         df = df[df["itemId"] == itemId].copy()
    #     else:
    #         df = df.copy()
    
    #     # --------------------------------------------------------
    #     # RAW linear score (pre-normalization)
    #     # --------------------------------------------------------
    #     df["due_score_raw_feat"] = (
    #         weights["daysSinceThisItemLastPurchased_log_feat"] * df["daysSinceThisItemLastPurchased_log_feat"]
    #       + weights["freq_30_feat"]              * df["freq_30_feat"]
    #       + weights["freq_90_feat"]              * df["freq_90_feat"]
    #     )
    
    #     # --------------------------------------------------------
    #     # Final due_score
    #     # --------------------------------------------------------
    #     if use_sigmoid:
    #         df["due_score"] = 1 / (1 + np.exp(-df["due_score_raw"]))
    
    #     elif normalize:
    #         mean = df["due_score_raw"].mean()
    #         std  = df["due_score_raw"].std() or 1.0
    #         df["due_score"] = (df["due_score_raw"] - mean) / std
    
    #     else:
    #         df["due_score"] = df["due_score_raw"]
    
    #     return df
    ###########################################################################################

    
    # def compute_bulkAdjustedUrgencyRatio_value(self, days_since, avg_between, bulk_flag, did_buy):
    #     """
    #     Compute one bulk-adjusted urgency ratio value.
    #     Shared by training + prediction. Caller decides did_buy.
    #     """
    #     if did_buy != 1:
    #         return 0.0

    #     bulk_factor_value = 2.5
    #     denominator = avg_between * (bulk_factor_value if bulk_flag == 1 else 1.0)

    #     if denominator == 0:
    #         return 0.0

    #     return days_since / denominator
    # ###########################################################################
   
    # def create_bulkAdjustedUrgencyRatio_for_training(self, df):
    #     """
    #     Create bulkAdjustedUrgencyRatio_feat using real didBuy_target values.
    #     """
    #     if "bulkFlag" not in df.columns:
    #         df["bulkAdjustedUrgencyRatio_feat"] = [0.0] * len(df)
    #         return dfFnor

    #     ratios = []
    #     for _, row in df.iterrows():
    #         ratios.append(self.compute_bulkAdjustedUrgencyRatio_value(
    #             row["daysSinceThisItemLastPurchased_log_feat"],
    #             row["avgDaysBetweenItemPurchases_log_feat"],
    #             row["bulkFlag"],
    #             row["didBuy_target"]      # <-- real 0/1
    #         ))

    #     df["bulkAdjustedUrgencyRatio_feat"] = ratios
    #     return df
    # ###########################################################################
    # def create_bulkAdjustedUrgencyRatio_for_prediction(self, df):
    #     """
    #     Create bulkAdjustedUrgencyRatio_feat assuming did_buy==1 for all rows.
    #     """
    #     if "bulkFlag" not in df.columns:
    #         df["bulkAdjustedUrgencyRatio_feat"] = [0.0] * len(df)
    #         return df

    #     ratios = []
    #     for _, row in df.iterrows():
    #         ratios.append(self.compute_bulkAdjustedUrgencyRatio_value(
    #             row["daysSinceThisItemLastPurchased_log_feat"],
    #             row["avgDaysBetweenItemPurchases_log_feat"],
    #             row["bulkFlag"],
    #             1                           # <-- always 1 at prediction time
    #         ))

    #     df["bulkAdjustedUrgencyRatio_feat"] = ratios
    #     return df
    # ###########################################################################



      #self.create_bulkAdjustedUrgencyRatio_for_training(df);
        # ============================================================
        # MERGE HABIT FEATURES
        # ============================================================
        # habit_df = build_habit_features(combined_df)
        # df = df.merge(habit_df, on="itemId",how="left")
        # df[["habitFrequency_feat", "habitSpan_feat", "habitDecay_feat"]] = (
        #     df[["habitFrequency_feat", "habitSpan_feat", "habitDecay_feat"]].fillna(0.0)
        # ) 


              # ============================================================
        # MERGE HABIT FEATURES
        # ============================================================
        # habit_df = build_habit_features(combined_df)
        # df = df.merge(habit_df, on="itemId",how="left")
        # df[["habitFrequency_feat", "habitSpan_feat", "habitDecay_feat"]] = (
        #     df[["habitFrequency_feat", "habitSpan_feat", "habitDecay_feat"]].fillna(0.0)
        # ) 