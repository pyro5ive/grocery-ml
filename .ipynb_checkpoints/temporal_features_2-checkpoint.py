import numpy as np
import pytz
import datetime
from math import exp
import pandas as pd

class TemporalFeatures:

    @staticmethod
    def compute_freq_ratios(freq7, freq30, freq365, epsilon=1e-6):
        """
        Computes base frequency ratios only.
        """
        freq7_over30 = freq7 / (freq30 + epsilon)
        freq30_over365 = freq30 / (freq365 + epsilon)
        return freq7_over30, freq30_over365
    #######################################################
    @staticmethod
    def get_period_for_column(col_name):
        """
        Returns the cycle period for a cyclical feature column.
        """

        if col_name == "month_cyc_raw":
            return 12

        if col_name == "day_cyc_raw":
            return 31

        if col_name == "dow_cyc_raw":
            return 7

        raise ValueError(f"Unknown cyclical feature column: {col_name}")
    ###############################################
    def compute_recent_purchase_penalty(df):
        """
        Higher values mean 'recently bought' → stronger penalty.
        Lower values mean 'long time ago' → weaker penalty.
        """  
        df["recentPurchasePenalty_raw"] = ( df["daysSinceThisItemLastPurchased_raw"] / df["avgDaysBetweenItemPurchases_feat"] ).replace([float("inf"), -float("inf")], 0).fillna(0)
    
        df["recentPurchasePenalty_raw"] = df["recentPurchasePenalty_raw"].apply(
            lambda x: exp(-x)
        )
        #return df; 
   ################################################
    @staticmethod
    def compute_expected_gap_ewma_feat(df, alpha=0.3):
        """
        Computes an exponentially weighted moving average (EWMA)
        of days between purchases, per item.
    
        - Only didBuy_target == 1 advances the EWMA
        - Negative rows inherit the last known expected gap
        - Alpha controls adaptation speed (0.2–0.4 typical)
        """
    
        print("compute_expected_gap_ewma()");
        df = df.sort_values(["itemId", "date"]).reset_index(drop=True)
    
        df["expectedDaysBetweenPurchases_ewma_feat"] = 0.0
    
        last_purchase_date_by_item = {}
        ewma_gap_by_item = {}
    
        for i in range(len(df)):
            itemId = df.at[i, "itemId"]
            current_date = df.at[i, "date"]
            didBuy = df.at[i, "didBuy_target"]
    
            if didBuy == 1:
                if itemId in last_purchase_date_by_item:
                    gap_days = (current_date - last_purchase_date_by_item[itemId]).days
    
                    if itemId in ewma_gap_by_item:
                        prev_ewma = ewma_gap_by_item[itemId]
                        new_ewma = (alpha * gap_days) + ((1.0 - alpha) * prev_ewma)
                    else:
                        new_ewma = float(gap_days)
    
                    ewma_gap_by_item[itemId] = new_ewma
                    df.at[i, "expectedDaysBetweenPurchases_ewma_feat"] = new_ewma
                else:
                    # first purchase → no gap yet
                    df.at[i, "expectedDaysBetweenPurchases_ewma_feat"] = 0.0
    
                last_purchase_date_by_item[itemId] = current_date
    
            else:
                # carry forward last known EWMA
                if itemId in ewma_gap_by_item:
                    df.at[i, "expectedDaysBetweenPurchases_ewma_feat"] = ewma_gap_by_item[itemId]
                else:
                    df.at[i, "expectedDaysBetweenPurchases_ewma_feat"] = 0.0
    
        return df
    ############################################################
    
    @staticmethod
    def compute_days_since_last_purchase_for_item(df, colName: str, reference_date_col="date"):
        df = df.sort_values(["itemId", reference_date_col]).reset_index(drop=True)
        df[colName] = np.nan
        last_purchase_date = {}
        for i in range(len(df)):
            itemId = df.at[i, "itemId"]
            current_date = df.at[i, reference_date_col]
            if itemId in last_purchase_date:
                df.at[i, colName] = (current_date - last_purchase_date[itemId]).days
            else:
                df.at[i, colName] = np.nan
            if "didBuy_target" in df.columns and df.at[i, "didBuy_target"] == 1:
                last_purchase_date[itemId] = current_date
    
        df[colName] = df[colName].fillna(0)
        return df
    ############################################################
    @staticmethod
    def compute_avg_days_between_item_purchases(df):
        df = df.sort_values(["itemId", "date"]).reset_index(drop=True)
        purchase_gap = df.where(df["didBuy_target"] == 1).groupby("itemId")["date"].diff().dt.days
        avg_gap = purchase_gap.groupby(df["itemId"]).expanding().mean().reset_index(level=0, drop=True)
        df["avgDaysBetweenItemPurchases_feat"] = avg_gap.groupby(df["itemId"]).ffill().fillna(0)

        return df
    #######################################################
    @staticmethod
    def compute_item_due_ratio(df, cap=3.0):
        ratio = df["daysSinceThisItemLastPurchased_raw"] / df["avgDaysBetweenItemPurchases_feat"]
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
        return ratio.clip(0, cap)
    ##########################################################################################

    ######## TRIP #######
    @staticmethod
    def create_days_since_last_trip(targetDf):
        return targetDf["date"].diff().dt.days.fillna(0)
    #######################################################
    @staticmethod
    def compute_days_since_last_trip_value(df, prediction_date, date_col: str = "date"):
        prediction_date_ts = pd.to_datetime(prediction_date)
        last_trip_date = pd.to_datetime(df[date_col]).max()

        if pd.isna(last_trip_date):
            return None

        return int((prediction_date_ts - last_trip_date).days)
    ########################################################
    @staticmethod
    def compute_avg_days_between_trips(targetDf):
        return targetDf["daysSinceLastTrip_feat"].replace(0, np.nan).expanding().mean().fillna(0)    
    #######################################################
    @staticmethod
    def compute_trip_due_ratio(targetDf):
        targetDf["tripDueRatio_feat"] = (targetDf["daysSinceLastTrip_feat"] / targetDf["avgDaysBetweenTrips_feat"]).fillna(0)
    ###########################################################################################
    @staticmethod
    def create_date_features(grouped):
        dt = grouped["date"]
        grouped["year_feat"]    = dt.dt.year
        grouped["month_cyc_raw"]   = dt.dt.month
        grouped["day_cyc_raw"]     = dt.dt.day
        grouped["dow_cyc_raw"]     = dt.dt.dayofweek
        grouped["doy_raw"]     = dt.dt.dayofyear
        grouped["quarter_feat"] = dt.dt.quarter
        return grouped;
    #######################################################
    @staticmethod
    def encode_sin_cos(value, period):
        angle = 2 * np.pi * value / period
        return np.sin(angle), np.cos(angle)
    #######################################################
    @staticmethod
    def fill_freq(group):
        group = group.copy()
        group = group.sort_values("date").reset_index(drop=True)
    
        history = []
    
        col_date = group.columns.get_loc("date")
        col_buy = group.columns.get_loc("didBuy_target")
        col_freq = {w: group.columns.get_loc(f"freq_{w}_raw") for w in freq_windows}
    
        for i in range(len(group)):
            cur_date = group.iat[i, col_date]
    
            # record purchase
            if group.iat[i, col_buy] == 1:
                history.append(cur_date)
    
            # prune history ONCE using largest window
            cutoff_max = cur_date - pd.Timedelta(days=max_w)
            history = [d for d in history if d >= cutoff_max]
    
            # compute windowed counts
            for w in freq_windows:
                cutoff = cur_date - pd.Timedelta(days=w)
                count = 0
                for d in history:
                    if d >= cutoff:
                        count += 1
                group.iat[i, col_freq[w]] = count
    
        return group
    ####################################################################################################
    # @staticmethod
    # def add_is_daylight_savings_raw(df, date_col="date"):
    #     local_tz = pytz.timezone("America/Chicago")
    #     df["isDayLightSavingsTime_feat"] = df[date_col].apply(
    #         lambda d: 1 if local_tz.localize(pd.to_datetime(d)).dst() != pd.Timedelta(0) else 0
    #     )
    #     return df
    # #####################################################################################################
    # @staticmethod
    # def add_dst_proximity_index(df, date_col="date"):
    #     """
    #     Adds dstProximityIndex_raw:
    #     distance in days to the *nearest DST boundary*
    #     (start or end), signed so models can sense transitions.
    #     """
    #     df["dstProximityIndex_feat"] = 0

    #     for i, row in df.iterrows():
    #         d = pd.to_datetime(row[date_col])

    #         # DST boundaries in US/Central for that year
    #         year = d.year
    #         dst_start = pd.to_datetime(f"{year}-03-08") + pd.offsets.Week(weekday=6)  # 2nd Sunday March
    #         dst_end   = pd.to_datetime(f"{year}-11-01") + pd.offsets.Week(weekday=6)  # 1st Sunday Nov

    #         # nearest boundary distance
    #         dist_to_start = (dst_start - d).days
    #         dist_to_end   = (dst_end - d).days
    #         nearest = dist_to_start if abs(dist_to_start) < abs(dist_to_end) else dist_to_end

    #         df.at[i, "dstProximityIndex_feat"] = nearest

    #     return df
    ####################################################################################################
    @staticmethod
    def add_dst_since_until_features(df, date_col="date"):
        """
        Adds:
          daysSinceDstChange_raw : days since last DST boundary (0 if boundary day)
          daysUntilDstChange_raw : days until next DST boundary (0 if boundary day)
        """
        import pandas as pd
    
        df["daysSinceDstChange_feat"] = 0
        df["daysUntilDstChange_feat"] = 0
    
        for i, row in df.iterrows():
            d = pd.to_datetime(row[date_col])
            year = d.year
    
            # DST change dates in US/Central
            dst_start = pd.to_datetime(f"{year}-03-08") + pd.offsets.Week(weekday=6)
            dst_end   = pd.to_datetime(f"{year}-11-01") + pd.offsets.Week(weekday=6)
    
            # pick last boundary and next boundary
            last_boundary  = dst_start if d >= dst_start else pd.to_datetime(f"{year-1}-11-01") + pd.offsets.Week(weekday=6)
            next_boundary1 = dst_end   if d < dst_end   else pd.to_datetime(f"{year+1}-03-08") + pd.offsets.Week(weekday=6)
            next_boundary2 = dst_start if d < dst_start else pd.to_datetime(f"{year+1}-11-01") + pd.offsets.Week(weekday=6)
            next_boundary  = next_boundary1 if next_boundary1 > d else next_boundary2
    
            df.at[i, "daysSinceDstChange_raw"] = (d - last_boundary).days
            df.at[i, "daysUntilDstChange_raw"] = (next_boundary - d).days
    
        return df
    ####################################################################################################

  
    