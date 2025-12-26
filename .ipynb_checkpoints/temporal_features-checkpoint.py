import numpy as np

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

        if col_name == "month_cyc_feat":
            return 12

        if col_name == "day_cyc_feat":
            return 31

        if col_name == "dow_cyc_feat":
            return 7

        raise ValueError(f"Unknown cyclical feature column: {col_name}")
    ###############################################
    
    @staticmethod
    def compute_days_since_last_purchase_for_item(df, reference_date_col="date"):
        df = df.sort_values(["itemId", reference_date_col]).reset_index(drop=True)
        df["daysSinceThisItemLastPurchased_feat"] = np.nan
        last_purchase_date = {}
        for i in range(len(df)):
            itemId = df.at[i, "itemId"]
            current_date = df.at[i, reference_date_col]
            if itemId in last_purchase_date:
                df.at[i, "daysSinceThisItemLastPurchased_feat"] = (current_date - last_purchase_date[itemId]).days
            else:
                df.at[i, "daysSinceThisItemLastPurchased_feat"] = np.nan
            if "didBuy_target" in df.columns and df.at[i, "didBuy_target"] == 1:
                last_purchase_date[itemId] = current_date
    
        df["daysSinceThisItemLastPurchased_feat"] = df["daysSinceThisItemLastPurchased_feat"].fillna(0)
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
    def compute_due_ratio(df, cap=3.0):
        ratio = df["daysSinceThisItemLastPurchased_feat"] / df["avgDaysBetweenItemPurchases_feat"]
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
        return ratio.clip(0, cap)
    #######################################################

    @staticmethod
    def create_days_since_last_trip(grouped_df):
        return grouped_df["date"].diff().dt.days.fillna(0)
    #######################################################

    @staticmethod
    def compute_avg_days_between_trips(grouped_df):
        return grouped_df["daysSinceLastTrip_feat"].replace(0, np.nan).expanding().mean().fillna(0)    
    #######################################################
    @staticmethod
    def create_date_features(grouped):
        dt = grouped["date"]
        grouped["year_feat"]    = dt.dt.year
        grouped["month_cyc_feat"]   = dt.dt.month
        grouped["day_cyc_feat"]     = dt.dt.day
        grouped["dow_cyc_feat"]     = dt.dt.dayofweek
        grouped["doy_feat"]     = dt.dt.dayofyear
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
        col_buy = group.columns.get_loc("didBuy_feat")
        col_freq = {w: group.columns.get_loc(f"freq_{w}_feat") for w in freq_windows}
    
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
    