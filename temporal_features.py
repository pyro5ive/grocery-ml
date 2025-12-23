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
    def ComputeDaysSinceLastTrip(grouped):
        return grouped["date"].diff().dt.days.fillna(0)
    #######################################################

    @staticmethod
    def ComputeAvgDaysBetweenTrips(grouped):
        return grouped["daysSinceLastTrip_feat"].replace(0, np.nan).expanding().mean().fillna(0)    
    #######################################################
    @staticmethod
    def CreateDateFeatures(grouped):
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
    