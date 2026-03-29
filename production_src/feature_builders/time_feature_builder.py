# import pandas as pd
# import numpy as np
# import logging
#
# class TimeFeatureBuilder:
#
#
#     def __init__(self):
#         self.logger = logging.getLogger(self.__class__.__name__);
#     ################################################################################
#
#     def build_features(self, df, timeColumn="time"):
#
#         self._validate_required_columns(df);
#
#         timeSeries = pd.to_datetime(df[timeColumn], format="%H:%M:%S", errors="coerce")
#
#         # raw extracted components (not suffixed)
#         df["hour"] = timeSeries.dt.hour
#         df["minute"] = timeSeries.dt.minute
#
#         minutesSinceMidnight = df["hour"] * 60 + df["minute"]
#
#         # engineered cyclical features (training-ready)
#         df["time_sin_feat"] = np.sin(2 * np.pi * minutesSinceMidnight / 1440)
#         df["time_cos_feat"] = np.cos(2 * np.pi * minutesSinceMidnight / 1440)
#
#         return df
#     ################################################################################
#      def _validate_required_columns(self, df):
#         missing = [f for f in self.requiredFeatures if f not in df.columns]
#         if missing:
#             raise Exception(f"{self.__class__.__name__} missing required columns: {missing}")
#     #-----------------------------------------------------------------#