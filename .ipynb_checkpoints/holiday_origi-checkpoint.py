




























# import pandas as pd
# from pandas.tseries.holiday import USFederalHolidayCalendar
# import numpy as np

# class HolidayFeatures:

#     extraHolidays = {
#         "MardiGras": [
#             "2024-02-13",
#             "2025-03-04",
#             "2026-02-17",
#             "2027-02-09",
#             "2028-02-29",
#             "2029-02-13",
#             "2030-03-05"
#         ]
#     }

#     @staticmethod
#     def compute_days_until_next_holiday(dates: pd.Series) -> pd.Series:
#         dates = pd.to_datetime(dates)
#         start = dates.min().normalize()
#         end = dates.max().normalize() + pd.Timedelta(days=366)
#         holidays = USFederalHolidayCalendar().holidays(start=start, end=end)
#         next_idx = holidays.searchsorted(dates, side="left")
#         next_holidays = pd.Series(holidays[next_idx], index=dates.index)
#         return (next_holidays - dates).dt.days
#     #####################################################################

#     @staticmethod
#     def compute_days_since_last_holiday(dates: pd.Series) -> pd.Series:
#         dates = pd.to_datetime(dates)
#         start = dates.min().normalize() - pd.Timedelta(days=366)
#         end = dates.max().normalize()
#         holidays = USFederalHolidayCalendar().holidays(start=start, end=end)
#         prev_idx = holidays.searchsorted(dates, side="right") - 1
#         prev_idx = np.clip(prev_idx, 0, None)
#         prev_holidays = pd.Series(holidays[prev_idx], index=dates.index)
#         return (dates - prev_holidays).dt.days
#     #####################################################################

#     @staticmethod
#     def compute_holiday_proximity_index(dates: pd.Series, scale: int = 30) -> pd.Series:
#         dates = pd.to_datetime(dates)
#         before = HolidayFeatures.compute_days_until_next_holiday(dates)
#         after = HolidayFeatures.compute_days_since_last_holiday(dates)
#         proximity = pd.Series(0.0, index=dates.index)
#         before_mask = before <= after
#         after_mask = after < before
#         proximity.loc[before_mask] = ((scale - before.loc[before_mask]) / scale).clip(lower=0)
#         proximity.loc[after_mask] = -((scale - after.loc[after_mask]) / scale).clip(lower=0)

#         return proximity
#     ###################################################################

#     @staticmethod
#     def build_federal_holiday_flag_and_proximity_features(
#         dates: pd.Series,
#         scale: int = 30
#     ) -> pd.DataFrame:
#         """
#         Build isXHoliday_feat and proximity_XHoliday_feat columns
#         for each US federal holiday.
#         """
#         dates = pd.to_datetime(dates).normalize()
#         calendar = USFederalHolidayCalendar()
#         rules = calendar.rules

#         result_df = pd.DataFrame(index=dates.index)

#         for rule in rules:
#             holiday_dates = rule.dates(
#                 start=dates.min() - pd.Timedelta(days=366),
#                 end=dates.max() + pd.Timedelta(days=366)
#             ).normalize()

#             clean_name = rule.name.replace(" ", "").replace("'", "")

#             is_col = "is" + clean_name + "_feat"
#             prox_col = "proximity_" + clean_name + "_feat"

#             result_df[is_col] = dates.isin(holiday_dates)
#             result_df[prox_col] = HolidayFeatures._compute_proximity_to_dates(
#                 dates,
#                 holiday_dates,
#                 scale
#             )

#         return result_df
#     #####################################################################

#     @staticmethod
#     def build_extra_holiday_flag_and_proximity_features(
#         dates: pd.Series,
#         holidayName: str,
#         scale: int = 30
#     ) -> pd.DataFrame:
#         """
#         Build isXHoliday_feat and proximity_XHoliday_feat
#         for a named extra holiday.
#         """
#         dates = pd.to_datetime(dates).normalize()
#         holiday_dates = pd.to_datetime(
#             HolidayFeatures.extraHolidays.get(holidayName, [])
#         ).normalize()

#         result_df = pd.DataFrame(index=dates.index)

#         is_col = "is" + holidayName + "_feat"
#         prox_col = "proximity_" + holidayName + "_feat"

#         result_df[is_col] = dates.isin(holiday_dates)
#         result_df[prox_col] = HolidayFeatures._compute_proximity_to_dates(
#             dates,
#             holiday_dates,
#             scale
#         )

#         return result_df
#     #####################################################################

#     @staticmethod
#     def _compute_proximity_to_dates(
#         dates: pd.Series,
#         holiday_dates: pd.DatetimeIndex,
#         scale: int
#     ) -> pd.Series:
#         """
#         Internal helper. Same proximity math as compute_holiday_proximity_index,
#         but scoped to a specific holiday.
#         """
#         next_idx = holiday_dates.searchsorted(dates, side="left")
#         prev_idx = np.clip(next_idx - 1, 0, None)

#         next_holidays = pd.Series(holiday_dates[next_idx], index=dates.index)
#         prev_holidays = pd.Series(holiday_dates[prev_idx], index=dates.index)

#         before = (next_holidays - dates).dt.days
#         after = (dates - prev_holidays).dt.days

#         proximity = pd.Series(0.0, index=dates.index)

#         before_mask = before <= after
#         after_mask = after < before

#         proximity.loc[before_mask] = ((scale - before.loc[before_mask]) / scale).clip(lower=0)
#         proximity.loc[after_mask] = -((scale - after.loc[after_mask]) / scale).clip(lower=0)

#         return proximity
#     #####################################################################

