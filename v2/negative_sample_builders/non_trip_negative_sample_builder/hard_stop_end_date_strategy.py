import pandas as pd

from abstractions.non_trip_neg_sample_builder.end_date_strategy_base import EndDateStrategyBase


class HardStopAtTruthEndDateStrategy(EndDateStrategyBase):

    def resolve_end_date(
        self,
        df: pd.DataFrame,
        dateColName: str,
        featureBuildRunDate: pd.Timestamp | None
    ) -> pd.Timestamp:

        return pd.to_datetime(df[dateColName]).dt.normalize().max()
    #--------------------------#
