import pandas as pd

from abstractions.non_trip_neg_sample_builder.end_date_strategy_base import EndDateStrategyBase


class AssumeSilenceEndDateStrategy(EndDateStrategyBase):

    def resolve_end_date(
        self,
        df: pd.DataFrame,
        dateColName: str,
        featureBuildRunDate: pd.Timestamp | None
    ) -> pd.Timestamp:

        if featureBuildRunDate is None:
            raise ValueError("featureBuildRunDate is required")

        return pd.to_datetime(featureBuildRunDate).normalize()
#--------------------------#
