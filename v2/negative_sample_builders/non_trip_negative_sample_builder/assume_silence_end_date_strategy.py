import pandas as pd
from abstractions.non_trip_neg_sample_builder.end_date_strategy_base import EndDateStrategyBase

class AssumeSilenceEndDateStrategy(EndDateStrategyBase):

    def resolve_end_date(self, df: pd.DataFrame, dateColName: str) -> pd.Timestamp:
        """
        Returns the system date as the negative end date.

        Parameters
        ----------
        df : pd.DataFrame
            Provided to satisfy the shared strategy interface.
            Not used in this implementation.

        dateColName : str
            Provided to satisfy the shared strategy interface. Not used here because this strategy does not inspectthe dataset.
            It assumes silence (absence of records) means non-trip up to the current system date.

        Returns
        -------
        pd.Timestamp
            Today's date (normalized to midnight).
        """

        # dateColName is unused in this strategy because we are not deriving the end date from the dataset.
        # We extend negatives to "today".
        return pd.Timestamp.now().normalize()
    #--------------------------#