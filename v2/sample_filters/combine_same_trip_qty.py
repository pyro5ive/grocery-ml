import logging
import pandas as pd

class SameTripQtyCombiner(object):

    def __init__(this):
        this.logger = logging.getLogger(__name__);

    ###########################################################

    def filter_df(this, df):
        this.logger.info("filtering df SameTripQtyCombiner")

        # Sum quantities for duplicate date/itemId combinations
        qty_summed = df.groupby(['date', 'itemId'], as_index=False)['qty'].sum()

        # Get all other columns (drop qty, keep first occurrence of each date/itemId)
        other_cols = df.drop(columns=['qty']).drop_duplicates(subset=['date', 'itemId'])

        # Merge summed qty back with other columns
        df_combined = other_cols.merge(qty_summed, on=['date', 'itemId'], how='left')

        return df_combined
    ###########################################################