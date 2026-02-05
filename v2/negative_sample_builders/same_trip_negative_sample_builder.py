import logging;
import pandas as pd;


class SameTripNegativeSampleBuilder:

    # TODO: These should grow up to be a param some day.
    didBuyTargetColName = 'didBuy_target';
    itemIdColName = "itemId"
    itemNameColName = "item"
    dateColName = "date";
    sourceColValue = "_same_trip_neg_sample_";
    sourceColName = "source";

    def __init__(this):
        this.logger = logging.getLogger(__name__);

    def build_samples(this, df):
        this.logger.info("Inserting negative samples for on trip days");
        df = this._insert_negative_samples(df)
        return df;

    def _insert_negative_samples(this, df):
        this.logger.info("building negative samples");

        # ensure purchase flag exists
        df = df.copy()

        # itemId → item name lookup
        item_lookup = (
            df[[this.itemIdColName, this.itemNameColName]]
            .drop_duplicates(subset=[this.itemIdColName])
        )

        # first purchase date per item (activation point)
        first_purchase = (
            df[df[this.didBuyTargetColName] == 1]
            .groupby(this.itemIdColName)[this.dateColName]
            .min()
        )

        # build valid (date, itemId) pairs ONLY after activation
        rows = []
        all_dates = df[this.dateColName].unique()

        for itemId, first_date in first_purchase.items():
            valid_dates = all_dates[all_dates >= first_date]
            for d in valid_dates:
                rows.append({this.dateColName: d, this.itemIdColName: itemId})

        full = pd.DataFrame(rows)

        # merge back original data
        df_full = full.merge(df, on=[this.dateColName, this.itemIdColName], how="left")

        # fill negatives
        df_full[this.didBuyTargetColName] = df_full[this.didBuyTargetColName].fillna(False).astype(bool)

        # restore item names
        df_full = df_full.merge(item_lookup, on=this.itemIdColName, how="left", suffixes=("", "_lookup"))
        df_full[this.itemNameColName] = df_full[this.itemNameColName].fillna(df_full["item_lookup"])
        df_full = df_full.drop(columns=["item_lookup"])

        # fill source fields for negatives
        df_full[this.sourceColName] = df_full[this.sourceColName].fillna(this.sourceColValue).astype(str)

        return df_full

