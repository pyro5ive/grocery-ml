import logging;
import pandas as pd;

class NonTripNegativeSampleBuilder:

    def __init__(this):
        this.logger = logging.getLogger(__name__);
    #****************************************************************#

    def build_samples(this, df):
        this.logger.info("Inserting negative samples for non-trip days");
        df = this.create(df);
        return df;
    #****************************************************************#

    def build_items(this, df):
        # itemId → item name lookup
        item_lookup = (
            df[[this.itemIdColName, this.itemNameColName]]
            .drop_duplicates(subset=[this.itemIdColName])
        )
        return item_lookup
    #****************************************************************

    def create(this, df, days: int = 365) -> pd.DataFrame:
        df = df.copy();
        df["date"] = pd.to_datetime(df["date"]).dt.normalize();

        # lookup that already exists in source df
        item_lookup = df[["itemId", "item"]].drop_duplicates("itemId")

        max_date = df["date"].max()
        min_date = max_date - pd.Timedelta(days=days - 1)

        calendar = (
            item_lookup[["itemId"]]
             .merge(
                    pd.DataFrame({"date": pd.date_range(min_date, max_date, freq="D")}),
                    how="cross"
                )
            )

        merged = calendar.merge(df, on=["itemId", "date"], how="left")

        # fill required fields
        merged["didBuy_target"] = merged["didBuy_target"].fillna(False).astype(bool)
        merged["source"] = merged["source"].fillna("_neg_sample_no_trip")

        # restore item deterministically (no NaNs possible)
        merged = merged.merge(item_lookup, on="itemId", how="left", suffixes=("", "_lk"))
        merged["item"] = merged["item"].fillna(merged["item_lk"])
        merged = merged.drop(columns=["item_lk"])

        merged = merged.sort_values(["itemId", "date"]).reset_index(drop=True)
        return merged[["date", "source", "itemId", "item", "qty", "didBuy_target"]]
    ############################################################################################