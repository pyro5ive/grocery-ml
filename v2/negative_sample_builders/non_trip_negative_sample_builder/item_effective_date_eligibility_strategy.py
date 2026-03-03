import pandas as pd

from abstractions.non_trip_neg_sample_builder.item_eligibility_strategy_base import ItemEligibilityStrategyBase


class ItemEffectiveDateEligibilityStrategy(ItemEligibilityStrategyBase):

    def build_item_calendar(
        self,
        df: pd.DataFrame,
        itemIdColName: str,
        itemNameColName: str,
        dateColName: str,
        negStartDate: pd.Timestamp,
        negEndDate: pd.Timestamp
    ) -> pd.DataFrame:

        itemUniverse = (
            df[[itemIdColName, itemNameColName, dateColName]]
            .groupby(itemIdColName, as_index=False)
            .agg(
                {
                    itemNameColName: "first",
                    dateColName: "min"
                }
            )
            .rename(columns={dateColName: "firstSeenDate"})
        )

        calendarDates = pd.DataFrame(
            {dateColName: pd.date_range(negStartDate, negEndDate, freq="D")}
        )

        calendar = itemUniverse[[itemIdColName, "firstSeenDate"]].merge(
            calendarDates,
            how="cross"
        )

        return calendar[calendar[dateColName] >= calendar["firstSeenDate"]]
#--------------------------#
