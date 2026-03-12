import pandas as pd

from abstractions.non_trip_neg_sample_builder.item_eligibility_strategy_base import ItemEligibilityStrategyBase


class GlobalItemEligibilityStrategy(ItemEligibilityStrategyBase):

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
            df[[itemIdColName, itemNameColName]]
            .drop_duplicates(itemIdColName)
        )

        calendarDates = pd.DataFrame(
            {dateColName: pd.date_range(negStartDate, negEndDate, freq="D")}
        )

        return itemUniverse[[itemIdColName]].merge(calendarDates, how="cross")
    #--------------------------#
