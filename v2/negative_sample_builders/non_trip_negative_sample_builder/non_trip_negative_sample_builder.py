import logging
import pandas as pd

from abstractions.non_trip_neg_sample_builder.end_date_strategy_base import EndDateStrategyBase
from abstractions.non_trip_neg_sample_builder.item_eligibility_strategy_base import ItemEligibilityStrategyBase
from abstractions.sample_builder_base import SampleBuilderBase


class NonTripNegativeSampleBuilder(SampleBuilderBase):

    itemIdColName: str = "itemId"
    itemNameColName: str = "item"
    dateColName: str = "date"
    didBuyTargetColName: str = "didBuy_target"
    sourceColName: str = "source"
    sourceColValue: str = "no trip neg sample"

    def __init__(self, endDateStrategy: EndDateStrategyBase, itemEligibilityStrategy: ItemEligibilityStrategyBase,featureBuildRunDate: pd.Timestamp | None = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.endDateStrategy = endDateStrategy
        self.itemEligibilityStrategy = itemEligibilityStrategy
        self.featureBuildRunDate = featureBuildRunDate
    #===================================================================================#

    def build_samples(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df[self.dateColName] = pd.to_datetime(df[self.dateColName]).dt.normalize()

        tripDates = set(df[self.dateColName].unique())
        earliestDate = min(tripDates)

        currentDate = earliestDate
        while currentDate in tripDates:
            currentDate = currentDate + pd.Timedelta(days=1)

        negStartDate = currentDate

        negEndDate = self.endDateStrategy.resolve_end_date(df, self.dateColName, self.featureBuildRunDate)

        calendar = self.itemEligibilityStrategy.build_item_calendar(df, self.itemIdColName, self.itemNameColName, self.dateColName, negStartDate,negEndDate)

        merged = calendar.merge(df,on=[self.itemIdColName, self.dateColName],how="left")

        merged[self.didBuyTargetColName] = (
            merged[self.didBuyTargetColName].fillna(False).astype(bool)
        )

        merged[self.sourceColName] = (
            merged[self.sourceColName].fillna(self.sourceColValue)
        )

        return merged.sort_values(
            [self.itemIdColName, self.dateColName]
        ).reset_index(drop=True)
    #===================================================================================#
