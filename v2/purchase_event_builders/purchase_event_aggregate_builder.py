import logging
import pandas as pd

from abstractions.event_df_builder_base import EventDfBuilderBase
from purchase_event_builders.winn_dixie_events_df_builder import WinnDixieEventsDfBuilder


class PurchaseEventAggregateBuilder:

    eventsDfs: list[pd.DataFrame]
    purchaseEventDfBuilders: list[EventDfBuilderBase]
    
    def __init__(
            self,
            dataSourcePaths,
            purchaseEventDfBuilders: list[EventDfBuilderBase],
    ):
        self.sourcePaths = dataSourcePaths;
        self.logger = logging.getLogger(self.__class__.__name__)
        self.eventsDfs = [];
        self.eventsDfBuilders = purchaseEventDfBuilders;
    #########################################################

    def build_df(self):
        ## TODO Create abstraction and add builders to array
        self.logger.info("Running eventsDf builders");

        ## use builder for each vendor
        # windixie_df = self.windixieEventsBuilder.build_df();
        # walmartEventsDf = self.walmartEventsBuilder.build_df(sourcePaths.get("walmart"))
        # manulEntryDf = self.manualEntryEventsBuilder.build_df(sourcePaths.get("??"));
        # self.eventsDfs.append(windixie_df);
        for eventDfBuilder in self.eventsDfBuilders:
            self.eventsDfs.append(eventDfBuilder.build_df());

        if len(self.eventsDfs) == 0:
            self.logger.info("eventsDf is broken");
            return pd.DataFrame()

        self.logger.info("eventsDf builders are complete");
        return pd.concat(self.eventsDfs, ignore_index=True)
    #########################################################
        
