import logging
import pandas as pd
from purchase_event_builders.winn_dixie_events_df_builder import WinnDixieEventsDfBuilder


class PurchaseEventAggregateBuilder:

    eventsDfs: list[pd.DataFrame]
    
    def __init__(this, dataSourcePaths):

        this.sourcePaths = dataSourcePaths;
        this.logger = logging.getLogger(this.__class__.__name__)
        this.windixieEventsBuilder = WinnDixieEventsDfBuilder(this.sourcePaths);
        # this.walmartEventsBuilder = WalMartEventsDfBuilder();
        # this.manualEntryEventsBuilder = ManualEntryEventsDfBuilder();
        this.eventsDfs = [];
        this.eventsDfBuilders = [];
    #########################################################

    def build_df(this):
        ## TODO Create abstraction and add builders to array
        this.logger.info("Running eventsDf builders");

        ## use builder for each vendor
        windixie_df = this.windixieEventsBuilder.build_df();
        # walmartEventsDf = this.walmartEventsBuilder.build_df(sourcePaths.get("walmart"))
        # manulEntryDf = this.manualEntryEventsBuilder.build_df(sourcePaths.get("??"));

        this.eventsDfs.append(windixie_df);

        if len(this.eventsDfs) == 0:
            this.logger.info("eventsDf is broken");
            return pd.DataFrame()

        this.logger.info("eventsDf builders are complete");
        
        return pd.concat(this.eventsDfs, ignore_index=True)
    #########################################################
        
