import logging
import pandas as pd


class PurchaseEventsDfBuilder:
    
    eventsDfs: list[pd.DataFrame]
    
    def __init__(this, dataSourcePaths):
        thisClassName = this.__class__.__name__
        this.logger = logging.getLogger(thisClassName)
        this.windixieEventsBuilder = WinnDixieEventsDfBuilder();
        this.walmartEventsBuilder = WalMartEventsDfBuilder();
        this.manualEntryEventsBuilder = ManualEntryEventsDfBuilder();
        this.sourcePaths = dataSourcePaths;
        this.eventsDfs = [];
        this.eventsDfBuilders = [];
    #########################################################

    def build_df(this):
        ## TODO Create abstraction and add builders to array
        this.logger.info("Running eventsDf builders");
        
        winndixie_df = this.windixieEventsBuilder.build_df(sourcePaths.get("winndixie"));
        walmartEventsDf = this.walmartEventsBuilder.build_df(sourcePaths.get("walmart"))
        # manulEntryDf = this.manualEntryEventsBuilder.build_df(sourcePaths.get("??"));
         
        if winndixieDf is not None and len(winndixieDf) > 0: this.eventsDfs.append(winndixieDf);
        if walmartDf is not None and len(walmartDf) > 0: this.eventsDfs.append(walmartDf);
        if manualEntryDf is not None and len(manualEntryDf) > 0: this.eventsDfs.append(manualEntryDf);

        if len(this.eventsDfs) == 0:
            this.logger.info("eventsDf is broken");
            return pd.DataFrame()

        this.logger.info("eventsDf builders are complete");
        
        return pd.concat(this.eventsDfs, ignore_index=True)
    #########################################################
        
