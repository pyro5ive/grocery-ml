import logging
import pandas as pd

class WalMartEventsDfBuilder:


    purchaseEventsDf: pd.Dataframe;
       
    def __init__(self, recptParser: WallmartRecptParser ):
        thisClassName = self.__class__.__name__
        self.logger = logging.getLogger(thisClassName)
        self.purchaseEventsDf = None;
        self.recptParser = recptParser;
    ############################################################
    def build_df(self): 
        self.logger.info("Building Walmart purchase events");
        purchaseEventsDf = self.recptParser.build_wall_mart_df(data_sources.get("walmart"));


        self.logger.info("Walmart purchase events builder is complete");
        return purchaseEventsDf
    ###########################################################