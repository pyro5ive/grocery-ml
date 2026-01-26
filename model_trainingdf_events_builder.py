import logging;
import pandas as pd;


class HistoricalPurchaseEventsDfBuilder:

    def __init__(self, eventColumns: list[str]):
        self.logger = logging.getLogger(self.__class__.__name__);
        self.eventColumns = eventColumns;
    #---------------------------------------------------------------#
    
    def build_df(self, trainingDf: pd.DataFrame) -> pd.DataFrame:
        if trainingDf is None or len(trainingDf) == 0:
            return pd.DataFrame();

        if "didBuy_target" not in trainingDf.columns:
            raise RuntimeError("missing didBuy_target");

        eventsDf = trainingDf[trainingDf["didBuy_target"] == 1];

        if len(eventsDf) == 0:
            return pd.DataFrame();

        eventsDf = eventsDf[self.eventColumns].copy();
        eventsDf.reset_index(drop=True, inplace=True);

        return eventsDf;
    #---------------------------------------------------------------##
