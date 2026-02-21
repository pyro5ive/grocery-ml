import logging
import pandas as pd
from pathlib import Path
from abstractions.event_df_builder_base import EventDfBuilderBase
from abstractions.purchase_event_mapper_base import PurchaseEventMapperBase
from .winn_dixie_recpt_parser import WinnDixieRecptParser
# from purchase_event_builders.event_df_builders.mappers.winn_dixie_events_df_mapper import  WinnDixieReceiptToPurchaseEventMapper


#======================================================#
class WinnDixieEventsDfBuilder(EventDfBuilderBase):

    vendorName: str = "winndixie"
    primarySourceKey: str = "winndixie"
    additionalSourceKey: str = "winndixieAdditional"

    logger: logging.Logger
    recptParser: WinnDixieRecptParser
    mapper: PurchaseEventMapperBase

    #======================================================#
    def __init__(
        self,
        recptParser: WinnDixieRecptParser,
        mapper: PurchaseEventMapperBase
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.recptParser = recptParser
        self.mapper = mapper
        self.logger.info("WinnDixieEventsDfBuilder initialized")

    #======================================================#
    def build_df(self, sourcePaths: dict) -> pd.DataFrame:
        self.logger.info("build_df(): start")

        winndixiePath: str = sourcePaths[self.primarySourceKey]
        winndixieAdditionalPath: str = sourcePaths[self.additionalSourceKey]

        winndixieDf = self._build_winn_dixie_df(winndixiePath)
        winndixieAdditionalDf = self._build_winn_dixie_additional_text_rcpts_df(
            winndixieAdditionalPath
        )

        dfs: list[pd.DataFrame] = []

        if not winndixieDf.empty:
            dfs.append(winndixieDf)

        if not winndixieAdditionalDf.empty:
            dfs.append(winndixieAdditionalDf)

        if len(dfs) == 0:
            self.logger.warning("build_df(): no data found")
            return pd.DataFrame()

        wireDf: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        domainDf: pd.DataFrame = self.mapper.to_domain_model(wireDf)

        self.logger.info(
            "build_df(): done rows=%s cols=%s",
            len(domainDf),
            len(domainDf.columns)
        )

        return domainDf

