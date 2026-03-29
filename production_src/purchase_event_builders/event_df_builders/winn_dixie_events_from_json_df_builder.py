import json
import logging
import pandas as pd

from abstractions.event_df_builder_base import EventDfBuilderBase
from abstractions.purchase_event_mapper_base import PurchaseEventMapperBase


#======================================================#
class WinnDixieEventsFromJsonDfBuilder(EventDfBuilderBase):
    """
    Builds PurchaseEvent DataFrames from WinnDixie app JSON purchase history.
    Acts as a repository: loads + flattens wire data, then maps to domain.
    """

    sourceKey: str = "winndixieAppJson"

    logger: logging.Logger
    mapper: PurchaseEventMapperBase

    #======================================================#
    def __init__(self, mapper: PurchaseEventMapperBase):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.mapper = mapper
        self.logger.info("WinnDixieEventsFromJsonDfBuilder initialized")

    #======================================================#
    def build_df(self, sourcePaths: dict) -> pd.DataFrame:
        self.logger.info("build_df(): start")

        if self.sourceKey not in sourcePaths:
            raise KeyError(
                f"sourcePaths missing required key '{self.sourceKey}'"
            )

        jsonPath: str = sourcePaths[self.sourceKey]

        rawData: list = self._load_json(jsonPath)
        self.logger.info("build_df(): loaded records=%s", len(rawData))

        rows: list = self._build_rows(rawData)
        self.logger.info("build_df(): built rows=%s", len(rows))

        wireDf: pd.DataFrame = pd.DataFrame(rows)
        self.logger.info("build_df(): wireDf shape=%s", wireDf.shape)

        domainDf: pd.DataFrame = self.mapper.to_domain_model(wireDf)

        # optional but recommended
        self.mapper.validate_domain_df(domainDf)

        self.logger.info(
            "build_df(): domainDf shape=%s",
            domainDf.shape
        )

        return domainDf

    #======================================================#
    def _load_json(self, jsonPath: str) -> list:
        self.logger.debug("_load_json(): loading path=%s", jsonPath)
        with open(jsonPath, "r", encoding="utf-8") as f:
            return json.load(f)

    #======================================================#
    def _build_rows(self, rawData: list) -> list:
        rows: list = []

        for record in rawData:
            transactionContext: dict = self._extract_transaction_context(record)
            itemRows: list = self._extract_item_rows(record, transactionContext)
            rows.extend(itemRows)

        return rows

    #======================================================#
    def _extract_transaction_context(self, record: dict) -> dict:
        totals: dict = record.get("totals", {})
        businessUnit: dict = record.get("businessUnit", {})
        address: dict = businessUnit.get("address", {})
        retailerSpecific: dict = record.get("retailerSpecific", {})
        rewards: dict = retailerSpecific.get("rewards", {})
        itemsSold = retailerSpecific.get("itemsSold")

        return {
            "transactionId":       record.get("transactionID"),
            "transactionDateTime": record.get("transactionDateTime"),
            "grossTotal":          totals.get("gross"),
            "grandTotal":          totals.get("grand"),
            "storeId":             businessUnit.get("id"),
            "banner":              businessUnit.get("banner"),
            "city":                address.get("city"),
            "territory":           address.get("territory"),
            "postalCode":          address.get("postalCode"),
            "itemsSold":           int(itemsSold) if itemsSold not in (None, "") else None,
            "basePoints":          rewards.get("basePoints"),
            "bonusPoints":         rewards.get("bonusPoints"),
            "totalTxnPoints":      rewards.get("totalTxnPoints")
        }

    #======================================================#
    def _extract_item_rows(self, record: dict, transactionContext: dict) -> list:
        rows: list = []

        for itemWrapper in record.get("itemsUngrouped", []):
            saleItem: dict = itemWrapper.get("saleItem")
            if saleItem is None:
                continue

            row: dict = dict(transactionContext)
            row["sku"] = saleItem.get("itemID")
            row["description"] = saleItem.get("description")
            rows.append(row)

        return rows
