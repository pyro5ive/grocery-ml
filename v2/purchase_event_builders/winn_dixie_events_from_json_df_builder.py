import json
import logging
import pandas as pd
from abstractions.event_df_builder_base import EventDfBuilderBase


#======================================================#
class WinnDixieEventsFromJsonDfBuilder(EventDfBuilderBase):
    """
    Builds an events DataFrame from WinnDixie app JSON purchase history.
    Loads transaction records from a local JSON file, flattens item rows,
    and derives item sold counts per transaction.
    """

    source: str = "winndixie_app_json"
    jsonPath: str = "datasets\\json_logs_from_winndixie_com\\detailed\\history.json"
    logger: logging.Logger

    #======================================================#
    def __init__(self):
        """
        Initializes the event DataFrame builder.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("WinnDixieEventsFromJsonDfBuilder initialized")

    #======================================================#
    def build_df(self) -> pd.DataFrame:
        """
        Build the WinnDixie events DataFrame from the JSON purchase history file.

        :returns: DataFrame of flattened transaction item rows with derived features.
        :rtype: pd.DataFrame
        """
        self.logger.info("build_df(): start")

        rawData: list = self._load_json()
        self.logger.info("build_df(): loaded records=%s", len(rawData))

        rows: list = self._build_rows(rawData)
        self.logger.info("build_df(): built rows=%s", len(rows))

        df: pd.DataFrame = pd.DataFrame(rows)
        self.logger.info("build_df(): DataFrame shape=%s", df.shape)

        df = self._add_derived_items_sold(df)
        self.logger.info("build_df(): done shape=%s", df.shape)

        return df

    #======================================================#
    def to_interop_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Project the raw events DataFrame into a standardized interop format.

        :param df: Raw events DataFrame produced by build_df.
        :type df: pd.DataFrame
        :returns: Interop DataFrame with standardized vendor, date, time and item columns.
        :rtype: pd.DataFrame
        """
        self.logger.info("to_interop_df(): start shape=%s", df.shape)

        dtSeries: pd.Series = pd.to_datetime(df["transactionDateTime"])

        interopDf: pd.DataFrame = pd.DataFrame({
            "vendor":            df["banner"],
            "source":            self.source,
            "date":              dtSeries.dt.strftime("%m/%d/%Y"),
            "time":              dtSeries.dt.strftime("%H:%M:%S"),
            "sku":               df["sku"],
            "item":              df["description"],
            "itemsSold":         df["itemsSold"],
            "derivedItemsSold":  df["derivedItemsSold"]
        })

        self.logger.info("to_interop_df(): done shape=%s", interopDf.shape)
        return interopDf

    #======================================================#
    def _load_json(self) -> list:
        """
        Load the WinnDixie purchase history JSON file.

        :returns: List of raw transaction records.
        :rtype: list
        """
        self.logger.debug("_load_json(): loading path=%s", self.jsonPath)
        with open(self.jsonPath) as f:
            return json.load(f)

    #======================================================#
    def _build_rows(self, rawData: list) -> list:
        """
        Flatten all transaction records into individual item rows.

        :param rawData: List of raw transaction records from the JSON file.
        :type rawData: list
        :returns: List of flattened item row dictionaries.
        :rtype: list
        """
        self.logger.debug("_build_rows(): start records=%s", len(rawData))

        rows: list = []

        for record in rawData:
            transactionContext: dict = self._extract_transaction_context(record)
            itemRows: list = self._extract_item_rows(record, transactionContext)
            rows.extend(itemRows)

        return rows

    #======================================================#
    def _extract_transaction_context(self, record: dict) -> dict:
        """
        Extract transaction-level context fields from a raw record.

        :param record: Raw transaction record dictionary.
        :type record: dict
        :returns: Dictionary of transaction context fields.
        :rtype: dict
        """
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
        """
        Extract individual item rows from a transaction record.

        :param record: Raw transaction record dictionary.
        :type record: dict
        :param transactionContext: Transaction context fields extracted from the record.
        :type transactionContext: dict
        :returns: List of item row dictionaries with transaction context merged in.
        :rtype: list
        """
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

    #======================================================#
    def _add_derived_items_sold(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive the item count per transaction and merge it back onto the DataFrame.

        :param df: Input DataFrame containing transactionId column.
        :type df: pd.DataFrame
        :returns: DataFrame with derivedItemsSold column added.
        :rtype: pd.DataFrame
        """
        self.logger.debug("_add_derived_items_sold(): start")

        derivedCounts: pd.DataFrame = (
            df.groupby("transactionId")
              .size()
              .reset_index(name="derivedItemsSold")
        )

        return df.merge(derivedCounts, on="transactionId", how="left")