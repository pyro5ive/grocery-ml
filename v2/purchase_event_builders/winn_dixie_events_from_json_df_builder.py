import json
import pandas as pd
import logging

class WinnDixieEventsFromJsonDfBuilder:

    source = "winndixie_app_json"
    
    def __init__(this):
        this.logger = logging.getLogger(this.__class__.__name__)
    ################################################################################

    def build_df(this):
        this.logger.info("Starting build_df")

        rawData = this._load_json()
        this.logger.info("Loaded JSON records: %d", len(rawData))

        rows = this._build_rows(rawData)
        this.logger.info("Built flattened rows: %d", len(rows))

        df = pd.DataFrame(rows)
        this.logger.info("DataFrame created with shape %s", df.shape)

        df = this._add_derived_items_sold(df)
        this.logger.info("Derived feature added: derivedItemsSold")

        this.logger.info("Completed build_df")
        return df
    ################################################################################

    def to_interop_df(this, df):
            this.logger.info("Projecting interop DataFrame")
    
            dtSeries = pd.to_datetime(df["transactionDateTime"])
    
            interopDf = pd.DataFrame({
                "vendor": df["banner"],
                "source": this.source,
                "date": dtSeries.dt.strftime("%m/%d/%Y"),
                "time": dtSeries.dt.strftime("%H:%M:%S"),
                "sku": df["sku"],
                "item": df["description"],
                "itemsSold": df["itemsSold"],
                "derivedItemsSold": df["derivedItemsSold"]
            })
    
            this.logger.info("Interop DataFrame shape %s", interopDf.shape)
            return interopDf
    ################################################################################

    def _load_json(this):
        this.logger.debug("Loading JSON file")
        with open("datasets\\json_logs_from_winndixie_com\\detailed\\history.json") as f:
            return json.load(f)
    ################################################################################

    def _build_rows(this, rawData):
        this.logger.debug("Building rows from raw data")

        rows = []

        for record in rawData:
            transactionContext = this._extract_transaction_context(record)
            itemRows = this._extract_item_rows(record, transactionContext)
            rows.extend(itemRows)

        return rows
    ################################################################################

    def _extract_transaction_context(this, record):
        totals = record.get("totals", {})
        businessUnit = record.get("businessUnit", {})
        address = businessUnit.get("address", {})
        retailerSpecific = record.get("retailerSpecific", {})
        rewards = retailerSpecific.get("rewards", {})

        itemsSold = retailerSpecific.get("itemsSold")

        return {
            "transactionId": record.get("transactionID"),
            "transactionDateTime": record.get("transactionDateTime"),
            "grossTotal": totals.get("gross"),
            "grandTotal": totals.get("grand"),
            "storeId": businessUnit.get("id"),
            "banner": businessUnit.get("banner"),
            "city": address.get("city"),
            "territory": address.get("territory"),
            "postalCode": address.get("postalCode"),
            "itemsSold": int(itemsSold) if itemsSold not in (None, "") else None,
            "basePoints": rewards.get("basePoints"),
            "bonusPoints": rewards.get("bonusPoints"),
            "totalTxnPoints": rewards.get("totalTxnPoints")
        }
    ################################################################################

    def _extract_item_rows(this, record, transactionContext):
        rows = []

        for itemWrapper in record.get("itemsUngrouped", []):
            saleItem = itemWrapper.get("saleItem")
            if saleItem is None:
                continue

            row = dict(transactionContext)
            row["sku"] = saleItem.get("itemID")
            row["description"] = saleItem.get("description")

            rows.append(row)

        return rows
    ################################################################################

    def _add_derived_items_sold(this, df):
        this.logger.debug("Calculating derivedItemsSold")

        derivedCounts = (
            df.groupby("transactionId")
              .size()
              .reset_index(name="derivedItemsSold")
        )

        return df.merge(
            derivedCounts,
            on="transactionId",
            how="left"
        )
    ################################################################################
