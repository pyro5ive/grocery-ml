import logging
import json


class ItemIdMapper:
    def __init__(self, logger: logging.Logger | None = None):
        thisClassName = self.__class__.__name__
        self.logger = logger if logger is not None else logging.getLogger(thisClassName)
        self.item_to_id: dict[str, int] | None = None
        self.id_to_item: dict[int, str] | None = None
        self.logger.info("ItemIdMapper initialized")
################################################################################

    def create_item_ids(self, df):
        self.logger.info("create_item_ids(): start rows=%s mapping_exists=%s", len(df), self.item_to_id is not None)
        if self.item_to_id is None:
            return self._build_item_ids(df)
        return self._map_existing_item_ids(df)
################################################################################

    def _build_item_ids(self, df):
        if "item" not in df.columns:
            raise ValueError("_build_item_ids(): df missing required column 'item'")

        self.logger.info("_build_item_ids(): building new itemId mapping rows=%s", len(df))

        self.item_to_id = {}
        self.id_to_item = {}

        for item in df["item"].unique():
            new_id = len(self.item_to_id)
            self.item_to_id[item] = new_id
            self.id_to_item[new_id] = item

        df["itemId"] = df["item"].map(self.item_to_id)

        if df["itemId"].isna().any():
            nan_count = int(df["itemId"].isna().sum())
            self.logger.error("_build_item_ids(): NaNs detected after mapping count=%s", nan_count)
            raise RuntimeError("itemId mapping produced NaNs during build")

        df.reset_index(drop=True, inplace=True)
        self.logger.info("_build_item_ids(): mapping_size=%s", len(self.item_to_id))
        return df
################################################################################

    def _map_existing_item_ids(self, df):
        if "item" not in df.columns:
            raise ValueError("_map_existing_item_ids(): df missing required column 'item'")

        self.logger.info("_map_existing_item_ids(): start rows=%s", len(df))

        known_items = set(self.item_to_id.keys())
        unseen_mask = ~df["item"].isin(known_items)
        dropped_count = int(unseen_mask.sum())

        if dropped_count > 0:
            unseen_items = df.loc[unseen_mask, "item"].dropna().unique().tolist()
            preview = unseen_items[:10]
            self.logger.warning(
                "_map_existing_item_ids(): dropping unseen items rows_dropped=%s unique_items=%s preview=%s",
                dropped_count, len(unseen_items), preview
            )
            df = df.loc[~unseen_mask].copy()

        if len(df) == 0:
            self.logger.warning("_map_existing_item_ids(): all rows dropped due to unseen items")
            return df.reset_index(drop=True)

        df["itemId"] = df["item"].map(self.item_to_id)

        if df["itemId"].isna().any():
            nan_count = int(df["itemId"].isna().sum())
            self.logger.error("_map_existing_item_ids(): unexpected NaNs after mapping count=%s", nan_count)
            raise RuntimeError("Unexpected NaNs after itemId mapping")

        df.reset_index(drop=True, inplace=True)
        self.logger.info("_map_existing_item_ids(): done rows=%s", len(df))
        return df
################################################################################

    def export_mapping(self, path: str):
        if self.item_to_id is None or self.id_to_item is None:
            raise RuntimeError("export_mapping(): mapping not initialized")

        payload = {
            "item_to_id": self.item_to_id,
            "id_to_item": self.id_to_item
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self.logger.info("export_mapping(): saved mapping path=%s items=%s", path, len(self.item_to_id))
################################################################################

    def import_mapping(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if "item_to_id" not in payload or "id_to_item" not in payload:
            raise RuntimeError("import_mapping(): invalid mapping file")

        self.item_to_id = {k: int(v) for k, v in payload["item_to_id"].items()}
        self.id_to_item = {int(k): v for k, v in payload["id_to_item"].items()}

        self.logger.info("import_mapping(): loaded mapping path=%s items=%s", path, len(self.item_to_id))
################################################################################

    def get_id_to_item(self) -> dict[int, str]:
        if self.id_to_item is None:
            raise RuntimeError("ItemId mapping not initialized")
        self.logger.info("get_id_to_item(): size=%s", len(self.id_to_item))
        return self.id_to_item
################################################################################

    def map_item_ids_to_names(self, df, col_name: str = "item"):
        if self.id_to_item is None:
            raise RuntimeError("ItemId mapping not initialized")
        self.logger.info("map_item_ids_to_names(): start rows=%s col_name='%s'", len(df), col_name)
        df[col_name] = df["itemId"].map(self.id_to_item)
        self.logger.info("map_item_ids_to_names(): done")
        return df
################################################################################
