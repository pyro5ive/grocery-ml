class DatasetUtils:

    def __init__(self):
        self.item_to_id = None
        self.id_to_item = None

    def CreateItemId(self, df):
        if self.id_to_item is not None:
            raise RuntimeError("ItemId mapping already initialized")

        unique_items = sorted(df["item"].unique())
        self.item_to_id = {item: idx for idx, item in enumerate(unique_items)}
        self.id_to_item = {idx: item for item, idx in self.item_to_id.items()}

        df["itemId"] = df["item"].map(self.item_to_id)
        df.reset_index(drop=True, inplace=True)
        return df

    def MapItemIdsToNames(self, df, col_name="item"):
        if self.id_to_item is None:
            raise RuntimeError("ItemId mapping not initialized")

        df[col_name] = df["itemId"].map(self.id_to_item)
        return df

    def GetIdToItem(self):
        if self.id_to_item is None:
            raise RuntimeError("ItemId mapping not initialized")
        return self.id_to_item