class DatasetUtils:

   @staticmethod
   def CreateItemId(df):
    unique_items = sorted(df["item"].unique())
    item_to_id = {item: idx for idx, item in enumerate(unique_items)}
    id_to_item = {idx: item for item, idx in item_to_id.items()}
    df["itemId"] = df["item"].map(item_to_id)
    df.reset_index(drop=True, inplace=True)
    return df, id_to_item