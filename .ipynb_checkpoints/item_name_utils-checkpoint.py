import re
import unicodedata

class ItemNameUtils:

    def __init__(self):
        self.item_to_id = None
        self.id_to_item = None
    ###########################################################################################

    def canonicalize_items(self, df, patterns, canonical_name):
        """
        For each pattern in `patterns`, find rows where `item` contains the pattern
        and replace df['item'] with `canonical_name`.
        """
        for p in patterns:
            mask = df["item"].str.contains(p, case=False, na=False)
            df.loc[mask, "item"] = canonical_name    
    ###########################################################################################

    def create_item_ids(self, df):
        if self.id_to_item is not None:
            raise RuntimeError("ItemId mapping already initialized")

        unique_items = sorted(df["item"].unique())
        self.item_to_id = {item: idx for idx, item in enumerate(unique_items)}
        self.id_to_item = {idx: item for item, idx in self.item_to_id.items()}

        df["itemId"] = df["item"].map(self.item_to_id)
        df.reset_index(drop=True, inplace=True)
        return df
    ###########################################################################################
    def get_id_to_item(self):
        if self.id_to_item is None:
            raise RuntimeError("ItemId mapping not initialized")
        return self.id_to_item    
    ###########################################################################################
    
    def map_item_ids_to_names(self, df, col_name="item"):
        if self.id_to_item is None:
            raise RuntimeError("ItemId mapping not initialized")
        df[col_name] = df["itemId"].map(self.id_to_item)
        return df
    ###########################################################################################

    ###########################################################################################
    
    @staticmethod
    def clean_item_name(name: str) -> str:
        if name is None:
            return ""
    
        # lowercase
        s = name.lower()
        # unicode → ascii
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        # common replacements
        s = s.replace("&", " and ")
        # remove punctuation
        s = re.sub(r"[^\w\s]", " ", s)
        # normalize separators
        s = re.sub(r"[_/]", " ", s)
        # remove sizes / counts / units
        s = re.sub(
            r"\b(\d+(\.\d+)?\s?(oz|ounce|fl|lb|lbs|ct|count|pk|pack|ml|g|kg|l))\b",
            " ",
            s
        )
        # remove standalone numbers
        s = re.sub(r"\b\d+\b", " ", s)
        # collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        # join with hyphen
        return s.replace(" ", "-")
     ###########################################################