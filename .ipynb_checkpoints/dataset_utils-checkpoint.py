import re
import unicodedata

class DatasetUtils:

    def __init__(self):
        self.item_to_id = None
        self.id_to_item = None
    ###################################
    
    def CreateItemId(self, df):
        if self.id_to_item is not None:
            raise RuntimeError("ItemId mapping already initialized")

        unique_items = sorted(df["item"].unique())
        self.item_to_id = {item: idx for idx, item in enumerate(unique_items)}
        self.id_to_item = {idx: item for item, idx in self.item_to_id.items()}

        df["itemId"] = df["item"].map(self.item_to_id)
        df.reset_index(drop=True, inplace=True)
        return df
    ###################################
    
    @staticmethod
    def canonicalize_item_name(name: str) -> str:
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
    @staticmethod
    def remove_seg_token(s: str) -> str:
        return re.sub(r"(^|-)\bseg\b(-|$)", "-", s).strip("-")
     ###########################################################
   
    def MapItemIdsToNames(self, df, col_name="item"):
            if self.id_to_item is None:
                raise RuntimeError("ItemId mapping not initialized")
    
            df[col_name] = df["itemId"].map(self.id_to_item)
            return df
     ###########################################################
        
    def GetIdToItem(self):
            if self.id_to_item is None:
                raise RuntimeError("ItemId mapping not initialized")
            return self.id_to_item