import re
import unicodedata
import spacy

class ItemNameUtils:

    def __init__(self):
        self.item_to_id = None
        self.id_to_item = None
        self.nlp = spacy.load("en_core_web_sm")
    ###########################################################################################
    def canonicalize(self, df):
        patterns = ["prairie-farm-milk","kleinpeter-milk", "kl-milk", "Milk, Fat Free,", "Fat-Free Milk"]
        self.canonicalize_items(df, patterns, "milk")
        #
        patterns = ["Bunny Bread", "sandwich-bread", "White Sandwich Bread", "bunny-bread","se-grocers-bread","seg-sandwich-bread", "seg-white-bread"]
        self.canonicalize_items(df, patterns, "bread")
        #
        patterns = ["white-bread"]
        self.canonicalize_items(df, patterns, "bread")
        #
        patterns = ["blue-bell", "ice-cream", "icescream"]
        self.canonicalize_items(df, patterns, "icecream")
        
        patterns = ["dandw-cheese", "kraft-cheese", "se-grocers-cheese", "know-and-love-cheese"]
        self.canonicalize_items(df, patterns, "cheese")
        #
        patterns = ["blue-plate-mayo", "blue-plate-mynnase"]
        self.canonicalize_items(df, patterns, "mayo")
        #
        patterns = ["gatorade", "powerade", "sports-drink"]
        self.canonicalize_items(df, patterns, "gatorade-powerade-sports-drink")
        #
        patterns = [ "tyson","chicken-cutlet", "chicken-leg", "chicken-thigh", "chicken-thighs"]
        self.canonicalize_items(df, patterns, "chicken-thigh-leg-cutlet-tyson")
        #
        patterns = ["steak","ribs", "pork", "ground-beef"]
        self.canonicalize_items(df, patterns, "steak-ribs-pork-ground-beef-cano")
        #
        patterns = ["jimmy-dean",]
        self.canonicalize_items(df, patterns, "frozen-breakfast-jimmy-dean-cano")
        #
        patterns = ["shampoo", "conditioner"]
        self.canonicalize_items(df, patterns, "shampoo-conditioner-cano")     
        #
        patterns = ["soap"]
        self.canonicalize_items(df, patterns, "soap")     

        patterns = ["chobani-yogrt-flip", "chobani-yogurt", "yogurt"]
        self.canonicalize_items(df, patterns, "yogurt")
        #
        patterns = ["coca-cola", "coca-cola-cola", "cocacola-soda", "coke", "cola"]
        self.canonicalize_items(df, patterns, "coke")
        #
        patterns = ["topcare", "top-care"]
        self.canonicalize_items(df, patterns, "otcmeds")
        #
        patterns = ["little-debbie" , "hugbi-pies", "hubig" "-hugbi-pies", "candy", "tastykake"]
        self.canonicalize_items(df, patterns, "junk-food")
        #
        patterns  = ["cereal", "kellogg-raisn-bran", "kellogg-raisin-bra", "apl-jck"]
        self.canonicalize_items(df, patterns, "cereal-raisn-bran-apl-jck_cano")
        #
        patterns = ["minute-maid-drink", "minute-maid-drinks", "minute-maid-lmnade"]
        self.canonicalize_items(df, patterns, "minute-maid-drink")
        #
        patterns = ["egglands-best-egg", "egglands-best-eggs", "eggs"]
        self.canonicalize_items(df, patterns, "eggs")
        #
        patterns = ["sprklng-water", "sparkling-ice-wtr", "sparkling-ice", "sparkling-water"]
        self.canonicalize_items(df, patterns, "sparkling-ice")
        #
        patterns = ["drinking-water", "purified-drinking",]
        self.canonicalize_items(df, patterns, "drinking-water")
        #       
        patterns = ["ground-beef"]
        self.canonicalize_items(df, patterns, "ground-beef")
        #
        patterns = ["monster-energy", "monster-enrgy", "monster"]
        self.canonicalize_items(df, patterns, "monster-energy")
        #
        patterns = ["smuckers", "jelly"]
        self.canonicalize_items(df, patterns, "jelly")
        ### TODO: use nlp libs to remove plural word
        patterns = ["cat-litter", "cats-litter"]
        self.canonicalize_items(df, patterns, "cat-litter")
        #
        patterns = ["pizza"]
        self.canonicalize_items(df, patterns, "pizza")
        #
        patterns = ["pringles"]
        self.canonicalize_items(df, patterns, "pringles")
        #
        patterns = ["dr-pepper"]
        self.canonicalize_items(df, patterns, "dr-pepper")                                      
        #
        patterns = ["aluminum-foil", "foil"]
        self.canonicalize_items(df, patterns, "aluminum-foil")                                      
        #
        patterns = ["sour-cream"]
        self.canonicalize_items(df, patterns, "sour-cream")
    
        return df;
    ###########################################################################################
    def canonicalize_items(self, df, patterns, canonical_name):
        """
        For each pattern in `patterns`, find rows where `item` contains the pattern
        and replace df['item'] with `canonical_name`.
        """
        for p in patterns:
            mask = df["item"].str.contains(p, case=False, na=False, regex=False)
            df.loc[mask, "item"] = canonical_name  
        return df
    ###########################################################################################
    def remove_items_matching_terms(self, df, text_column, exclude_terms):
        """
        Removes rows where text_column contains any of the exclude_terms.
        Throws if NaN values are found in the text column.
        """
        if df[text_column].isna().any():
            raise ValueError(f"NaN values found in column '{text_column}'")
    
        lowered_terms = []
        for term in exclude_terms:
            lowered_terms.append(term.lower())

        mask = []
        for value in df[text_column]:
            text_value = value.lower()
            found = False
            for term in lowered_terms:
                if term in text_value:
                    found = True
                    break
            mask.append(not found)
    
        return df.loc[mask].reset_index(drop=True)
    ############################################################################################
    def create_item_ids(self, df, allow_new_items=False):
        # initialize maps if needed
        if self.item_to_id is None or self.id_to_item is None:
            self.item_to_id = {}
            self.id_to_item = {}
    
        # detect unseen items
        unseen_items = [item for item in df["item"].unique() if item not in self.item_to_id]
    
        if unseen_items and not allow_new_items:
            raise RuntimeError(f"Unknown items encountered: {len(unseen_items)}")
    
        # add new items if allowed
        for item in unseen_items:
            new_id = len(self.item_to_id)
            self.item_to_id[item] = new_id
            self.id_to_item[new_id] = item
    
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
            r"\b(\d+(\.\d+)?\s?(oz|ounce|fl|lb|lbs|ct|count|pk|pack|ml|g|kg|gallon|half-gallon|l))\b",
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
    def lemmatize_item_name(self, text):
        doc = self.nlp(text)
        return " ".join(token.lemma_ for token in doc if not token.is_punct)
    ###########################################################
    @staticmethod
    def strip_prefixes_from_column(df, col_name: str, prefixes: list[str]):
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found")
    
        if not prefixes:
            return df
    
        escaped = sorted([re.escape(p.strip()) for p in prefixes], key=len, reverse=True)
        pattern = r"^(" + "|".join(escaped) + r")"
    
        df[col_name] = (
            df[col_name]
                .str.replace(pattern, "", regex=True, case=False)
                .str.strip()
                .str.lstrip("-")
    )
    
        return df
     ###########################################################
    
   
