import logging
import re
import unicodedata
import spacy


class ItemNameUtils:
    def __init__(self, logger: logging.Logger | None = None):
        thisClassName = self.__class__.__name__
        self.logger = logger if logger is not None else logging.getLogger(thisClassName)
        self.nlp = spacy.load("en_core_web_sm")
        self.logger.debug("ItemNameUtils initialized")
####################################################################################################

    def canonicalize(self, df):
        self.logger.debug("canonicalize(): start (rows=%s)", len(df))

        patterns = ["prairie-farm-milk", "kleinpeter-milk", "kl-milk", "Milk, Fat Free,", "Fat-Free Milk"]
        self.canonicalize_items(df, patterns, "milk")

        patterns = ["Bunny Bread", "sandwich-bread", "White Sandwich Bread", "bunny-bread", "se-grocers-bread", "seg-sandwich-bread", "seg-white-bread"]
        self.canonicalize_items(df, patterns, "bread")

        patterns = ["white-bread"]
        self.canonicalize_items(df, patterns, "bread")

        patterns = ["blue-bell", "ice-cream", "icescream"]
        self.canonicalize_items(df, patterns, "icecream")

        patterns = ["dandw-cheese", "kraft-cheese", "se-grocers-cheese", "know-and-love-cheese"]
        self.canonicalize_items(df, patterns, "cheese")

        patterns = ["blue-plate-mayo", "blue-plate-mynnase"]
        self.canonicalize_items(df, patterns, "mayo")

        patterns = ["gatorade", "powerade", "sports-drink"]
        self.canonicalize_items(df, patterns, "gatorade-powerade-sports-drink")

        patterns = ["tyson", "chicken-cutlet", "chicken-leg", "chicken-thigh", "chicken-thighs"]
        self.canonicalize_items(df, patterns, "chicken-thigh-leg-cutlet-tyson")

        patterns = ["steak", "ribs", "pork", "ground-beef"]
        self.canonicalize_items(df, patterns, "steak-ribs-pork-ground-beef-cano")

        patterns = ["jimmy-dean"]
        self.canonicalize_items(df, patterns, "frozen-breakfast-jimmy-dean-cano")

        patterns = ["shampoo", "conditioner"]
        self.canonicalize_items(df, patterns, "shampoo-conditioner-cano")

        patterns = ["soap"]
        self.canonicalize_items(df, patterns, "soap")

        patterns = ["chobani-yogrt-flip", "chobani-yogurt", "yogurt"]
        self.canonicalize_items(df, patterns, "yogurt")

        patterns = ["coca-cola", "coca-cola-cola", "cocacola-soda", "coke", "cola"]
        self.canonicalize_items(df, patterns, "coke")

        patterns = ["topcare", "top-care"]
        self.canonicalize_items(df, patterns, "otcmeds")

        patterns = ["little-debbie", "hugbi-pies", "hubig", "-hugbi-pies", "candy", "tastykake"]
        self.canonicalize_items(df, patterns, "junk-food")

        patterns = ["cereal", "kellogg-raisn-bran", "kellogg-raisin-bra", "apl-jck"]
        self.canonicalize_items(df, patterns, "cereal-raisn-bran-apl-jck_cano")

        patterns = ["minute-maid-drink", "minute-maid-drinks", "minute-maid-lmnade"]
        self.canonicalize_items(df, patterns, "minute-maid-drink")

        patterns = ["egglands-best-egg", "egglands-best-eggs", "eggs"]
        self.canonicalize_items(df, patterns, "eggs")

        patterns = ["sprklng-water", "sparkling-ice-wtr", "sparkling-ice", "sparkling-water"]
        self.canonicalize_items(df, patterns, "sparkling-ice")

        patterns = ["drinking-water", "purified-drinking"]
        self.canonicalize_items(df, patterns, "drinking-water")

        patterns = ["ground-beef"]
        self.canonicalize_items(df, patterns, "ground-beef")

        patterns = ["monster-energy", "monster-enrgy", "monster"]
        self.canonicalize_items(df, patterns, "monster-energy")

        patterns = ["smuckers", "jelly"]
        self.canonicalize_items(df, patterns, "jelly")

        patterns = ["cat-litter", "cats-litter"]
        self.canonicalize_items(df, patterns, "cat-litter")

        patterns = ["pizza"]
        self.canonicalize_items(df, patterns, "pizza")

        patterns = ["pringles"]
        self.canonicalize_items(df, patterns, "pringles")

        patterns = ["dr-pepper"]
        self.canonicalize_items(df, patterns, "dr-pepper")

        patterns = ["aluminum-foil", "foil"]
        self.canonicalize_items(df, patterns, "aluminum-foil")

        patterns = ["sour-cream"]
        self.canonicalize_items(df, patterns, "sour-cream")

        self.logger.debug("canonicalize(): done (rows=%s)", len(df))
        return df
####################################################################################################

    def canonicalize_items(self, df, patterns: list[str], canonical_name: str):
        self.logger.info("canonicalize_items(): start canonical='%s' patterns=%s", canonical_name, len(patterns))

        total_replaced = 0
        for pattern in patterns:
            mask = df["item"].str.contains(pattern, case=False, na=False, regex=False)
            match_count = int(mask.sum())
            if match_count > 0:
                df.loc[mask, "item"] = canonical_name
                total_replaced += match_count
            self.logger.debug("canonicalize_items(): pattern='%s' matches=%s", pattern, match_count)

        self.logger.info("canonicalize_items(): done canonical='%s' total_replaced=%s", canonical_name, total_replaced)
        return df
####################################################################################################

    def remove_items_matching_terms(self, df, text_column: str, exclude_terms: list[str]):
        self.logger.debug("remove_items_matching_terms(): start col='%s' rows=%s terms=%s", text_column, len(df), len(exclude_terms))

        if df[text_column].isna().any():
            raise ValueError(f"NaN values found in column '{text_column}'")

        lowered_terms: list[str] = []
        for term in exclude_terms:
            lowered_terms.append(term.lower())

        mask: list[bool] = []
        removed_count = 0
        for value in df[text_column]:
            text_value = value.lower()
            found = False
            for term in lowered_terms:
                if term in text_value:
                    found = True
                    break
            keep = not found
            if not keep:
                removed_count += 1
            mask.append(keep)

        result = df.loc[mask].reset_index(drop=True)
        self.logger.debug("remove_items_matching_terms(): done rows_before=%s rows_after=%s removed=%s", len(df), len(result), removed_count)
        return result
    ####################################################################################################

    @staticmethod
    def clean_item_name(name: str) -> str:
        if name is None:
            return ""

        s = name.lower()
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = s.replace("&", " and ")
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"[_/]", " ", s)
        s = re.sub(r"\b(\d+(\.\d+)?\s?(oz|ounce|fl|lb|lbs|ct|count|pk|pack|ml|g|kg|gallon|half-gallon|l))\b", " ", s)
        s = re.sub(r"\b\d+\b", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s.replace(" ", "-")
    ####################################################################################################

    def lemmatize_item_name(self, text: str) -> str:
        self.logger.debug("lemmatize_item_name(): start")
        doc = self.nlp(text)
        lemmas: list[str] = []
        for token in doc:
            if token.is_punct:
                continue
            lemmas.append(token.lemma_)
        result = " ".join(lemmas)
        self.logger.debug("lemmatize_item_name(): done")
        return result
####################################################################################################

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
####################################################################################################