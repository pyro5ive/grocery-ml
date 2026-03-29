BRAND_PREFIXES = [
        "great-value-",
        "gv-",
        "seg-",
        "se-grocers-",
        "marketside-",
        "sam-s-choice-",
        "equate-",
        "parent-s-choice-",
        "member-s-mark-",
        "kirkland-",
        "know-and-love-",
        "walmart-",
        "kgl-",
        "kand1",
        "kandl",
        "wr-",
        ]
        
EXCLUDED_ITEMS = [
        "shirt", "joggers", "underwear", "sandals", "socks",
        "toy", "doll", "game", "plush", "fleece",
        "cleaner", "shorts", "pants", "mens", 
        "birthday", "christmas", "halloween",
        "greeting-cards", "greeting", "hallmark", "sleeves"
        ]


CANONICAL_ITEM_MAP: dict[str, list[str]] = {
    "milk": ["prairie-farm-milk", "kleinpeter-milk", "kl-milk", "Milk, Fat Free,", "Fat-Free Milk"],
    "bread": ["Bunny Bread", "sandwich-bread", "White Sandwich Bread", "bunny-bread", "se-grocers-bread", "seg-sandwich-bread", "seg-white-bread", "white-bread"],
    "icecream": ["blue-bell", "ice-cream", "icescream"],
    "cheese": ["dandw-cheese", "kraft-cheese", "se-grocers-cheese", "know-and-love-cheese"],
    "mayo": ["blue-plate-mayo", "blue-plate-mynnase"],
    "gatorade-powerade-sports-drink": ["gatorade", "powerade", "sports-drink"],
    "chicken-thigh-leg-cutlet-tyson": ["tyson", "chicken-cutlet", "chicken-leg", "chicken-thigh", "chicken-thighs"],
    "steak-ribs-pork-ground-beef-cano": ["steak", "ribs", "pork", "ground-beef"],
    "frozen-breakfast-jimmy-dean-cano": ["jimmy-dean"],
    "shampoo-conditioner-cano": ["shampoo", "conditioner"],
    "soap": ["soap"],
    "yogurt": ["chobani-yogrt-flip", "chobani-yogurt", "yogurt"],
    "coke": ["coca-cola", "coca-cola-cola", "cocacola-soda", "coke", "cola"],
    "otcmeds": ["topcare", "top-care"],
    "junk-food": ["little-debbie", "hugbi-pies", "hubig", "-hugbi-pies", "candy", "tastykake"],
    "cereal-raisn-bran-apl-jck_cano": ["cereal", "kellogg-raisn-bran", "kellogg-raisin-bra", "apl-jck"],
    "minute-maid-drink": ["minute-maid-drink", "minute-maid-drinks", "minute-maid-lmnade"],
    "eggs": ["egglands-best-egg", "egglands-best-eggs", "eggs"],
    "sparkling-ice": ["sprklng-water", "sparkling-ice-wtr", "sparkling-ice", "sparkling-water"],
    "drinking-water": ["drinking-water", "purified-drinking"],
    "ground-beef": ["ground-beef"],
    "monster-energy": ["monster-energy", "monster-enrgy", "monster"],
    "jelly": ["smuckers", "jelly"],
    "cat-litter": ["cat-litter", "cats-litter"],
    "pizza": ["pizza"],
    "pringles": ["pringles"],
    "dr-pepper": ["dr-pepper"],
    "aluminum-foil": ["aluminum-foil", "foil"],
    "sour-cream": ["sour-cream"]
}
