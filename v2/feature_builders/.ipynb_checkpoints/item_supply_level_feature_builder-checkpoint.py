import logging
import pandas as pd

class ItemSupplyLevel_FeatBuidler:

    requiredFeatures = [
        "daysSinceLastPurchase_feat",
        "avgDaysBetweenItemPurchases_feat"
    ];

    producedFeatures = ["itemSupplyLevel_feat"];
    
    def __init__(this):
        this.logger = logging.getLogger(this.__class__.__name__);
    ################################################################
    
    def build_feature(this, df):
        this.logger.info("Creating Item Supply Level Ratio Feature")   
        try:
            ratio = np.where(
                df["avgDaysBetweenItemPurchases_feat"] > 0,
                df["daysSinceThisItemLastPurchased_raw"] / df["avgDaysBetweenItemPurchases_feat"],
                0.0
            )
    
            df["itemSupplyLevel_feat"] = np.clip(1.0 - ratio, 0.0, 1.0)
        except Exception as ex:
            logger.info("create_item_supply_level_feat() failed")
            logger.info(ex)
            raise
    
        return df

    ################################################################