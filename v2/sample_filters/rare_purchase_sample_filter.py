def drop_rare_purchases(self, df):
    logger.info("drop_rare_purchases()")
    df = df[df["itemPurchaseCount_raw"] != 1].reset_index(drop=True)
    return df;