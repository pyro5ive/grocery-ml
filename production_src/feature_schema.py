import pandas as pd
import logging


class FeatureSchema:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # =============================================#
    def get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        cols = [c for c in df.columns if c.endswith("_feat")]
        self.logger.info("FeatureSchema: total feature columns detected: %s", len(cols))
        return cols

    # =============================================#

    def get_continuous_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c.endswith("_cont")]

    # =============================================#

    def get_binary_cols(self, df: pd.DataFrame) -> list[str]:
        cols = [
            c for c in df.columns
            if c.endswith("_bin_feat") and pd.api.types.is_bool_dtype(df[c])
        ]
        self.logger.info("FeatureSchema: binary features detected: %s", len(cols))
        return cols
    # =============================================#

    def get_cyc_cols(self, df: pd.DataFrame) -> list[str]:
        cols = [c for c in df.columns if c.endswith("_cyc_feat")]
        self.logger.info("FeatureSchema: cyclic features detected: %s", len(cols))
        return cols
    # =============================================#
    def get_target_col(self, df: pd.DataFrame) -> str:

        target_cols = [c for c in df.columns if c.endswith("_target")]

        if len(target_cols) != 1:
            raise ValueError(
                f"FeatureSchema: expected exactly one target column, found: {target_cols}"
            )

        self.logger.info("FeatureSchema: target column identified: %s", target_cols[0])

        return target_cols[0]
