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

        continuous: list[str] = []
        excluded: list[str] = []

        for col in self.get_feature_cols(df):
            if col.endswith("_cyc_feat"):
                excluded.append(col)
                continue

            series = df[col]

            if pd.api.types.is_bool_dtype(series):
                excluded.append(col)
                continue

            if not pd.api.types.is_numeric_dtype(series):
                self.logger.warning(
                    "FeatureSchema: column '%s' is not numeric and will be excluded",
                    col
                )
                excluded.append(col)
                continue

            continuous.append(col)

        self.logger.info(
            "FeatureSchema: continuous features selected: %s | excluded: %s",
            len(continuous),
            len(excluded)
        )

        return continuous

    # =============================================#
    def get_binary_cols(self, df: pd.DataFrame) -> list[str]:

        cols = [
            c for c in self.get_feature_cols(df)
            if pd.api.types.is_bool_dtype(df[c])
        ]

        self.logger.info("FeatureSchema: binary features detected: %s", len(cols))

        return cols

    # =============================================#
    def get_cyc_cols(self, df: pd.DataFrame) -> list[str]:

        cols = [
            c for c in self.get_feature_cols(df)
            if c.endswith("_cyc_feat")
        ]

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
