import pandas as pd
import logging

from overrides.signature import ensure_all_kwargs_defined_in_sub


class ContinousFeatureNormalizer:

    def __init__(this):
        this.logger = logging.getLogger(__name__);
        this.normParamaters = None;
    #==========================================================================#
    def fit_normalization_params(self, featureCols: list[str], df: pd.DataFrame) -> dict:
        params = {}

        for col in featureCols:
            series = df[col];
            if not self._is_numeric_series(series):
                continue

            params[col] = {
                "mean": float(series.mean()),
                "std": float(series.std())
            }

        self._snapshot_params(params);
        return params;
    # ========================================================================#
    def _snapshot_params(self, params):
        self.normParamaters = params;
        self.logger.info("norm params is cached in this instance")
    # ========================================================================#
    def get_params(self) -> dict:
        return self.normParamaters
    #========================================================================#
    def _is_numeric_series(self, series: pd.Series) -> bool:
        if pd.api.types.is_bool_dtype(series):
            self.logger.warning(
                "Column '%s' is boolean, skipping normalization",
                series.name
            )
            return False

        if not pd.api.types.is_numeric_dtype(series):
            self.logger.warning(
                "Column '%s' is not numeric, skipping normalization",
                series.name
            )
            return False

        return True
    #========================================================================#
    def normalize_features(this, df, params: dict | None = None):
        this.logger.info("normalize_features()")
        if params is None:
            effective_params = this.normParamaters;
        else:
            effective_params = params;

        if effective_params is None:
            raise RuntimeError("No normalization params available")

        normalized_df = df.copy()
        for col, cfg in effective_params.items():
            mean_val = cfg["mean"]
            std_val = cfg["std"]
            norm_col = col.replace("_feat", "_feat_norm")

            if std_val == 0:
                normalized_df[norm_col] = 0.0
            else:
                normalized_df[norm_col] = (df[col] - mean_val) / std_val

            #normalized_df.drop(columns=[col], inplace=True)
        return normalized_df
    ###########################################################################################

