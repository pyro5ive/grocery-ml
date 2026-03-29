import pandas as pd
import logging
from abstractions.normalizer_base import NormalizerBase


#======================================================#
class ContinuousFeatureNormalizer(NormalizerBase):
    """
    Normalizes continuous feature columns using z-score standardization.
    Learns mean and standard deviation per column during fit.
    Applies stored parameters during transform to produce normalized feature columns.
    Normalized columns are named by replacing '_cont' suffix with '_cont_norm_feat'.
    """

    normParameters: dict
    fitted: bool
    logger: logging.Logger

    #======================================================#
    def __init__(self):
        """
        Initializes the normalizer with empty state.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.normParameters = {}
        self.fitted = False
        self.logger.info("ContinuousFeatureNormalizer initialized")

    #======================================================#
    def fit(self, featureCols: list[str], df: pd.DataFrame) -> 'ContinuousFeatureNormalizer':
        """
        Learn mean and standard deviation for each numeric feature column.
        Boolean columns are skipped with a warning.

        :param featureCols: List of column names to compute normalization parameters for.
        :type featureCols: list[str]
        :param df: Training DataFrame containing the feature columns.
        :type df: pd.DataFrame
        :returns: self, to allow method chaining.
        :rtype: ContinuousFeatureNormalizer
        """
        self.logger.info("fit(): start cols=%s", featureCols)

        self.normParameters = {}

        for col in featureCols:
            series: pd.Series = df[col]
            if not self._is_numeric_series(series):
                continue
            self.normParameters[col] = {
                "mean": float(series.mean()),
                "std":  float(series.std())
            }

        self.fitted = True
        self.logger.info("fit(): done params_count=%s", len(self.normParameters))
        return self

    #======================================================#
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned normalization parameters to produce normalized feature columns.
        Columns with zero standard deviation are set to 0.0.
        Normalized columns are named by replacing '_cont' with '_cont_norm_feat'.

        :param df: Input DataFrame containing feature columns to normalize.
        :type df: pd.DataFrame
        :returns: DataFrame with normalized feature columns added.
        :rtype: pd.DataFrame
        :raises RuntimeError: If called before fit.
        """
        self.logger.info("transform(): start rows=%s", len(df))

        if not self.fitted:
            raise RuntimeError("transform(): must call fit() before transform()")

        normalizedDf: pd.DataFrame = df.copy()

        for col, cfg in self.normParameters.items():
            meanVal: float = cfg["mean"]
            stdVal: float = cfg["std"]
            normCol: str = col.replace("_cont", "_cont_norm_feat")

            if stdVal == 0:
                normalizedDf[normCol] = 0.0
            else:
                normalizedDf[normCol] = (df[col] - meanVal) / stdVal

        self.logger.info("transform(): done rows=%s", len(normalizedDf))
        return normalizedDf

    #======================================================#
    def fit_transform(self, featureCols: list[str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to the training DataFrame then transform it in one call.

        :param featureCols: List of column names to compute normalization parameters for.
        :type featureCols: list[str]
        :param df: Training DataFrame containing the feature columns.
        :type df: pd.DataFrame
        :returns: DataFrame with normalized feature columns added.
        :rtype: pd.DataFrame
        """
        self.logger.info("fit_transform(): start rows=%s", len(df))
        return self.fit(featureCols, df).transform(df)

    #======================================================#
    def get_params(self) -> dict:
        """
        Return the learned normalization parameters.

        :returns: Dictionary mapping column names to mean and std values.
        :rtype: dict
        :raises RuntimeError: If called before fit.
        """
        if not self.fitted:
            raise RuntimeError("get_params(): must call fit() before get_params()")
        return dict(self.normParameters)

    #======================================================#
    def is_fitted(self) -> bool:
        """
        Return whether the normalizer has been fitted.

        :returns: True if fit has been called, False otherwise.
        :rtype: bool
        """
        return self.fitted

    #======================================================#
    def _is_numeric_series(self, series: pd.Series) -> bool:
        """
        Check whether a Series is numeric and not boolean.

        :param series: The Series to check.
        :type series: pd.Series
        :returns: True if the Series is numeric and not boolean, False otherwise.
        :rtype: bool
        """
        if pd.api.types.is_bool_dtype(series):
            self.logger.warning("_is_numeric_series(): column '%s' is boolean, skipping", series.name)
            return False

        if not pd.api.types.is_numeric_dtype(series):
            self.logger.warning("_is_numeric_series(): column '%s' is not numeric, skipping", series.name)
            return False

        return True