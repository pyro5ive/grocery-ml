import pandas as pd
from abc import ABC, abstractmethod


###############################################################
class NormalizerBase(ABC):
    """
    Abstract base class defining the contract for feature normalizers.
    Implementations are responsible for learning normalization parameters
    from training data and applying them to any DataFrame.
    Follows the fit/transform pattern — fit at train time, transform at predict time.
    """

    ###############################################################
    @abstractmethod
    def fit(self, featureCols: list[str], df: pd.DataFrame) -> 'NormalizerBase':
        """
        Learn normalization parameters from the training DataFrame.

        :param featureCols: List of column names to compute normalization parameters for.
        :type featureCols: list[str]
        :param df: Training DataFrame containing the feature columns.
        :type df: pd.DataFrame
        :returns: self, to allow method chaining.
        :rtype: NormalizerBase
        """
        pass

    ###############################################################
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned normalization parameters to the DataFrame.

        :param df: Input DataFrame containing feature columns to normalize.
        :type df: pd.DataFrame
        :returns: DataFrame with normalized feature columns added.
        :rtype: pd.DataFrame
        :raises RuntimeError: If called before fit.
        """
        pass

    ###############################################################
    @abstractmethod
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
        pass

    ###############################################################
    @abstractmethod
    def get_params(self) -> dict:
        """
        Return the learned normalization parameters.

        :returns: Dictionary of column names to mean and std values.
        :rtype: dict
        :raises RuntimeError: If called before fit.
        """
        pass

    ###############################################################
    @abstractmethod
    def is_fitted(self) -> bool:
        """
        Return whether the normalizer has been fitted.

        :returns: True if fit has been called, False otherwise.
        :rtype: bool
        """
        pass