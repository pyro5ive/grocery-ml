import logging
import pandas as pd

from abstractions.target_column_builder_base import TargetColumnBuilderBase


class TargetColumnBuilder(TargetColumnBuilderBase):
    """
    Builds the target/label column for training data.
    Training-only component.
    """

    def __init__(self, targetColName: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.targetColName = targetColName
    #--------------------------#

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add the target column, defaulting all rows to True.
        Negative sample builders are expected to update this column
        for their generated rows.
        """
        self.logger.info(
            "TargetColumnBuilder.build(): start targetColName=%s",
            self.targetColName
        )

        df[self.targetColName] = True
        df[self.targetColName] = df[self.targetColName].astype(bool)

        return df
    #--------------------------#
