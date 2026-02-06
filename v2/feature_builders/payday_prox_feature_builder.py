import pandas as pd
import numpy as np
import logging
from datetime import timedelta


class PaydayProximity_FeatureBuilder:

    def __init__(self, personName: str, anchorPayday: pd.Timestamp, dateCol: str = "date"):
        self.personName = personName
        self.anchorPayday = pd.Timestamp(anchorPayday).tz_localize(None)
        self.dateCol = dateCol
        self.cycleLength = 14
        self.rawCol = f"payday_{self.personName}_raw"
        self.proximityCol = f"payday_proximity_{self.personName}_feat"
        self.scaledCol = f"payday_proximity_{self.personName}_feat_scaled"
        self.sinCol = f"payday_proximity_{self.personName}_feat_sin"
        self.cosCol = f"payday_proximity_{self.personName}_feat_cos"
        self.isPaydayCol = f"isPayday_{self.personName}_feat"
        self.logger = logging.getLogger(self.__class__.__name__)

    #===================================#
    def buildProximity(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Building payday proximity feature for %s", self.personName)
        df = df.copy()
        df[self.dateCol] = pd.to_datetime(df[self.dateCol]).dt.tz_localize(None)
        df[self.rawCol] = df[self.dateCol].apply(self._nearestPayday)
        df[self.proximityCol] = (df[self.rawCol] - df[self.dateCol]).abs().dt.days
        return df

    #===================================#
    def buildIsPayday(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Building isPayday feature for %s", self.personName)
        df = df.copy()
        df = self.buildProximity(df)
        df[self.isPaydayCol] = (df[self.proximityCol] == 0)
        return df

    #===================================#
    def buildScaledProximity(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Building scaled payday proximity feature for %s", self.personName)
        df = df.copy()
        df = self.buildProximity(df)
        df[self.scaledCol] = df[self.proximityCol] / float(self.cycleLength)
        return df

    #===================================#
    def buildCyclicalProximity(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Building cyclical payday proximity features for %s", self.personName)
        df = df.copy()
        df = self.buildProximity(df)
        angle = 2.0 * np.pi * (df[self.proximityCol] / float(self.cycleLength))
        df[self.sinCol] = np.sin(angle)
        df[self.cosCol] = np.cos(angle)
        return df

    #===================================#
    def build_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._buildAll(df)

    #===================================#
    def _buildAll(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Building all payday features for %s", self.personName)
        df = df.copy()
        df = self.buildProximity(df)
        df[self.scaledCol] = df[self.proximityCol] / float(self.cycleLength)
        angle = 2.0 * np.pi * (df[self.proximityCol] / float(self.cycleLength))
        df[self.sinCol] = np.sin(angle)
        df[self.cosCol] = np.cos(angle)
        df[self.isPaydayCol] = (df[self.proximityCol] == 0)
        return df

    #===================================#
    def _nearestPayday(self, currentDate: pd.Timestamp) -> pd.Timestamp:
        currentDate = pd.Timestamp(currentDate).tz_localize(None)
        daysDiff = (currentDate - self.anchorPayday).days
        cycleOffset = int(round(daysDiff / float(self.cycleLength)))
        return self.anchorPayday + timedelta(days=cycleOffset * self.cycleLength)
    #===================================#
