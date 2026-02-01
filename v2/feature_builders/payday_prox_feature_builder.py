import pandas as pd
import numpy as np
import logging
from datetime import timedelta

class PaydayProximity_FeatureBuilder:

    def __init__(self, personName: str, anchorPayday: pd.Timestamp, dateCol: str = "date"):
        thisAnchorPayday = pd.Timestamp(anchorPayday).tz_localize(None)

        self.personName = personName
        self.anchorPayday = thisAnchorPayday
        self.dateCol = dateCol
        self.cycleLength = 14

        self.rawCol = f"payday_raw_{self.personName}"
        self.proximityCol = f"payday_proximity_{self.personName}_feat"

        self.normCol = f"payday_proximity_{self.personName}_norm_feat"
        self.sinCol = f"payday_proximity_{self.personName}_sin_feat"
        self.cosCol = f"payday_proximity_{self.personName}_cos_feat"

        self.isPaydayCol = f"isPayday_{self.personName}"

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

#--------------------------#

    def buildProximity(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Building payday proximity feature for %s", self.personName)

        df = df.copy()
        df[self.dateCol] = pd.to_datetime(df[self.dateCol]).dt.tz_localize(None)

        df[self.rawCol] = df[self.dateCol].apply(self._nearestPayday)
        df[self.proximityCol] = (df[self.rawCol] - df[self.dateCol]).abs().dt.days

        return df

#--------------------------#

    def buildIsPayday(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Building isPayday feature for %s", self.personName)

        df = df.copy()
        df = self.buildProximity(df)

        df[self.isPaydayCol] = (df[self.proximityCol] == 0)

        return df

#--------------------------#

    def buildNormalizedProximity(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Building normalized payday proximity feature for %s", self.personName)

        df = df.copy()
        df = self.buildProximity(df)

        df[self.normCol] = df[self.proximityCol] / float(self.cycleLength)

        return df

#--------------------------#

    def buildCyclicalProximity(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Building cyclical payday proximity features for %s", self.personName)

        df = df.copy()
        df = self.buildProximity(df)

        angleSeries = 2.0 * np.pi * (df[self.proximityCol] / float(self.cycleLength))
        df[self.sinCol] = np.sin(angleSeries)
        df[self.cosCol] = np.cos(angleSeries)

        return df

#--------------------------#

    def buildAll(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Building all payday features for %s", self.personName)

        df = df.copy()
        df = self.buildProximity(df)

        df[self.normCol] = df[self.proximityCol] / float(self.cycleLength)

        angleSeries = 2.0 * np.pi * (df[self.proximityCol] / float(self.cycleLength))
        df[self.sinCol] = np.sin(angleSeries)
        df[self.cosCol] = np.cos(angleSeries)

        df[self.isPaydayCol] = (df[self.proximityCol] == 0)

        return df

#--------------------------#

    def _nearestPayday(self, currentDate: pd.Timestamp) -> pd.Timestamp:
        currentDate = pd.Timestamp(currentDate).tz_localize(None)

        daysDiff = (currentDate - self.anchorPayday).days
        cycleOffset = int(round(daysDiff / float(self.cycleLength)))
        nearest = self.anchorPayday + timedelta(days=cycleOffset * self.cycleLength)

        return nearest

#--------------------------#
