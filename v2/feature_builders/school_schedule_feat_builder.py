import logging
import pandas as pd
import numpy as np

from feature_builders.school_features import SchoolFeatures


class SchoolSchedule_FeatureBuilder:

    requiredFeatures = []
    producedFeatures = [
        "daysUntilSchoolStart_raw",
        "daysUntilSchoolEnd_raw",
        "isSchoolInSession_feat",
        "schoolCycle_sin_feat",
        "schoolCycle_cos_feat"
    ]

    def __init__(this, dateCol: str = "date"):
        this.logger = logging.getLogger(this.__class__.__name__)
        this.dateCol = dateCol
    #-----------------------------------------------------------------#

    def build_feature(this, df: pd.DataFrame) -> pd.DataFrame:

        this.logger.info("Building School Schedule Features")

        if this.dateCol not in df.columns:
            raise Exception(f"Missing required date column: {this.dateCol}")

        dates = pd.to_datetime(df[this.dateCol])

        df["daysUntilSchoolStart_raw"] = SchoolFeatures.compute_days_until_school_start(dates)
        df["daysUntilSchoolEnd_raw"] = SchoolFeatures.compute_days_until_school_end(dates)

        df["isSchoolInSession_feat"] = SchoolFeatures.compute_is_school_in_session(dates)

        cycle_pos = SchoolFeatures.compute_school_cycle_position(dates)

        df["schoolCycle_sin_feat"] = np.sin(2 * np.pi * cycle_pos)
        df["schoolCycle_cos_feat"] = np.cos(2 * np.pi * cycle_pos)

        return df

    #-----------------------------------------------------------------#
