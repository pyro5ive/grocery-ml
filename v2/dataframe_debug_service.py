import os
from datetime import datetime
import pandas as pd


class DataFrameDebugExportService:
    """
    Infrastructure service for exporting DataFrames to disk for debugging.
    """

    def __init__(self, baseDir: str = "debug", enabled: bool = True):
        self.baseDir = baseDir
        self.enabled = enabled

        if self.enabled:
            os.makedirs(self.baseDir, exist_ok=True)

    #--------------------------#
    def export(self, df: pd.DataFrame, name: str | None = None) -> None:
        if not self.enabled:
            return

        dfName: str = name or df.attrs.get("debug_name", "df")
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fileName: str = f"{dfName}-{timestamp}.csv"
        path: str = os.path.join(self.baseDir, fileName)

        df.to_csv(path, index=False)