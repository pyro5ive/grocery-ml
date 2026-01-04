import pandas as pd
import numpy as np
from datetime import timedelta

class DataCreator:

    @staticmethod
    def build_synthetic_rows_fuzzy(itemId, avgDaysBetweenPurchases, startDate, countRows, fuzzRangeDays=3):
        rows = []
        startDate = pd.to_datetime(startDate)
        stopDate = pd.to_datetime(stopDate)
    
        currentDate = startDate
        for _ in range(countRows):
            fuzz = np.random.randint(-fuzzRangeDays, fuzzRangeDays+1)
            rows.append({
                "itemId": itemId,
                "date": currentDate + timedelta(days=fuzz),
                "didBuy_target": 1,
                "source": "__syntheic__"
            })
            currentDate = currentDate + timedelta(days=avgDaysBetweenPurchases)
        return pd.DataFrame(rows)
    #########################################################################################################
    @staticmethod
    def build_synthetic_rows_until(itemId, avgDaysBetweenPurchases, startDate, stopDate, fuzzRangeDays=3):
    
        print("build_synthetic_rows_until()")
        startDate = pd.to_datetime(startDate)
        stopDate = pd.to_datetime(stopDate)
    
        rows = []
        currentDate = startDate

    
        while currentDate <= stopDate:
            fuzz = np.random.randint(-fuzzRangeDays, fuzzRangeDays+1)
            rows.append({
                "itemId": itemId,
                "date": currentDate + timedelta(days=fuzz),
                "source": "__syntheic__",
                "didBuy_target": 1
            })
            currentDate = currentDate + timedelta(days=avgDaysBetweenPurchases)
    
        return pd.DataFrame(rows)
    ##########################################################################################################