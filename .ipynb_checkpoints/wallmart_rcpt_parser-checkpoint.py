import pandas as pd
import os

class WallmartRecptParser:
    
    @staticmethod
    def BuildWallMart(folder_path: str) -> pd.DataFrame:
        """
        Import all Walmart receipt CSV files from a folder.
        Adds a 'source' column set to the CSV filename.
        """
        dataframes = []
    
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                dataframe = pd.read_csv(file_path)
                dataframe["source"] = file_name
                dataframes.append(dataframe)
    
        if len(dataframes) == 0:
            return pd.DataFrame()
    
        df = pd.concat(dataframes, ignore_index=True)
        
        df["Product Description"] = (
            df["Product Description"]
            .str.replace("Great Value", "", regex=False)
            .str.replace("Freshness Guaranteed", "", regex=False)
            .str.strip()
        )
        
        
        ## remove some non-food items
        df = df[
            ~df["Product Description"].str.contains("Mainstays", case=False, na=False)
            &
            ~df["Product Description"].str.contains("Sizes", case=False, na=False)
            &
            ~df["Product Description"].str.contains("Pen+Gear", case=False, na=False, regex=False)
            &
            ~df["Product Description"].str.contains("Athletic", case=False, na=False)  
        ]

        df = df.rename(columns={"Order Date": "date","Product Description": "item"})
        df["date"] = pd.to_datetime(df["date"])
        
        return df
    ##########################################################################