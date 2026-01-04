        
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime
import asyncio
import json
import gc
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import spacy 

## Domain
from grocery_ml_core import GroceryMLCore
from school_features import SchoolFeatures
from weather_features import WeatherFeatures
from item_name_utils import ItemNameUtils
from temporal_features_2 import TemporalFeatures
from data_creator import DataCreator
from holiday_features import HolidayFeatures
from wallmart_rcpt_parser import WallmartRecptParser
from winn_dixie_recpt_parser import WinnDixieRecptParser 
from hidden_layer_param_builder import HiddenLayerParamSetBuilder
from weather_service import NwsWeatherService;

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class GroceryML:

    brand_prefixes = [
    "great-value-",
    "gv-",
    "se-grocers-",
    "marketside-",
    "sam-s-choice-",
    "equate-",
    "parent-s-choice-",
    "member-s-mark-",
    "kirkland-",
    "walmart-",
    "kgl-",
    "kand1",
    "kandl",
    "wr-",
    ]
    
    exclude_items = [
        "shirt", "joggers", "underwear", "sandals", "socks",
        "toy", "doll", "game", "plush", "fleece"
        "cleaner", "shorts", "pants", "mens", 
        "birthday", "christmas", "halloween",
        "greeting-cards", "greeting", "hallmark", "sleeves"
        ]
    
    _combined_df: pd.DataFrame = None 
    training_df: pd.DataFrame = None
    live_df: pd.DataFrame = None
    groceryMLCore: GroceryMLCore = None;
    itemNameUtils = None; 
    weatherService = None; 
    trainingSources  = {
        "walmart": r"data\training\walmart",
        "winndixie": r"data\training\winndixie\txt",
        "winndixieAdditional" : r"data\training\winndixie\additionalTxtRcpts",
        "weather": r"data\live\weather\VisualCrossing-70062 2000-01-01 to 2025-12-14.csv"
    }
    
    liveSources  = {
        "walmart": r"data\live\walmart",
        "winndixie": r"data\live\winndixie\txt",
        "winndixieAdditional" : r"data\live\winndixie\additionalTxtRcpts",
        "weather": r"data\live\weather\VisualCrossing-70062 2000-01-01 to 2025-12-14.csv"
    }
        
    def __init__(self):
        pass;
        self.itemNameUtils = ItemNameUtils();
        self.weatherService = NwsWeatherService();
        self.groceryMLCore = GroceryMLCore();
    ###########################################################################################        
    
    def build_training_df(self):
        self.training_df =  self._build_combined_df(self.trainingSources)
    ###########################################################################################        
    def build_live_df(self):
        self.live_df = self._build_combined_df(self.liveSources)
        
    ###########################################################################################
    def _build_combined_df(self, data_sources: Dict):
     
        print(f"build_combined_df()") 
        winndixie_df = self.groceryMLCore.build_winn_dixie_df(data_sources.get("winndixie"));
        winndixie_add_txt_rpts_df =  self.groceryMLCore.build_winn_dixie_additional_text_rcpts_df(data_sources.get("winndixieAdditional"))
        winndixie_df = pd.concat([winndixie_df, winndixie_add_txt_rpts_df], ignore_index=True)
        #       
        wallmart_df = WallmartRecptParser.build_wall_mart_df(data_sources.get("walmart"));
        self._combined_df = pd.concat([winndixie_df, wallmart_df[["date", "item", "source"]]],ignore_index=True)        
        if winndixie_df is None:
            raise RuntimeError("build_winn_dixie_df() returned None")
        if winndixie_add_txt_rpts_df is None:
            raise RuntimeError("build_winn_dixie_additional_text_rcpts_df() returned None")
        if wallmart_df is None:
            raise RuntimeError("build_wall_mart_df() returned None")
        if self._combined_df is None:
            raise RuntimeError("build_wall_mart_df() returned None")

        # item name and id operations
        self._combined_df = self.itemNameUtils.remove_items_matching_terms(self._combined_df, "item", self.exclude_items);
        self._combined_df["itemName_lemma"] = self._combined_df["item"].apply(self.itemNameUtils.lemmatize_item_name)
        self._combined_df["item"] = self._combined_df["item"].apply(ItemNameUtils.clean_item_name)
        self._combined_df = ItemNameUtils.strip_prefixes_from_column(self._combined_df ,"item", self.brand_prefixes);
        self._combined_df = self.groceryMLCore.canonicalize(self._combined_df)
        
        self._combined_df = self.itemNameUtils.create_item_ids(self._combined_df, allow_new_items=True)
        
             
        # synthetic_df = DataCreator.build_synthetic_rows_until(408, 24, "01/01/2020", "12/31/2020")
        # df = pd.concat([df, synthetic_df], ignore_index=True)
        # synthetic_df = self.create_synthetic_samples(df, "01-01-2023", "12-31-2023", 3)
        # df = pd.concat([df, synthetic_df], ignore_index=True)
        
        self._combined_df = self.groceryMLCore.create_didBuy_target_col(self._combined_df, "didBuy_target");
        self._combined_df = self.groceryMLCore.insert_negative_samples(self._combined_df);
        
        self._combined_df = TemporalFeatures.compute_days_since_last_purchase_for_item(self._combined_df)
        self._combined_df = TemporalFeatures.compute_expected_gap_ewma_feat(self._combined_df);
        self._combined_df = TemporalFeatures.compute_avg_days_between_item_purchases(self._combined_df)  
        self._combined_df = self.groceryMLCore.create_item_supply_level_feat(self._combined_df);
        
        self._combined_df["item_due_ratio_feat"] = TemporalFeatures.compute_item_due_ratio(self._combined_df)  
        
        TemporalFeatures.compute_recent_purchase_penalty(self._combined_df)        
        
        self._combined_df = self.groceryMLCore.add_item_total_purchase_count_feat(self._combined_df, "itemPurchaseCount_feat");     
        #self._combined_df = self.groceryMLCore.build_purchase_item_freq_cols(self._combined_df)
        
        # item level
        # self.
        # self.build_habit_frequency_for_training();
        
        # self.groceryMLCore.build_freq_ratios()

        self._combined_df = self._combined_df[self._combined_df["itemPurchaseCount_feat"] != 1].reset_index(drop=True)
        ######## trip level ######
        trip_df = self.build_trip_level_features(self._combined_df)
        if trip_df is None: 
            raise RuntimeError("build_trip_level_features() returned None")
        self._combined_df = self._combined_df.merge(trip_df, on="date", how="left")
                
        # df_weather = WeatherFeatures.BuildWeather(r"data\training\weather\VisualCrossing-70062 2000-01-01 to 2025-12-14.csv").reset_index()
        # self._combined_df = self._combined_df.merge(df_weather, on="date", how="left")
        
        #self.drop_rare_items()
        
        
        #df = df.drop(columns=["source"]) 
        #self.groceryMLCore.validate_no_empty_columns(self._combined_df)
        print("self._combined_df() done")
    
        ##self.groceryMLCore.export_df_to_excel_table(self._combined_df, "./combined_df", sheet_name="combined_df")
        #self.create_bulkAdjustedUrgencyRatio_for_training(df);
        # ============================================================
        # MERGE HABIT FEATURES
        # ============================================================
        # habit_df = build_habit_features(combined_df)
        # df = df.merge(habit_df, on="itemId",how="left")
        # df[["habitFrequency_feat", "habitSpan_feat", "habitDecay_feat"]] = (
        #     df[["habitFrequency_feat", "habitSpan_feat", "habitDecay_feat"]].fillna(0.0)
        # ) 

        return self._combined_df
    ###########################################################################################
 
        #def drop_rare_items(self):
        # df = self.drop_rare_items_with_zero_freq(df, "freq_7_feat")
        # df = self.drop_rare_items_with_zero_freq(df, "freq_15_feat")
        # df = self.drop_rare_items_with_zero_freq(df, "freq_30_feat")
        # df = self.drop_rare_items_with_zero_freq(df, "freq_90_feat")
    ###########################################################################################
    def create_synthetic_samples(self, df, startDate, stopDate, fuzzRangeDays=3):
        print("create_synthetic_samples()")
        df_sorted = df.sort_values(["itemId", "date"]).reset_index(drop=True)
    
        # take the LAST row per itemId so avgGap reflects learned history (not the initial 0)
        item_stats_df = df_sorted.groupby("itemId", as_index=False).tail(1)[
            ["itemId", "avgDaysBetweenItemPurchases_raw"]
        ]
    
        dfs = []
    
        for i in range(len(item_stats_df)):
            itemId = int(item_stats_df.iloc[i]["itemId"])
            avgGap = float(item_stats_df.iloc[i]["avgDaysBetweenItemPurchases_raw"])
    
            if avgGap <= 0:
                continue
    
            df_item = DataCreator.build_synthetic_rows_until(
                itemId=itemId,
                avgDaysBetweenPurchases=int(round(avgGap)),
                startDate=startDate,
                stopDate=stopDate,
                fuzzRangeDays=fuzzRangeDays
            )
    
            dfs.append(df_item)
    
        if len(dfs) == 0:
            return pd.DataFrame()
    
        return pd.concat(dfs, ignore_index=True)
    #
    ###########################################################################################
    def drop_rare_items_with_zero_freq(self, df, freq_col):
        mask = ~(
            (df["itemTotalPurchaseCount_feat"] == 1) &
            (df[freq_col] == 0)
        )
        return df[mask].copy()
    ###########################################################################################
    def build_trip_level_features(self, df):
        print("build_trip_level_features()");
        grouped_df = ( df[["date"]]
            .drop_duplicates()
            .sort_values("date")
            .reset_index(drop=True)
        )
    
        # grouped_df["daysSinceLastTrip_feat"] = TemporalFeatures.create_days_since_last_trip(grouped_df)
        # grouped_df["avgDaysBetweenTrips_feat"] = TemporalFeatures.compute_avg_days_between_trips(grouped_df)
        #
        # grouped_df["daysUntilNextHoliday_feat"]   = grouped_df["date"].apply(HolidayFeatures.compute_days_until_next_holiday)
        # grouped_df["daysSinceLastHoliday_feat"]   = grouped_df["date"].apply(HolidayFeatures.compute_days_since_last_holiday)
        # grouped_df["holidayProximityIndex_feat"]  = grouped_df["date"].apply(HolidayFeatures.compute_holiday_proximity_index)
        #
        grouped_df["daysUntilSchoolStart_feat"]   = grouped_df["date"].apply(SchoolFeatures.compute_days_until_school_start)
        grouped_df["daysUntilSchoolEnd_feat"]     = grouped_df["date"].apply(SchoolFeatures.compute_days_until_school_end)
        grouped_df["schoolSeasonIndex_feat"]      = grouped_df["date"].apply(SchoolFeatures.compute_school_season_index)
        #
        # grouped_df =  TemporalFeatures.add_dst_since_until_features(grouped_df);
        # TemporalFeatures.compute_trip_due_ratio(grouped_df);
        # grouped_df = TemporalFeatures.create_date_features(grouped_df)
        #
        return grouped_df;
    ###########################################################################################
   
    def build_and_compile_model(self, feat_cols_count, item_count, build_params):
          pass;
    ###########################################################################################
    def train_model(self, model, df, feature_cols, target_col, train_params):
          pass;
        
    ###########################################################################################

    def save_model(self, model, train_df, history, base_dir):
       """
       Save core artifacts directly into base_dir:
       - combined_df snapshot (parquet)   (pre-normalized training state)
       - model + weights
       - training history (json)
       """
       print(f"[save_model] starting artifact save → {base_dir}")
       
       # create base + model dirs in one shot
       modelDir = os.path.join(base_dir, "model")
       os.makedirs(modelDir, exist_ok=True)

       print(f"[save_model] writing train_df snapshot (parquet, pre-normalized)")
       train_df.to_parquet(os.path.join(modelDir, "combined_training_df_frozen.parquet"), compression="snappy")
       
       print("[save_model] writing training history json")
       self.groceryMLCorewrite_json(history.history, os.path.join(modelDir, "history.json"))
       
       print("[save_model] saving model directory files")
       model.save(modelDir)

       print("[save_model] saving model weights (separate file)")
       model.save_weights(os.path.join(modelDir, "weights.h5"))

       print(f"[save_model] all artifacts saved successfully → {modelDir}")
    ###########################################################################################
    def load_model(self, model_dir):
        """
        Loads a trained model + frozen combined_df snapshot from disk.
        Returns (model, combined_df_frozen).
        """
        print(f"[load_model_artifacts] loading artifacts from → {model_dir}")

        model_sub_dir = os.path.join(model_dir, "model")
        print(f"[load_model_artifacts] loading model → {model_sub_dir}")
        model = tf.keras.models.load_model(model_sub_dir)

        frozen_path = os.path.join(model_sub_dir, "combined_training_df_frozen.parquet")
        print(f"[load_model_artifacts] loading frozen combined_df → {frozen_path}")
        combined_df_frozen = pd.read_parquet(frozen_path)

        print("[load_model_artifacts] done")
        return model, combined_df_frozen
    ###########################################################################################

    def run_experiment(self, combined_df,  modelBuildParams, modelTrainParams, baseDir):
        
        exp_dir_path = self.create_experiment_dir(modelBuildParams, modelTrainParams, baseDir);
        
        self.tensorboard = self.create_tensorboard(f"./{exp_dir_path}/tensorflow/logs/")
        print(f"run_experiment()  baseDir: {exp_dir_path}  ");
        print(f"run_experiment()  when: {datetime.now()} params: {modelTrainParams}  ");
                
       
        feature_cols = self.get_normalized_feature_col_names(normalized_df);
        target_cols = [c for c in normalized_df.columns if c.endswith("_target")]
    
        if len(target_cols) != 1:
            raise ValueError("Exactly one target column is required")
        target_col = target_cols[0]
    
        feat_cols_count = len(feature_cols)
        item_count = int(normalized_df["itemId"].max()) + 1

        ## create model
        model = self.build_and_compile_model(feat_cols_count, item_count, modelBuildParams)
        ## train model
        history = self.train_model(model, normalized_df, feature_cols, target_col, modelTrainParams)

        # get test predictions
        pred_input = self.build_prediction_input(combined_df, pd.Timestamp.now(), norm_params)
        print("Running Model.Predict()");
        predictions = model.predict( [pred_input["x_features"], pred_input["x_item_idx"]])
    
        prediction_df = pred_input["prediction_df"]
        latest_rows_df = pred_input["latest_rows_df"]
        prediction_df.insert(3, "prediction",  predictions)    
        prediction_df = self.itemNameUtils.map_item_ids_to_names(prediction_df)
        prediction_df = prediction_df.sort_values("prediction", ascending=False).reset_index(drop=True)
        
        dataframes = {
            "latest_rows_df": latest_rows_df,
            "predictions": prediction_df,
            "normalized_df": normalized_df,
            #"combined_df": combined_df,
        }
        
        self.save_experiment(model, combined_df, dataframes, history,  modelBuildParams, modelTrainParams, exp_dir_path)
    ###########################################################################################
    def create_experiment_dir(self,  build_params, train_params, base_dir):
        name_parts = []

        if "embedding_dim" in build_params:
            name_parts.append(f"e{build_params['embedding_dim']}")

        if "layers" in build_params:
            layer_units = "-".join(str(layer["units"]) for layer in build_params["layers"])
            name_parts.append(f"l{layer_units}")

        if "epochs" in train_params:
            name_parts.append(f"ep{train_params['epochs']}")

        if "output_activation" in build_params:
            name_parts.append(f"oa_{build_params['output_activation']}")

        base_name = "__".join(name_parts) if name_parts else "exp_unlabeled"

        short_id = str(abs(hash(time.time())))[:6]
        exp_name = f"{base_name}__{short_id}"

        exp_dir_name = os.path.join(base_dir, exp_name)
        print(f"Creating dir: {exp_dir_name}")
        os.makedirs(exp_dir_name, exist_ok=True)

        #return exp_name, exp_dir
        return exp_dir_name
    ###########################################################################################
    def save_experiment(self, model, training_df,  extra_dataframes, history, build_params, train_params, exp_dir):
        name_parts = []

        # if "embedding_dim" in build_params:
        #     name_parts.append(f"e{build_params['embedding_dim']}")
        # # if "layers" in build_params:
        # #     hl = "-".join(str(x) for x in build_params["layers"])
        # #     name_parts.append(f"l{hl}")
        # if "epochs" in train_params:
        #     name_parts.append(f"ep{train_params['epochs']}")
        # if "output_activation" in build_params:
        #     name_parts.append(f"oa_{build_params['output_activation']}")
    
        # base_name = "__".join(name_parts) if name_parts else "exp_unlabeled"
        # short_id = str(abs(hash(time.time())))[:6]
        # exp_name = f"{base_name}__{short_id}"
        # exp_dir = os.path.join(base_dir, exp_name)      
        # print(f"Creating dir: {exp_dir}")
        # os.makedirs(exp_dir, exist_ok=True)
        
        print("Exporting extra_dataframes:")
        self.groceryMLCore.export_dataframes_with_exp_name(extra_dataframes, exp_dir)
        
        self.save_model(model, training_df, history, exp_dir)
        self.groceryMLCorewrite_json(build_params, os.path.join(exp_dir, "build_params.json"))
        self.groceryMLCorewrite_json(train_params, os.path.join(exp_dir, "train_params.json"))
        
        print("Saved experiment →", exp_dir)  
    ###########################################################################################
  
    def RunPredictionsOnly(self, combined_df_latest, model_dir, prediction_date):
        """
        Uses latest combined_df (with new receipts) to build prediction input,
        but restores frozen snapshot to compute normalization + filter unknown items.
        """
        print(f"RunPredictionsOnly() prediction_date: {prediction_date}")

   # def build_habit_frequency_for_training(self, df):
    #     print("build_habit_frequency_for_training()");
    #     df = didBuy_target"] == 1]
    #     latest_trip_date = df["date"].max()
    #     freq_map = self.compute_habit_frequency_map(df, latest_trip_date)
    #     df["itemPurchaseHabitFrequency_feat"] = df["itemId"].map(freq_map)
    ############################################################################################
    def recompute_habit_frequency_for_prediction_time(self, prediction_date: datetime):
        pass;
        # print("recompute_habit_frequency_for_prediction_time()");
        # df = df[df["didBuy_target"] == 1]
        # latest_trip_date = prediction_date
        # freq_map = self.compute_habit_frequency_map(df, latest_trip_date)
        # return freq_map
    ############################################################################################
    def compute_habit_frequency_map(self, filtered_df: pd.DataFrame, ref_date: datetime):
        print("compute_habit_frequency_map()");
        oldest_date = filtered_df["date"].min()
        days_span = (ref_date - oldest_date).days
        if days_span <= 0:
            return {}

        counts = filtered_df.groupby("itemId")["date"].count()
        freq_map = (counts / days_span).to_dict()
        return freq_map
    ############################################################################################         
    def build_prediction_input(self, prediction_date):
        print(f"build_prediction_input() prediction_date={prediction_date}")
    
        # build / refresh live source
        self.build_live_df()
    
        # latest PURCHASE per item only (from live_df)
        latest_rows_df = (
            self.live_df[self.live_df["didBuy_target"] == 1]
            .sort_values("date")
            .groupby("itemId")
            .tail(1)
            .copy()
            .reset_index(drop=True)
        )
    
        # force prediction date
        latest_rows_df["date"] = prediction_date
    
        # recompute days since last purchase using live_df as history
        tmp_df = pd.concat([self.live_df, latest_rows_df], ignore_index=True)
        tmp_df = TemporalFeatures.compute_days_since_last_purchase_for_item(tmp_df)
    
        latest_rows_df = (
            tmp_df.sort_values("date")
            .groupby("itemId")
            .tail(1)
            .reset_index(drop=True)
        )
    
        # carry forward avg days between purchases (from live_df)
        last_avg_vals = (
            self.live_df[self.live_df["didBuy_target"] == 1]
            .sort_values("date")
            .groupby("itemId")
            .tail(1)["avgDaysBetweenItemPurchases_raw"]
            .values
        )
        latest_rows_df["avgDaysBetweenItemPurchases_raw"] = last_avg_vals
    
        # derived item features
        latest_rows_df = self.groceryMLCore.create_item_supply_level_feat(latest_rows_df)
    
        # calendar features
        latest_rows_df["daysUntilSchoolStart_feat"] = SchoolFeatures.compute_days_until_school_start(prediction_date)
        latest_rows_df["daysUntilSchoolEnd_feat"] = SchoolFeatures.compute_days_until_school_end(prediction_date)
        latest_rows_df["schoolSeasonIndex_feat"] = SchoolFeatures.compute_school_season_index(prediction_date)
    
        # ensure no leakage
        if "didBuy_target" in latest_rows_df.columns:
            latest_rows_df.drop(columns=["didBuy_target"], inplace=True)
    
        # ensure itemId matches training expectations
        latest_rows_df["itemId"] = latest_rows_df["itemId"].astype("category")
    
        # remove non-feature column
        if "source" in latest_rows_df.columns:
            latest_rows_df.drop(columns=["source"], inplace=True)
    
        # build feature matrix INCLUDING itemId
        feature_cols = ["itemId"] + self.groceryMLCore.get_feature_col_names(latest_rows_df)
        X_pred = latest_rows_df[feature_cols]
    
        print("build_prediction_input() done")
    
        return { "prediction_df": latest_rows_df, "X_pred": X_pred, "feature_cols": feature_cols}
    #
    ###########################################################################################    
    

    ###########################################################################################
    # def build_freq_ratios(self, df):
    #     (
    #         df["freq7_over30_feat"],
    #         df["freq30_over365_feat"],
    #     ) = TemporalFeatures.compute_freq_ratios(
    #         df["freq_7_feat"],
    #         df["freq_30_feat"],
    #         df["freq_365_feat"],
    #     )
    # ###########################################################################################
        
 # def compute_bulkAdjustedUrgencyRatio_value(self, days_since, avg_between, bulk_flag, did_buy):
    #     """
    #     Compute one bulk-adjusted urgency ratio value.
    #     Shared by training + prediction. Caller decides did_buy.
    #     """
    #     if did_buy != 1:
    #         return 0.0

    #     bulk_factor_value = 2.5
    #     denominator = avg_between * (bulk_factor_value if bulk_flag == 1 else 1.0)

    #     if denominator == 0:
    #         return 0.0

    #     return days_since / denominator
    # ###########################################################################
   
    # def create_bulkAdjustedUrgencyRatio_for_training(self, df):
    #     """
    #     Create bulkAdjustedUrgencyRatio_feat using real didBuy_target values.
    #     """
    #     if "bulkFlag" not in df.columns:
    #         df["bulkAdjustedUrgencyRatio_feat"] = [0.0] * len(df)
    #         return dfFnor

    #     ratios = []
    #     for _, row in df.iterrows():
    #         ratios.append(self.compute_bulkAdjustedUrgencyRatio_value(
    #             row["daysSinceThisItemLastPurchased_log_feat"],
    #             row["avgDaysBetweenItemPurchases_log_feat"],
    #             row["bulkFlag"],
    #             row["didBuy_target"]      # <-- real 0/1
    #         ))

    #     df["bulkAdjustedUrgencyRatio_feat"] = ratios
    #     return df
    # ###########################################################################
    # def create_bulkAdjustedUrgencyRatio_for_prediction(self, df):
    #     """
    #     Create bulkAdjustedUrgencyRatio_feat assuming did_buy==1 for all rows.
    #     """
    #     if "bulkFlag" not in df.columns:
    #         df["bulkAdjustedUrgencyRatio_feat"] = [0.0] * len(df)
    #         return df

    #     ratios = []
    #     for _, row in df.iterrows():
    #         ratios.append(self.compute_bulkAdjustedUrgencyRatio_value(
    #             row["daysSinceThisItemLastPurchased_log_feat"],
    #             row["avgDaysBetweenItemPurchases_log_feat"],
    #             row["bulkFlag"],
    #             1                           # <-- always 1 at prediction time
    #         ))

    #     df["bulkAdjustedUrgencyRatio_feat"] = ratios
    #     return df
    # ###########################################################################



       # def build_habit_features(self, df, tau_days=120):
    #     df = df.copy()
    #     df["date"] = pd.to_datetime(df["date"])
    
    #     total_trips = df["date"].nunique()
    #     timeline_days = (df["date"].max() - df["date"].min()).days or 1
    
    #     rows = []
    
    #     for itemId, g in df.groupby("itemId"):
    #         buys = g[g["didBuy_target"] == 1]["date"]
    
    #         if len(buys) == 0:
    #             rows.append({
    #                 "itemId": itemId,
    #                 "habitFrequency_feat": 0.0,
    #                 "habitSpan_feat": 0.0,
    #                 "habitDecay_feat": 0.0,
    #             })
    #             continue
    
    #         first = buys.min()
    #         last = buys.max()
    
    #         habitFrequency = len(buys) / total_trips
    #         habitSpan = (last - first).days / timeline_days
    #         days_since_last = (df["date"].max() - last).days
    #         habitDecay = np.exp(-days_since_last / tau_days)
    
    #         rows.append({
    #             "itemId": itemId,
    #             "habitFrequency_feat": habitFrequency,
    #             "habitSpan_feat": habitSpan,
    #             "habitDecay_feat": habitDecay,
    #         })
    
    #     return pd.DataFrame(rows)
    # ###########################################################################################     
        
    # def compute_due_score(self, df, itemId=None, use_sigmoid=True, normalize=False, weights=None):
        
    #     if weights is None:
    #         weights = {
    #             "daysSinceThisItemLastPurchased_log_feat": 1.5,
    #             "freq_30_feat": 1.0,
    #             "freq_90_feat": 0.5
    #         }
    
    #     # --------------------------------------------------------
    #     # Optional itemId filter
    #     # --------------------------------------------------------
    #     if itemId is not None:
    #         df = df[df["itemId"] == itemId].copy()
    #     else:
    #         df = df.copy()
    
    #     # --------------------------------------------------------
    #     # RAW linear score (pre-normalization)
    #     # --------------------------------------------------------
    #     df["due_score_raw_feat"] = (
    #         weights["daysSinceThisItemLastPurchased_log_feat"] * df["daysSinceThisItemLastPurchased_log_feat"]
    #       + weights["freq_30_feat"]              * df["freq_30_feat"]
    #       + weights["freq_90_feat"]              * df["freq_90_feat"]
    #     )
    
    #     # --------------------------------------------------------
    #     # Final due_score
    #     # --------------------------------------------------------
    #     if use_sigmoid:
    #         df["due_score"] = 1 / (1 + np.exp(-df["due_score_raw"]))
    
    #     elif normalize:
    #         mean = df["due_score_raw"].mean()
    #         std  = df["due_score_raw"].std() or 1.0
    #         df["due_score"] = (df["due_score_raw"] - mean) / std
    
    #     else:
    #         df["due_score"] = df["due_score_raw"]
    
    #     return df
    ###########################################################################################
