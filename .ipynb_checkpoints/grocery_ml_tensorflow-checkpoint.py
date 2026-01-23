
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime
import asyncio
import json
import gc
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
import logging

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


### Domain
from excel_export_merger import ExcelExportMerger
from grocery_ml_core import GroceryMLCore
from school_features import SchoolFeatures
from weather_features import WeatherFeatures
from item_name_utils import ItemNameUtils
from item_id_mapper import ItemIdMapper
from temporal_features_2 import TemporalFeatures
from data_creator import DataCreator
from holiday_features import HolidayFeatures
from wallmart_rcpt_parser import WallmartRecptParser
from winn_dixie_recpt_parser import WinnDixieRecptParser 
from hidden_layer_param_builder import HiddenLayerParamSetBuilder
from weather_service import NwsWeatherService;
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class GroceryML:

    expNameParts: str = None
    _combined_df: pd.DataFrame = None 
    _training_df: pd.DataFrame = None
    _live_df: pd.DataFrame = None
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
        self.itemIdMapper = ItemIdMapper();
        self.itemNameUtils = ItemNameUtils();
        self.weatherService = NwsWeatherService();
        self.groceryMLCore = GroceryMLCore();
        self.excelMerger = ExcelExportMerger();
    ###########################################################################################        
    def build_training_df(self):
        print("Building Training DF: Start")
        self.training_df =  self._build_combined_df(self.trainingSources)
        # ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # self.groceryMLCore.export_dataframe_to_csv(self.training_df, f"training_df_{ts}");
        print("Building Training DF: Done")
    ###########################################################################################        
    def build_live_df(self):
        print("Building Live DF: Start")
        self.live_df = self._build_combined_df(self.liveSources)
        # ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # self.groceryMLCore.export_dataframe_to_csv(self.live_df, f"live_df_{ts}");
        print("Building Live DF: Done")
    ###########################################################################################        
    def _build_combined_df(self, data_sources: Dict):
     
        print(f"_build_combined_df()")
        self._build_sources(data_sources);
        # item name and id operations
        self._combined_df = self.groceryMLCore.normalize_item_names(self._combined_df);
        self._combined_df = self.itemIdMapper.create_item_ids(self._combined_df)
                    
        # synthetic_df = DataCreator.build_synthetic_rows_until(408, 24, "01/01/2020", "12/31/2020")
        # df = pd.concat([df, synthetic_df], ignore_index=True)
        # synthetic_df = self.create_synthetic_samples(df, "01-01-2023", "12-31-2023", 3)
        # df = pd.concat([df, synthetic_df], ignore_index=True)

        self._combined_df = self.groceryMLCore.create_didBuy_target_col(self._combined_df, "didBuy_target");

        # neg samples
        self._combined_df = self.groceryMLCore.insert_negative_samples(self._combined_df);
        self._combined_df = self.groceryMLCore.create_full_calendar_and_merge(self._combined_df);
                
        ######## Item level ########
        self._combined_df = TemporalFeatures.compute_days_since_last_purchase_for_item(self._combined_df,"daysSinceThisItemLastPurchased_raw")      
        self._combined_df["daysSinceThisItemLastPurchased_log_feat"] = self.groceryMLCore.log_feature(self._combined_df["daysSinceThisItemLastPurchased_raw"])
        #
        self._combined_df = TemporalFeatures.compute_expected_gap_ewma_feat(self._combined_df);
        self._combined_df = TemporalFeatures.compute_avg_days_between_item_purchases(self._combined_df, "avgDaysBetweenItemPurchases_feat")
         # self._combined_df = TemporalFeatures.compute_avg_days_between_item_purchases(self._combined_df)  
        
        # item supply level
        self._combined_df = self.groceryMLCore.create_item_supply_level_feat(self._combined_df);
        # item due ratio
        self._combined_df["item_due_ratio_feat"] = TemporalFeatures.compute_item_due_ratio(self._combined_df)  
        
        # item purchase count
        self._combined_df = self.groceryMLCore.add_item_total_purchase_count_feat(self._combined_df, "itemPurchaseCount_raw");     
        # self._combined_df["itemPurchaseCount_log_feat"] = self.groceryMLCore.log_feature(self._combined_df["itemPurchaseCount_raw"]);
        
        #TemporalFeatures.compute_recent_purchase_penalty(self._combined_df)                
        #self._combined_df = self.groceryMLCore.build_purchase_item_freq_cols(self._combined_df)
        # self.groceryMLCore.build_freq_ratios()
             
        ######## trip level ######
        self._combined_df = self._build_trip_level_feats(self._combined_df);
        self._combined_df = self.groceryMLCore.build_trip_interveral_feautres(self._combined_df)
        #
        self._combined_df = self.groceryMLCore.drop_rare_purchases(self._combined_df)
        
        self._combined_df = TemporalFeatures.create_date_features(self._combined_df);

        self.groceryMLCore.validate_no_empty_columns(self._combined_df, ["qty"])
        print("self._build_combined_df() done")
        # ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # self.groceryMLCore.export_df_to_excel_table(self._combined_df, f"./combined_df_debug-{ts}", sheet_name="combined_df")
        return self._combined_df;
    ###########################################################################################
    def _build_trip_level_feats(self, df):

        print("_build_trip_level_feats()")
        df = self.groceryMLCore.build_school_schedule_features(df);
        df = self.groceryMLCore.build_holiday_features(df);
        
        df["daysUntilSchoolStart_log_feat"] = self.groceryMLCore.log_feature(df["daysUntilSchoolStart_raw"])
        df["daysUntilSchoolEnd_log_feat"] = self.groceryMLCore.log_feature(df["daysUntilSchoolEnd_raw"])
        # df["daysUntilNextHoliday_log_feat"] = self.groceryMLCore.log_feature(df["daysUntilNextHoliday_raw"])
        # df["daysSinceLastHoliday_log_feat"] = self.groceryMLCore.log_feature(df["daysSinceLastHoliday_raw"])

        df["isDayLightSavingsTime_feat"] = TemporalFeatures.is_dst_series(df["date"])
        
        return df;
    ###########################################################################################
    def _build_sources(self, data_sources: Dict):
        print("_build_sources()");
        winndixie_df = self.groceryMLCore.build_winn_dixie_df(data_sources.get("winndixie"));
        winndixie_add_txt_rpts_df =  self.groceryMLCore.build_winn_dixie_additional_text_rcpts_df(data_sources.get("winndixieAdditional"))
        winndixie_df = pd.concat([winndixie_df, winndixie_add_txt_rpts_df], ignore_index=True)
        #winndixie_df["isBulkVendor_feat"] = 0;        
        #       
        wallmart_df = WallmartRecptParser.build_wall_mart_df(data_sources.get("walmart"));
        #wallmart_df["isBulkVendor_feat"] = 0;        
        
        if winndixie_df is None:
            raise RuntimeError("build_winn_dixie_df() returned None")
        if winndixie_add_txt_rpts_df is None:
            raise RuntimeError("build_winn_dixie_additional_text_rcpts_df() returned None")
        #if wallmart_df is None:
        #    raise RuntimeError("build_wall_mart_df() returned None")
        
        self._combined_df = pd.concat([winndixie_df, wallmart_df[["date", "item", "source", "qty"]]],ignore_index=True)        
        #self._combined_df = winndixie_df[["date", "item", "source","qty"]].reset_index(drop=True)

    ########################################################################################### 
    def create_synthetic_samples(self, df, startDate, stopDate, fuzzRangeDays=3):
        print("create_synthetic_samples()")
          
        df_sorted = df.sort_values(["itemId", "date"]).reset_index(drop=True)
    
        # take the LAST row per itemId so avgGap reflects learned history (not the initial 0)
        item_stats_df = df_sorted.groupby("itemId", as_index=False).tail(1)[
            ["itemId", "avgDaysBetweenItemPurchases_feat"]
        ]
    
        dfs = []
    
        for i in range(len(item_stats_df)):
            itemId = int(item_stats_df.iloc[i]["itemId"])
            avgGap = float(item_stats_df.iloc[i]["avgDaysBetweenItemPurchases_feat"])
    
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
    ########################################################################################### 

    def is_binary_column(self, df, colName: str):
        col = df[colName]
        if col.dtype == bool: return True
        unique_vals = col.dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
            return True

        return False
    ###########################################################################################
    def get_normalized_feature_col_names(self, df):
        """Returns normalized feature columns used for model input."""
        return [c for c in df.columns if c.endswith("_norm") and not c.endswith("_raw_norm")]
    ###########################################################################################
    def get_target_col(self, df):
        """Returns the single target column. Throws if missing or ambiguous."""
        target_cols = [c for c in df.columns if c.endswith("_target")]
        if len(target_cols) != 1:
            raise ValueError(f"Expected exactly one target column, found: {target_cols}")
        return target_cols[0]
    ###########################################################################################
    def fit_normalization_params(self, df):
        params = {}
        feature_cols = self.groceryMLCore.get_feature_col_names(df)
        cyc_cols = [c for c in feature_cols if c.endswith("_cyc_feat")]
        num_cols = [c for c in feature_cols if c not in cyc_cols]
    
        for col in num_cols:
            if self.is_binary_column(df, col):
                continue
    
            params[col] = {
                "mean": df[col].mean(),
                "std": df[col].std()
            }
    
        for col in cyc_cols:
            params[col] = {
                "period": TemporalFeatures.get_period_for_column(col)
            }
    
        return params
    #
    ###########################################################################################
    def normalize_features(self, df, norm_params):
        print("normalize_features()")
    
        normalized_df = df.copy()
    
        for col, cfg in norm_params.items():
    
            if col.endswith("_cyc_feat"):
                sin_col, cos_col = TemporalFeatures.encode_sin_cos(
                    df[col],
                    cfg["period"]
                )
    
                normalized_df[f"{col}_sin_norm"] = sin_col
                normalized_df[f"{col}_cos_norm"] = cos_col
                normalized_df.drop(columns=[col], inplace=True)
    
            else:
                mean_val = cfg["mean"]
                std_val = cfg["std"]
                norm_col = col.replace("_feat", "_norm")
    
                if std_val == 0:
                    normalized_df[norm_col] = 0.0
                else:
                    normalized_df[norm_col] = (df[col] - mean_val) / std_val
    
                normalized_df.drop(columns=[col], inplace=True)
    
        return normalized_df
    ###########################################################################################
    def get_model_feature_col_names(self, df):
        """
        Returns all feature columns used for model input:
        - normalized continuous features (*_norm)
        - binary features (*_feat, not normalized)
        """
        norm_cols = [c for c in df.columns if c.endswith("_norm")]
    
        binary_feat_cols = []
        for c in df.columns:
            if c.endswith("_feat") and self.is_binary_column(df, c):
                binary_feat_cols.append(c)
    
        return norm_cols + binary_feat_cols
    ##########################################################################################
    def build_and_compile_model(self, feat_cols_count, item_count, build_params):

        num_in = layers.Input(shape=(feat_cols_count,))
        item_in = layers.Input(shape=(), dtype="int32")
    
        emb = layers.Embedding(
            input_dim=item_count,
            output_dim=build_params["embedding_dim"],
            name="item_embedding"       # <-- REQUIRED FOR TENSORBOARD PROJECTOR
        )(item_in)
    
        x = layers.Concatenate()([num_in, layers.Flatten()(emb)])

        for spec in build_params["layers"]:
            x = layers.Dense(spec["units"], activation=spec["activation"])(x)

        outputLayer = layers.Dense(1, activation=build_params["output_activation"])(x)

        model = models.Model(inputs=[num_in, item_in], outputs=outputLayer)
    
        optimizer_name = build_params.get("optimizer", "adam")
        learning_rate = build_params.get("learning_rate")
    
        if optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == "adamw":
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
        model.compile(
            optimizer=optimizer,
            loss=build_params.get("loss"),
            metrics=build_params.get("metrics")
        )
    
        return model
    ###########################################################################################
    def create_tensorboard(self, log_dir):

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            embeddings_freq=1,
            embeddings_metadata=f"{log_dir}/embeddingslabels.tsv"
        )
        return tensorboard;
    ###########################################################################################
    def train_model(self, model, df, feature_cols, target_col, train_params):
        print("train_model()");
        callbacks = []
        x_feat = df[feature_cols].to_numpy(np.float32)
        x_item = df["itemId"].to_numpy(np.int32)
        y = df[target_col].to_numpy(np.float32)
    
        x_feat_tr, x_feat_te, x_item_tr, x_item_te, y_tr, y_te = train_test_split(
            x_feat, x_item, y, test_size=0.2, random_state=42
        )

        callbacks.append(tf.keras.callbacks.EarlyStopping( monitor="val_loss", patience=5, restore_best_weights=True ))
        callbacks.append(self.tensorboard)
        
        history = model.fit(
            [x_feat_tr, x_item_tr],
            y_tr,
            validation_split=0.1,
            epochs=train_params["epochs"],
            batch_size=train_params.get("batch_size", 32),
            verbose=0,
            callbacks=callbacks 
        )
    
        return history
    ###########################################################################################
    def save_model(self, model, training_df, history, base_dir):
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

       print(f"[save_model] writing training_df snapshot (parquet, pre-normalized)")
       training_df.to_parquet(os.path.join(modelDir, "training_df_frozen.parquet"), compression="snappy")
       #
       print("[save_model] writing training history json")
       self.groceryMLCore.write_json(history.history, os.path.join(modelDir, "history.json"))       
       #
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

        frozen_path = os.path.join(model_sub_dir, "training_df_frozen.parquet")
        print(f"[load_model_artifacts] loading frozen combined_df → {frozen_path}")
        combined_df_frozen = pd.read_parquet(frozen_path)

        print("[load_model_artifacts] done")
        return model, combined_df_frozen
    ###########################################################################################   
    def run_experiment_with_consecutive_predictions(self, training_df, modelBuildParams, modelTrainParams, baseDir, start_date, days):
        self.expNameParts = self.create_exp_name_parts(modelBuildParams, modelTrainParams)
        exp_dir_path = self.create_experiment_dir(self.expNameParts, baseDir)
        self.tensorboard = self.create_tensorboard(f"{exp_dir_path}/tensorflow/logs/")
        print(f"run_experiment_with_consecutive_predictions() exp_dir: {exp_dir_path}")
        print(f"when: {datetime.now()} params: {modelTrainParams}")
    
        target_cols = self.get_target_col(training_df)
        norm_params = self.fit_normalization_params(training_df)
        normalized_training_df = self.normalize_features(training_df, norm_params)
        model_feature_cols = self.get_model_feature_col_names(normalized_training_df)
        model = self.build_and_compile_model(
            len(model_feature_cols),
            int(normalized_training_df["itemId"].max()) + 1,
            modelBuildParams
        )
        history = self.train_model(
            model,
            normalized_training_df,
            model_feature_cols,
            target_cols,
            modelTrainParams
        )
    
        prediction_results = []
        for offset in range(days):
            prediction_date = start_date + pd.Timedelta(days=offset)
            artifacts = self.build_prediction_input(prediction_date, norm_params)
            y = model.predict([artifacts["x_features"], artifacts["x_item_idx"]])
            df = artifacts["normalized_latest_rows_df"]
            # df.insert(0, "prediction_date", prediction_date)
            df.insert(3, "readyToBuy_probability", y)
            df = self.itemIdMapper.map_item_ids_to_names(
                df.sort_values("readyToBuy_probability", ascending=False).reset_index(drop=True)
            )
            prediction_results.append(df)
    
        all_predictions_df = pd.concat(prediction_results, axis=0, ignore_index=True)
        #
    
        self.last_val_mse = history.history.get("val_mse", [None])[-1]
        self.last_val_auc = history.history.get("val_auc", [None])[-1]
        self.last_val_mae = history.history.get("val_mae", [None])[-1]
    
        self.save_experiment(model,training_df, {
                "normalized_training_df": normalized_training_df,
                "consecutive_predictions": all_predictions_df
            },history,modelBuildParams,modelTrainParams,exp_dir_path)
    ############################################################################################
    def run_experiment(self, training_df,  modelBuildParams, modelTrainParams, baseDir):
        
        self.expNameParts = self.create_exp_name_parts(modelBuildParams, modelTrainParams);
        exp_dir_path = self.create_experiment_dir(self.expNameParts, baseDir);
        
        self.tensorboard = self.create_tensorboard(f"{exp_dir_path}/tensorflow/logs/")
        print(f"run_experiment()  exp_dir: {exp_dir_path}  ");
        print(f"run_experiment()  when: {datetime.now()} params: {modelTrainParams}  ");
        
        #training_df = self.groceryMLCore.drop_raw_columns(training_df)
        #######################################################
        #### normalize training_df, build and train model #####
        target_cols = self.get_target_col(training_df);
        norm_params = self.fit_normalization_params(training_df)
        normalized_training_df = self.normalize_features(training_df, norm_params)       
        # normalized_feature_cols = self.get_normalized_feature_col_names(normalized_training_df);
        model_feature_cols = self.get_model_feature_col_names(normalized_training_df)
        normalized_feature_cols_count = len(model_feature_cols)
        item_count = int(normalized_training_df["itemId"].max()) + 1
        ## create model
        model = self.build_and_compile_model(normalized_feature_cols_count, item_count, modelBuildParams)
        ## train model
        history = self.train_model(model, normalized_training_df, model_feature_cols, target_cols, modelTrainParams)
        #################################################
        
        #################################################
        ###### test predictions  ########################
        prediction_time_artifacts = self.build_prediction_input(pd.Timestamp.now(), norm_params)
        print("Running Model.Predict()");
        prediction_values_col = model.predict( [prediction_time_artifacts["x_features"], prediction_time_artifacts["x_item_idx"]])
        prediction_df = prediction_time_artifacts["normalized_latest_rows_df"]
        ### raw_latest_rows_df = prediction_time_artifacts["raw_latest_rows_df"]
        prediction_df.insert(3, "readyToBuy_proabability",  prediction_values_col)
        prediction_df = prediction_df.sort_values("readyToBuy_proabability", ascending=False).reset_index(drop=True)
        prediction_df = self.itemIdMapper.map_item_ids_to_names(prediction_df)
        ##############################
        dataframes = {
            "normalized_training_df": normalized_training_df,
            ##"raw_latest_rows_df": raw_latest_rows_df,
            "predictions": prediction_df,
        }
        self.last_val_mse = history.history.get("val_mse", [None])[-1]
        self.last_val_auc = history.history.get("val_auc", [None])[-1]
        self.last_val_mae = history.history.get("val_mae", [None])[-1]
        # self.excelMerger.add_dataframe(prediction_df, sheet_name = exp_name_parts, output_dir =  baseDir)
        self.save_experiment(model, training_df, dataframes, history,  modelBuildParams, modelTrainParams, exp_dir_path)
    ###########################################################################################
    def build_prediction_input(self, prediction_date, norm_params):
        print(f"build_prediction_input() prediction_date={prediction_date}")
        self.build_live_df()
        # ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # self.groceryMLCore.export_df_to_excel_table(self.live_df, f"./live_df-{ts}") 
        raw_latest_rows_df = self._build_latest_rows_df(self.live_df)
        raw_latest_rows_df["date"] = prediction_date
        
        ### ITEM level  daysSinceThisItemLastPurchased
        tmp_df = pd.concat([self.live_df, raw_latest_rows_df], ignore_index=True)
        tmp_df = TemporalFeatures.compute_days_since_last_purchase_for_item(tmp_df, "daysSinceThisItemLastPurchased_raw")
        raw_latest_rows_df = tmp_df.sort_values("date").groupby("itemId").tail(1).reset_index(drop=True)
        raw_latest_rows_df["daysSinceThisItemLastPurchased_log_feat"] = self.groceryMLCore.log_feature(raw_latest_rows_df["daysSinceThisItemLastPurchased_raw"])
        raw_latest_rows_df = TemporalFeatures.compute_avg_days_between_item_purchases(raw_latest_rows_df, "avgDaysBetweenItemPurchases_feat")
        
        ### ITEM level
        raw_latest_rows_df = self.groceryMLCore.create_item_supply_level_feat(raw_latest_rows_df)
        raw_latest_rows_df["item_due_ratio_feat"] = TemporalFeatures.compute_item_due_ratio(raw_latest_rows_df)

        # TRIP level
        raw_latest_rows_df = self._build_trip_level_feats(raw_latest_rows_df );
        
        raw_latest_rows_df["daysSinceLastTrip_raw"] = TemporalFeatures.compute_days_since_last_trip_value(self.live_df, prediction_date)
        raw_latest_rows_df["avgDaysBetweenTrips_feat"] = self.live_df.sort_values("date").tail(1)["avgDaysBetweenTrips_feat"].iloc[0]
    
        raw_latest_rows_df = TemporalFeatures.create_date_features(raw_latest_rows_df);
        raw_latest_rows_df["itemId"] = raw_latest_rows_df["itemId"].astype("category")
        raw_latest_rows_df.drop(columns=["source", "didBuy_target", "qty"], inplace=True)
        
        normalized_latest_rows_df = self.normalize_features(raw_latest_rows_df, norm_params)
        model_feature_cols = self.get_model_feature_col_names(normalized_latest_rows_df)
        x_features = normalized_latest_rows_df[model_feature_cols].to_numpy(np.float32)
        x_item_idx = normalized_latest_rows_df["itemId"].to_numpy(np.int32)
        return {"normalized_latest_rows_df": normalized_latest_rows_df, "x_features": x_features, "x_item_idx": x_item_idx}
    ###########################################################################################
    def save_experiment(self, model, training_df, extra_dataframes, history, build_params, train_params, exp_dir):
        name_parts = []

        print("Exporting extra_dataframes:")
        self.export_dataframes_to_excel(extra_dataframes, exp_dir, self.expNameParts)
        # TODO: self.export_dataframes_to_csv(extra_dataframes, exp_dir, expNameParts)
               
        self.save_model(model, training_df, history, exp_dir)
        self.groceryMLCore.write_json(build_params, os.path.join(exp_dir, "build_params.json"))
        self.groceryMLCore.write_json(train_params, os.path.join(exp_dir, "train_params.json"))
        
        print("Saved experiment →", exp_dir)  
    ###########################################################################################    
    def create_exp_name_parts(self, build_params, train_params):
        """
        Build the full experiment name, including a short unique suffix.
        """
        name_parts = []

        if "embedding_dim" in build_params:
            name_parts.append(f"e{build_params['embedding_dim']}")
        if "layers" in build_params:
            layer_units = "-".join(str(layer["units"]) for layer in build_params["layers"])
            name_parts.append(f"l{layer_units}")
        if "epochs" in train_params:
            name_parts.append(f"ep{train_params['epochs']}")
        if "output_activation" in build_params:
            oa = str(build_params["output_activation"])[:3]
            name_parts.append(oa)
        if "learning_rate" in train_params:
            name_parts.append(f"lr{train_params['learning_rate']}")
        
        base_name = "_".join(name_parts) if name_parts else "exp_unlabeled"
        short_id = str(abs(hash(time.time())))[:3]
        full_name = f"{base_name}_{short_id}"
        if len(full_name) > 31: full_name = full_name[:31]
        return full_name
    ###########################################################################################    
    def create_experiment_dir(self, exp_name, base_dir):
        exp_dir_name = os.path.join(base_dir, exp_name)
        print(f"Creating dir: {exp_dir_name}")
        os.makedirs(exp_dir_name, exist_ok=True)
        return exp_dir_name
    ###########################################################################################
    def export_dataframes_to_excel(self, dataframes, path, nameParts):
        print("grocery_ml_tensorflow.export_dataframes_to_excel()");
        for name, df in dataframes.items():
            base = os.path.join(path, f"{name}-{nameParts}")
            self.groceryMLCore.export_df_to_excel_table(df, base, sheet_name=f"{nameParts}")
    ########################################################################################### 
    def RunPredictionsOnly(self, model_dir, prediction_date):
        """
        Uses latest combined_df (with new receipts) to build prediction input,
        but restores frozen snapshot to compute normalization + filter unknown items.
        """
        print(f"RunPredictionsOnly() prediction_date: {prediction_date}")

        # load model + frozen snapshot
        model, combined_df_frozen = self.load_model(model_dir)

        print("computing normalization parameters from frozen snapshot")
        norm_params = self.fit_normalization_params(combined_df_frozen)

        print("building prediction input (latest receipts)")
        pred_input = self.build_prediction_input(prediction_date, norm_params)

        print("filtering unseen itemIds")
        known_ids = set(combined_df_frozen["itemId"].unique())
        mask = pred_input["prediction_df"]["itemId"].isin(known_ids)

        pred_input["prediction_df"] = pred_input["prediction_df"][mask].reset_index(drop=True)
        pred_input["x_item_idx"]     = pred_input["x_item_idx"][mask]
        pred_input["x_features"]     = pred_input["x_features"][mask]

        print(f"kept {mask.sum()} rows out of {mask.size}")

        print("Running Model.Predict()")
        predictions = model.predict([pred_input["x_features"], pred_input["x_item_idx"]])

        prediction_df = pred_input["prediction_df"].copy()
        prediction_df["prediction"] = predictions
        prediction_df = self.itemIdMapper.map_item_ids_to_names(prediction_df)
        prediction_df = prediction_df.sort_values("prediction", ascending=False).reset_index(drop=True)

        return prediction_df
    ###########################################################################################
    def _build_latest_rows_df(self, sourcedf):
         # latest PURCHASE per item only 
        print("_build_latest_rows_df: start")
        latest_rows_df = (
            sourcedf[sourcedf["didBuy_target"] == 1]
            .sort_values("date")
            .groupby("itemId")
            .tail(1)
            .copy()
            .reset_index(drop=True)
        )
        print("_build_latest_rows_df: done")
        return latest_rows_df;
    ##########################################################################################
    def build_wit_input_df(self, model_dir):
        print("build_wit_input_df()")
        model, combined_df_frozen = self.load_model(model_dir)
    
        print("fitting normalization params from frozen snapshot...")
        norm_params = self.fit_normalization_params(combined_df_frozen)
        df_norm = self.normalize_features(combined_df_frozen.copy(), norm_params)
        
        feature_cols = self.get_normalized_feature_col_names(df_norm);
        x_features = df_norm[feature_cols].to_numpy(np.float32)
        x_item_idx = df_norm["itemId"].to_numpy(np.int32)
    
        # --- run predictions on ALL rows
        print("running model.predict on entire normalized dataset...")
        preds = model.predict([x_features, x_item_idx], verbose=0).flatten()
    
        # --- attach predictions
        df_norm["prediction"] = preds
    
        # --- rename ground truth if needed
        # if "didBuy_target" in df_norm.columns:
        #     df_norm = df_norm.rename(columns={"didBuy_target": "target"})
    
        # --- drop non-feature non-required cols (optional)
        # drop_cols = ["source"]
        # df_norm = df_norm.drop(columns=[c for c in drop_cols if c in df_norm.columns])
    
        print("build_wit_input_df() done")
        return df_norm
    ##########################################################################################
   



    