
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

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


### Domain
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
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


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
        
  def _build_combined_df(self, data_sources: Dict):
     
        print(f"build_combined_df()")
        self._build_sources(data_sources);
        # item name and id operations
        self._normalize_item_names();
        self._combined_df = self.itemNameUtils.create_item_ids(self._combined_df, allow_new_items=True)
                    
        # synthetic_df = DataCreator.build_synthetic_rows_until(408, 24, "01/01/2020", "12/31/2020")
        # df = pd.concat([df, synthetic_df], ignore_index=True)
        # synthetic_df = self.create_synthetic_samples(df, "01-01-2023", "12-31-2023", 3)
        # df = pd.concat([df, synthetic_df], ignore_index=True)

        self._combined_df = self.groceryMLCore.create_didBuy_target_col(self._combined_df, "didBuy_target");
        ######## Item level ########
        # neg samples
        self._combined_df = self.groceryMLCore.insert_negative_samples(self._combined_df);
        # item purchase 
        self._combined_df = TemporalFeatures.compute_days_since_last_purchase_for_item(self._combined_df)
        self._combined_df = TemporalFeatures.compute_expected_gap_ewma_feat(self._combined_df);
        self._combined_df = TemporalFeatures.compute_avg_days_between_item_purchases(self._combined_df)  
        # item supply level
        self._combined_df = self.groceryMLCore.create_item_supply_level_feat(self._combined_df);
        # item due ratio
        self._combined_df["item_due_ratio_feat"] = TemporalFeatures.compute_item_due_ratio(self._combined_df)  
        # item purchase count
        self._combined_df = self.groceryMLCore.add_item_total_purchase_count_feat(self._combined_df, "itemPurchaseCount_raw");     
        
        #TemporalFeatures.compute_recent_purchase_penalty(self._combined_df)                
        #self._combined_df = self.groceryMLCore.build_purchase_item_freq_cols(self._combined_df)
        # self.groceryMLCore.build_freq_ratios()
             
        self._combined_df = self._combined_df[self._combined_df["itemPurchaseCount_raw"] != 1].reset_index(drop=True)
        #
        ######## trip level ######
        self._build_trip_level_feats();
        #
        self._drop_rare_purchases()
        #df = df.drop(columns=["source"]) 
        self.groceryMLCore.validate_no_empty_columns(self._combined_df)
        print("self._build_combined_df() done")
    
        self.groceryMLCore.export_df_to_excel_table(self._combined_df, "./combined_df_debug", sheet_name="combined_df")
        return self._combined_df
    ###########################################################################################
    def _build_trip_level_feats(self):
        
        holiday_df = self.groceryMLCore._build_holiday_features(self._combined_df);
        trip_df = self.groceryMLCore._build_trip_interveral_feautres(self._combined_df);
        school_df = self.groceryMLCore._build_school_schedule_features(self._combined_df);
        
        if holiday_df is None: 
            raise RuntimeError("holiday_df is nul")
        if trip_df is None: 
            raise RuntimeError("trip_df is nul")
        if school_df is None: 
            raise RuntimeError("school_df is nul")
            
        self._combined_df = self._combined_df.merge(holiday_df, on="date", how="left")
        self._combined_df = self._combined_df.merge(trip_df, on="date", how="left")
        self._combined_df = self._combined_df.merge(school_df, on="date", how="left")
        # df_weather = WeatherFeatures.BuildWeather(r"data\training\weather\VisualCrossing-70062 2000-01-01 to 2025-12-14.csv").reset_index()
        # self._combined_df = self._combined_df.merge(df_weather, on="date", how="left")
    ###########################################################################################
    def _build_sources(self, data_sources: Dict):
        print("_build_sources()");
        winndixie_df = self.groceryMLCore.build_winn_dixie_df(data_sources.get("winndixie"));
        winndixie_add_txt_rpts_df =  self.groceryMLCore.build_winn_dixie_additional_text_rcpts_df(data_sources.get("winndixieAdditional"))
        winndixie_df = pd.concat([winndixie_df, winndixie_add_txt_rpts_df], ignore_index=True)
        #       
        #wallmart_df = WallmartRecptParser.build_wall_mart_df(data_sources.get("walmart"));
        
        if winndixie_df is None:
            raise RuntimeError("build_winn_dixie_df() returned None")
        if winndixie_add_txt_rpts_df is None:
            raise RuntimeError("build_winn_dixie_additional_text_rcpts_df() returned None")
        #if wallmart_df is None:
        #    raise RuntimeError("build_wall_mart_df() returned None")
        
        # self._combined_df = pd.concat([winndixie_df, wallmart_df[["date", "item", "source"]]],ignore_index=True)        
        self._combined_df = winndixie_df[["date", "item", "source",  "cashier_raw"]].reset_index(drop=True)

    ###########################################################################################
    def _normalize_item_names(self):
        
        self._combined_df = self.itemNameUtils.remove_items_matching_terms(self._combined_df, "item", self.exclude_items);
        # self._combined_df["itemName_lemma"] = self._combined_df["item"].apply(self.itemNameUtils.lemmatize_item_name)
        self._combined_df = ItemNameUtils.strip_prefixes_from_column(self._combined_df ,"item", self.brand_prefixes);
        self._combined_df["item"] = self._combined_df["item"].apply(ItemNameUtils.clean_item_name)
        self._combined_df = self.groceryMLCore.canonicalize(self._combined_df)
    ###########################################################################################
        #def drop_rare_items(self):
        # df = self.drop_rare_items_with_zero_freq(df, "freq_7_feat")
        # df = self.drop_rare_items_with_zero_freq(df, "freq_15_feat")
        # df = self.drop_rare_items_with_zero_freq(df, "freq_30_feat")
        # df = self.drop_rare_items_with_zero_freq(df, "freq_90_feat")
    ###########################################################################################
    def _drop_rare_purchases(self):
        print("_drop_rare_purchases()")
        self._combined_df = self._combined_df[self._combined_df["itemPurchaseCount_raw"] != 1].reset_index(drop=True)
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
    def build_freq_ratios(self):
        (
            self.combined_df["freq7_over30_feat"],
            self.combined_df["freq30_over365_feat"],
        ) = TemporalFeatures.compute_freq_ratios(
            self.combined_df["freq_7_feat"],
            self.combined_df["freq_30_feat"],
            self.combined_df["freq_365_feat"],
        )
    ########################################################################################### 
    
    def fit_normalization_params(self, df):
        params = {}
        feature_cols = self.get_feature_col_names(df);
        cyc_cols = [c for c in feature_cols if c.endswith("_cyc_feat")]
        num_cols = [c for c in feature_cols if c not in cyc_cols]
        
        for col in num_cols:
            params[col] = {"mean": df[col].mean(),"std": df[col].std()}
    
        for col in cyc_cols:
            params[col] = {"period": TemporalFeatures.get_period_for_column(col)}
    
        return params
    ###########################################################################################
    def normalize_features(self, combined_df, norm_params):
        print("normalize_features()");
        normalized_df = combined_df.copy()
        for col, cfg in norm_params.items():
            if col.endswith("_cyc_feat"):
                sin_col, cos_col = TemporalFeatures.encode_sin_cos(combined_df[col], cfg["period"])
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
                    normalized_df[norm_col] = (combined_df[col] - mean_val) / std_val
    
                normalized_df.drop(columns=[col], inplace=True)
    
        return normalized_df
    ###########################################################################################    
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
        
        x_feat = df[feature_cols].to_numpy(np.float32)
        x_item = df["itemId"].to_numpy(np.int32)
        y = df[target_col].to_numpy(np.float32)
    
        x_feat_tr, x_feat_te, x_item_tr, x_item_te, y_tr, y_te = train_test_split(
            x_feat, x_item, y, test_size=0.2, random_state=42
        )
        
        history = model.fit(
            [x_feat_tr, x_item_tr],
            y_tr,
            validation_split=0.1,
            epochs=train_params["epochs"],
            batch_size=train_params.get("batch_size", 32),
            verbose=0,
            callbacks=[self.tensorboard]
        )
    
        return history
    ###########################################################################################

    def save_model(self, model, combined_df, history, base_dir):
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

       print(f"[save_model] writing combined_df snapshot (parquet, pre-normalized)")
       combined_df.to_parquet(os.path.join(modelDir, "combined_training_df_frozen.parquet"), compression="snappy")
       
       print("[save_model] writing training history json")
       groceryMLCore.write_json(history.history, os.path.join(modelDir, "history.json"))
       
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
                
        norm_params = self.fit_normalization_params(combined_df)
        normalized_df = self.normalize_features(combined_df, norm_params)
    
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
    def save_experiment(self, model, combined_df, extra_dataframes, history, build_params, train_params, exp_dir):
        name_parts = []

        print("Exporting extra_dataframes:")
        groceryMLCore.export_dataframes_with_exp_name(extra_dataframes, exp_dir)
        
        groceryMLCore.save_model(model, combined_df, history, exp_dir)
        groceryMLCore.write_json(build_params, os.path.join(exp_dir, "build_params.json"))
        groceryMLCore.write_json(train_params, os.path.join(exp_dir, "train_params.json"))
        
        print("Saved experiment →", exp_dir)  
    ###########################################################################################
    

    def RunPredictionsOnly(self, combined_df_latest, model_dir, prediction_date):
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
        pred_input = self.build_prediction_input(combined_df_latest, prediction_date, norm_params)

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
        prediction_df = self.itemNameUtils.map_item_ids_to_names(prediction_df)
        prediction_df = prediction_df.sort_values("prediction", ascending=False).reset_index(drop=True)

        return prediction_df
    ###########################################################################################
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
        latest_rows_df = (tmp_df.sort_values("date")  .groupby("itemId") .tail(1) .reset_index(drop=True))
    
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

        # overwrite time-varying trip feature
        latest_rows_df["daysSinceLastTrip_raw"] = TemporalFeatures.compute_days_since_last_trip_value(self.live_df, prediction_date)
        # carry-forward stateful trip feature
        latest_rows_df["avgDaysBetweenTrips_raw"] = (self.live_df.sort_values("date").tail(1)["avgDaysBetweenTrips_raw"].iloc[0])

        latest_rows_df["item_due_ratio_feat"] = TemporalFeatures.compute_item_due_ratio(latest_rows_df)
                                                                                       
        # calendar features
        latest_rows_df["daysUntilSchoolStart_raw"] = SchoolFeatures.compute_days_until_school_start(prediction_date)
        latest_rows_df["daysUntilSchoolEnd_raw"] = SchoolFeatures.compute_days_until_school_end(prediction_date)
        latest_rows_df["schoolSeasonIndex_feat"] = SchoolFeatures.compute_school_season_index(prediction_date)
        latest_rows_df["daysUntilNextHoliday_raw"]   = latest_rows_df["date"].apply(HolidayFeatures.compute_days_until_next_holiday)
        latest_rows_df["daysSinceLastHoliday_raw"]   = latest_rows_df["date"].apply(HolidayFeatures.compute_days_since_last_holiday)
        latest_rows_df["holidayProximityIndex_feat"]  = latest_rows_df["date"].apply(HolidayFeatures.compute_holiday_proximity_index)
        

        # ensure no leakage
        if "didBuy_target" in latest_rows_df.columns:
            latest_rows_df.drop(columns=["didBuy_target"], inplace=True)
    
        # ensure itemId matches training expectations
        latest_rows_df["itemId"] = latest_rows_df["itemId"].astype("category")
    
        # remove non-feature column
        if "source" in latest_rows_df.columns:
            latest_rows_df.drop(columns=["source"], inplace=True)
            
        normalized_latest_rows_df = self.normalize_features(latest_rows_df, norm_params);
        print("build_prediction_input() done")               
        feature_cols = self.get_normalized_feature_col_names(normalized_latest_rows_df);
        x_features = normalized_latest_rows_df[feature_cols].to_numpy(np.float32)
        x_item_idx = normalized_latest_rows_df["itemId"].to_numpy(np.int32)
        self.export_df_to_excel_table(normalized_latest_rows_df, "normalized_latest_rows_df.xlsx", ".")
        self.export_df_to_excel_table(latest_rows_df, "latest_rows_df.xlsx", ".")
        print("build_prediction_input() is done")
        
        return {
            "latest_rows_df": latest_rows_df,
            "prediction_df": normalized_latest_rows_df,
            "x_features": x_features,
            "x_item_idx": x_item_idx,
            "feature_cols": feature_cols
        }
    ###########################################################################################    

    # def build_prediction_input(self, combined_df, prediction_date, norm_params):
    
    #     print(f" build_prediction_input()   Prediction date: {prediction_date.strftime('%Y-%m-%d')}")
    
    #     latest_rows_df = (
    #         combined_df.sort_values("date")
    #         .groupby("itemId")
    #         .tail(1)
    #         .copy()
    #         .reset_index(drop=True)
    #     )

    #     #freq_map = self.recompute_habit_frequency_for_prediction_time(prediction_date)
    #     #latest_rows_df["itemPurchaseHabitFrequency_feat"] = latest_rows_df["itemId"].map(freq_map).fillna(0)
        
    #     latest_rows_df["date"] = prediction_date
       
    #     # days since last trip
    #     max_data_date = combined_df["date"].max()
    #     days_forward = (prediction_date - max_data_date).days
    #     # latest_rows_df["daysSinceLastTrip_raw"] = days_forward
    #     # avg days between trips (global)
    #     # latest_rows_df["avgDaysBetweenTrips_raw"] = combined_df["avgDaysBetweenTrips_raw"].iloc[-1]
    #     # trip due ratio
    #     # TemporalFeatures.compute_trip_due_ratio(latest_rows_df)
        
    #     # === extend "days since this item last purchased"
    #     # last known value already correct per item at max_data_date
    #     last_vals = combined_df.sort_values("date").groupby("itemId").tail(1)
    #     latest_rows_df["daysSinceThisItemLastPurchased_raw"] = (last_vals["daysSinceThisItemLastPurchased_raw"].values + days_forward)
    
    #     # === extend avgDaysBetweenItemPurchases per item
    #     last_avg_vals = combined_df.sort_values("date").groupby("itemId").tail(1)
    #     latest_rows_df["avgDaysBetweenItemPurchases_raw"] = last_avg_vals["avgDaysBetweenItemPurchases_raw"].values

    #     latest_rows_df = groceryMLCore.create_item_supply_level_feat(latest_rows_df);
        
    #     latest_rows_df[[
    #         "daysSinceThisItemLastPurchased_raw",
    #         "avgDaysBetweenItemPurchases_raw"
    #     ]] = latest_rows_df[[
    #         "daysSinceThisItemLastPurchased_raw",
    #         "avgDaysBetweenItemPurchases_raw"
    #     ]].fillna(0)
    
    #     latest_rows_df["item_due_ratio_feat"] = TemporalFeatures.compute_item_due_ratio(latest_rows_df)
    #     #TemporalFeatures.compute_recent_purchase_penalty(latest_rows_df)
        
    #     #
    #     latest_rows_df["daysUntilNextHoliday_feat"] = HolidayFeatures.compute_days_until_next_holiday(prediction_date)
    #     latest_rows_df["daysSinceLastHoliday_feat"] = HolidayFeatures.compute_days_since_last_holiday(prediction_date)
    #     latest_rows_df["holidayProximityIndex_feat"] = HolidayFeatures.compute_holiday_proximity_index(prediction_date)
    #     #
    #     latest_rows_df["daysUntilSchoolStart_feat"] = SchoolFeatures.compute_days_until_school_start(prediction_date)
    #     latest_rows_df["daysUntilSchoolEnd_feat"] = SchoolFeatures.compute_days_until_school_end(prediction_date)
    #     latest_rows_df["schoolSeasonIndex_feat"] = SchoolFeatures.compute_school_season_index(prediction_date)
    #     #
    #     # latest_rows_df = TemporalFeatures.add_dst_since_until_features(latest_rows_df);
    #     #       
    #     latest_rows_df["year_feat"] = prediction_date.year
    #     latest_rows_df["month_cyc_feat"] = prediction_date.month
    #     latest_rows_df["day_cyc_feat"] = prediction_date.day
    #     latest_rows_df["dow_cyc_feat"] = prediction_date.weekday()
    #     latest_rows_df["doy_feat"] = prediction_date.timetuple().tm_yday
    #     latest_rows_df["quarter_feat"] = ((prediction_date.month - 1) // 3) + 1
    
    #     if "didBuy_target" in latest_rows_df.columns:
    #         latest_rows_df.drop(columns=["didBuy_target"], inplace=True)
    
    #     normalized_latest_rows_df = self.normalize_features(latest_rows_df, norm_params)
               
    #     feature_cols = self.get_normalized_feature_col_names(normalized_latest_rows_df);
    #     x_features = normalized_latest_rows_df[feature_cols].to_numpy(np.float32)
    #     x_item_idx = normalized_latest_rows_df["itemId"].to_numpy(np.int32)
    
    #     # self.export_df_to_excel_table(normalized_latest_rows_df, "normalized_latest_rows_df.xlsx", ".")
    #     # self.export_df_to_excel_table(latest_rows_df, "latest_rows_df.xlsx", ".")
    #     print("build_prediction_input() is done")
        
    #     return {
    #         "latest_rows_df": latest_rows_df,
    #         "prediction_df": normalized_latest_rows_df,
    #         "x_features": x_features,
    #         "x_item_idx": x_item_idx,
    #         "feature_cols": feature_cols
    #     }
    ########################################################################################## 
    def get_normalized_feature_col_names(self, df):
        """Returns normalized feature columns used for model input."""
        return [c for c in df.columns if c.endswith("_norm")]
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
 