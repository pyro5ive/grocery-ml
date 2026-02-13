import os
import json
import pandas as pd
import tensorflow as tf
from typing import Dict, Any
from .model_bundle import ModelBundle
from models.models import BuildParams, TrainingParams
import logging


class KerasFileSystemModelRepository:


    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
    # -------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------

    def save(self,  bundle:ModelBundle, base_dir: str) -> None:

        if bundle.model is not None:
            self.save_model(bundle.model, base_dir)
        else:
            self.logger.warning("Modelbundle.model is None. Skipping save_model().")

        if bundle.training_df is not None:
            self.save_training_snapshot(bundle.training_df, base_dir)
        else:
            self.logger.warning("Modelbundle.training_df is None. Skipping save_training_snapshot().")

        if bundle.normalization_params is not None:
            self.save_normalization_params(bundle.normalization_params, base_dir)
        else:
            self.logger.warning("Modelbundle.normalization_params is None. Skipping save_normalization_params().")

        if bundle.history is not None:
            self.save_history(bundle.history, base_dir)
        else:
            self.logger.warning("Modelbundle.history is None. Skipping save_history().")

        if bundle.build_params is not None:
            self.save_build_params(bundle.build_params, base_dir)
        else:
            self.logger.warning("Modelbundle.build_params is None. Skipping save_build_params().")

        if bundle.train_params is not None:
            self.save_train_params(bundle.train_params, base_dir)
        else:
            self.logger.warning("Modelbundle.train_params is None. Skipping save_train_params().")

    # -------------------------------------------------------------

    def load(self, base_dir: str) -> ModelBundle:
        model = self.load_model(base_dir)
        training_df = self.load_training_snapshot(base_dir)
        normalization_params = self.load_normalization_params(base_dir)
        history = self.load_history(base_dir)
        build_params = self.load_build_params(base_dir)
        train_params = self.load_train_params(base_dir)

        return ModelBundle(
            model=model,
            training_df=training_df,
            normalization_params=normalization_params,
            history=history,
            build_params=build_params,
            train_params=train_params,
        )

    # -------------------------------------------------------------
    # Individual Save Methods
    # -------------------------------------------------------------

    def save_model(self, model: tf.keras.Model, base_dir: str) -> None:
        model_dir = self._ensure_model_dir(base_dir)
        model.save(model_dir)

    # -------------------------------------------------------------

    def save_training_snapshot(self, training_df: pd.DataFrame, base_dir: str) -> None:
        model_dir = self._ensure_model_dir(base_dir)
        training_df.to_parquet(
            os.path.join(model_dir, "training_df_frozen.parquet"),
            compression="snappy"
        )

    # -------------------------------------------------------------

    def save_normalization_params(self, norm_params: Dict[str, Any], base_dir: str) -> None:
        model_dir = self._ensure_model_dir(base_dir)
        self._write_json(
            norm_params,
            os.path.join(model_dir, "normalization_params.json")
        )

    # -------------------------------------------------------------

    def save_history(self, history: Dict[str, Any], base_dir: str) -> None:
        model_dir = self._ensure_model_dir(base_dir)
        self._write_json(
            history,
            os.path.join(model_dir, "history.json")
        )

    # -------------------------------------------------------------

    def save_build_params(self, build_params: BuildParams, base_dir: str) -> None:
        model_dir = self._ensure_model_dir(base_dir)
        self._write_json(
            build_params.to_dict(),
            os.path.join(model_dir, "build_params.json")
        )

    # -------------------------------------------------------------

    def save_train_params(self, params: TrainingParams, base_dir: str) -> None:
        model_dir = self._ensure_model_dir(base_dir)
        self._write_json(
            params.to_dict(),
            os.path.join(model_dir, "train_params.json")
        )
    # -------------------------------------------------------------
    # Individual Load Methods
    # -------------------------------------------------------------

    def load_model(self, base_dir: str) -> tf.keras.Model:
        model_dir = os.path.join(base_dir, "model")
        return tf.keras.models.load_model(model_dir)

    # -------------------------------------------------------------

    def load_training_snapshot(self, base_dir: str) -> pd.DataFrame:
        model_dir = os.path.join(base_dir, "model")
        return pd.read_parquet(
            os.path.join(model_dir, "training_df_frozen.parquet")
        )

    # -------------------------------------------------------------

    def load_normalization_params(self, base_dir: str) -> Dict[str, Any]:
        model_dir = os.path.join(base_dir, "model")
        return self._read_json(
            os.path.join(model_dir, "normalization_params.json")
        )

    # -------------------------------------------------------------

    def load_history(self, base_dir: str) -> Dict[str, Any]:
        model_dir = os.path.join(base_dir, "model")
        return self._read_json(
            os.path.join(model_dir, "history.json")
        )

    # -------------------------------------------------------------

    def load_build_params(self, base_dir: str) -> BuildParams:
        model_dir = os.path.join(base_dir, "model")
        data = self._read_json(
            os.path.join(model_dir, "build_params.json")
        )
        return BuildParams.from_dict(data)

    # -------------------------------------------------------------

    def load_train_params(self, base_dir: str) -> TrainingParams:
        model_dir = os.path.join(base_dir, "model")
        data = self._read_json(
            os.path.join(model_dir, "train_params.json")
        )
        return TrainingParams.from_dict(data)

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------

    def _ensure_model_dir(self, base_dir: str) -> str:
        model_dir = os.path.join(base_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    # -------------------------------------------------------------

    def _write_json(self, data: Dict[str, Any], path: str) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # -------------------------------------------------------------

    def _read_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)
