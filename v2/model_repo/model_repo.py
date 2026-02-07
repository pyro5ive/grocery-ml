import os
import json
import pandas as pd
import tensorflow as tf
from typing import Dict, Any
from .model_artifacts import ModelArtifacts

class KerasFileSystemModelRepository:

    # -------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------

    def save_all(self, artifacts: ModelArtifacts, base_dir: str) -> None:
        self.save_model(artifacts.model, base_dir)
        self.save_training_snapshot(artifacts.training_df, base_dir)
        self.save_normalization_params(artifacts.normalization_params, base_dir)
        self.save_history(artifacts.history, base_dir)
        self.save_build_params(artifacts.build_params, base_dir)
        self.save_train_params(artifacts.train_params, base_dir)

    # -------------------------------------------------------------

    def load_all(self, base_dir: str) -> ModelArtifacts:
        model = self.load_model(base_dir)
        training_df = self.load_training_snapshot(base_dir)
        normalization_params = self.load_normalization_params(base_dir)
        history = self.load_history(base_dir)
        build_params = self.load_build_params(base_dir)
        train_params = self.load_train_params(base_dir)

        return ModelArtifacts(
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
        self._write_json(norm_params, os.path.join(model_dir, "normalization_params.json"))

    # -------------------------------------------------------------

    def save_history(self, history: Dict[str, Any], base_dir: str) -> None:
        model_dir = self._ensure_model_dir(base_dir)
        self._write_json(history, os.path.join(model_dir, "history.json"))

    # -------------------------------------------------------------

    def save_build_params(self, build_params: Dict[str, Any], base_dir: str) -> None:
        model_dir = self._ensure_model_dir(base_dir)
        self._write_json(build_params, os.path.join(model_dir, "build_params.json"))

    # -------------------------------------------------------------

    def save_train_params(self, train_params: Dict[str, Any], base_dir: str) -> None:
        model_dir = self._ensure_model_dir(base_dir)
        self._write_json(train_params, os.path.join(model_dir, "train_params.json"))

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

    def load_build_params(self, base_dir: str) -> Dict[str, Any]:
        model_dir = os.path.join(base_dir, "model")
        return self._read_json(
            os.path.join(model_dir, "build_params.json")
        )

    # -------------------------------------------------------------

    def load_train_params(self, base_dir: str) -> Dict[str, Any]:
        model_dir = os.path.join(base_dir, "model")
        return self._read_json(
            os.path.join(model_dir, "train_params.json")
        )

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
