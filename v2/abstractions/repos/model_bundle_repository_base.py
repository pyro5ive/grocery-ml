from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
import tensorflow as tf

from models.models import BuildParams, TrainingParams
from models.model_bundle import ModelBundle


class ModelBundleRepositoryBase(ABC):
    """
    Base abstraction for persisting and loading trained model bundles.
    Implementations define how models and their associated artifacts
    are stored and retrieved.
    """

    @abstractmethod
    def save(self, bundle: ModelBundle, base_dir: str) -> None:
        """
        Persist a model bundle to storage.

        :param bundle: Model bundle containing model, data snapshot,
            normalization parameters, training history, and configs.
        :param base_dir: Base directory or storage location.
        """
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def load(self, base_dir: str) -> ModelBundle:
        """
        Load a model bundle from storage.

        :param base_dir: Base directory or storage location.
        :returns: Loaded ModelBundle instance.
        """
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def save_model(self, model: tf.keras.Model, base_dir: str) -> None:
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def load_model(self, base_dir: str) -> tf.keras.Model:
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def save_training_snapshot(self, training_df: pd.DataFrame, base_dir: str) -> None:
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def load_training_snapshot(self, base_dir: str) -> pd.DataFrame:
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def save_normalization_params(
        self,
        norm_params: Dict[str, Any],
        base_dir: str
    ) -> None:
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def load_normalization_params(self, base_dir: str) -> Dict[str, Any]:
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def save_history(self, history: Dict[str, Any], base_dir: str) -> None:
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def load_history(self, base_dir: str) -> Dict[str, Any]:
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def save_build_params(self, build_params: BuildParams, base_dir: str) -> None:
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def load_build_params(self, base_dir: str) -> BuildParams:
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def save_train_params(self, params: TrainingParams, base_dir: str) -> None:
        raise NotImplementedError
    #--------------------------#

    @abstractmethod
    def load_train_params(self, base_dir: str) -> TrainingParams:
        raise NotImplementedError
    #--------------------------#
