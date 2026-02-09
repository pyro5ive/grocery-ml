from dataclasses import dataclass
from typing import List

from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class LayerSpec:
    units: int
    activation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LayerSpec":
        return LayerSpec(
            units=data["units"],
            activation=data["activation"]
        )


@dataclass
class BuildParams:
    embeddingDimCount: int
    layers: List[LayerSpec]
    outputActivation: str
    optimizer: str
    learningRate: float
    loss: str
    metrics: List[str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BuildParams":
        layers = [LayerSpec.from_dict(layer) for layer in data["layers"]]

        return BuildParams(
            embeddingDimCount=data["embeddingDimCount"],
            layers=layers,
            outputActivation=data["outputActivation"],
            optimizer=data["optimizer"],
            learningRate=data["learningRate"],
            loss=data["loss"],
            metrics=data["metrics"]
        )


@dataclass
class TrainingParams:
    epochs: int
    batchSize: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TrainingParams":
        return TrainingParams(
            epochs=data["epochs"],
            batchSize=data["batchSize"]
        )
