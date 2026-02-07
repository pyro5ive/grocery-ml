from dataclasses import dataclass
from typing import List

@dataclass
class LayerSpec:
    units: int
    activation: str


@dataclass
class BuildParams:
    embeddingDimCount: int
    layers: List[LayerSpec]
    outputActivation: str
    optimizer: str
    learningRate: float
    loss: str
    metrics: List[str]


@dataclass
class TrainingParams:
    epochs: int
    batchSize: int
