
from dataclasses import dataclass, asdict
from typing import Dict, Any

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
