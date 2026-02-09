from experiment_runner import ExperimentRunner
from models.models import *

from datetime import datetime
import os

expRunner = ExperimentRunner();

layers_cfg = [
    LayerSpec(units=8, activation="relu")
]

build_config = BuildParams(
    embeddingDimCount=4,
    layers=layers_cfg,
    outputActivation="sigmoid",
    optimizer="adam",
    learningRate=0.001,
    loss="binary_crossentropy",
    metrics=["AUC"]
)

train_config = TrainingParams(
    epochs=1,
    batchSize=4
)

timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("debug", f"test-exp_{timestamp}")

expRunner.run(build_config, train_config, run_dir);
