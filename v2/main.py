from experiment_runner import ExperimentRunner
from models.models import *

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

expRunner.run(build_config, train_config);
