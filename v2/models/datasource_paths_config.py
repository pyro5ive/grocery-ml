class DataSourcePathsConfig:
    def __init__(
        this,
        trainingPaths: dict[str, str],
        livePaths: dict[str, str]
    ):
        this.trainingPaths = dict(trainingPaths)
        this.livePaths = dict(livePaths)
#--------------------------#
