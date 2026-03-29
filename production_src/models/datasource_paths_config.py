class DataSourcePathsConfig:
    def __init__(this, trainingPaths: dict[str, str], livePaths: dict[str, str]):
        this.trainingPaths = dict(trainingPaths)
        this.livePaths = dict(livePaths)
    # =============================================================================#

    def get_training_path(this, key: str) -> str:
        if key not in this.trainingPaths:
            raise KeyError(f"trainingPaths missing key '{key}'")
        return this.trainingPaths[key]
    #=============================================================================#

    def get_live_path(this, key: str) -> str:
        if key not in this.livePaths:
            raise KeyError(f"livePaths missing key '{key}'")
        return this.livePaths[key]
    #=============================================================================#