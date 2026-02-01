import logging
import pandas as pd

class PredictionService:
    
    def __init__(self):
        self.newPurchaseEventsDfBuilder = PurchaseEventsDfBuilder();
        self.historicalPurchaseEventsDfBuilder = PurchaseEventsDfBuilder();
        thisClassName = self.__class__.__name__
        self.logger = logging.getLogger(thisClassName)
    #########################################################################################
    
    def run_prediction(self, trainingDF, model, predictionDates):
        newPurchaseEventsDf = self.newPurchaseEventsDfBuilder.build_df();
        trainingEventsDf = self.historicalPurchaseEventsDfBuilder.build_df(trainingDF);
        
    #########################################################################################
    
    def load_model(self, model_dir):
        logger.info(f"[load_model_artifacts] loading artifacts from → {model_dir}")

        model_sub_dir = os.path.join(model_dir, "model")
        logger.info(f"[load_model_artifacts] loading model → {model_sub_dir}")
        model = tf.keras.models.load_model(model_sub_dir)

        frozen_path = os.path.join(model_sub_dir, "training_df_frozen.parquet")
        logger.info(f"[load_model_artifacts] loading frozen combined_df → {frozen_path}")
        combined_df_frozen = pd.read_parquet(frozen_path)

        logger.info("[load_model_artifacts] done")
        return model, combined_df_frozen
    #########################################################################################