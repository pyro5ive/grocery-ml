import logging
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard


class KerasModelBuilder:

    featureColCount: int
    featureCols: list[str]
    buildParams: dict
    trainingParams: dict
    trainDf: pd.DataFrame

    def __init__(this):
        pass

    #==========================================================================================#
    def build_and_train_model(this, train_df: pd.DataFrame, featureCols: list[str], buildParams: dict, trainParams: dict) -> None:
        this.featureColCount = len(featureCols);
        this.featureCols = featureCols;
        this.trainingParams = trainParams;
        this.buildParams = buildParams;
        this.trainDf = train_df

    #==========================================================================================#

    def build_model(this):
        num_in = layers.Input(shape=(this.featureColCount,))
        item_in = layers.Input(shape=(), dtype="int32")

        emb = layers.Embedding(input_dim=item_count,
            output_dim=this.buildParams["embeddingDimCount"],
            name="item_embedding"  # <-- REQUIRED FOR TENSORBOARD PROJECTOR
        )(item_in)

        x = layers.Concatenate()([num_in, layers.Flatten()(emb)])

        for spec in this.buildParams["layers"]:
            x = layers.Dense(spec["units"], activation=spec["activation"])(x)

        outputLayer = layers.Dense(1, activation=this.buildParams["output_activation"])(x)

        model = models.Model(inputs=[num_in, item_in], outputs=outputLayer)

        optimizer_name = this.buildParams.get("optimizer", "adam")
        learning_rate = this.buildParams.get("learning_rate")

        if optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == "adamw":
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        model.compile(
            optimizer=optimizer,
            loss=this.buildParams.get("loss"),
            metrics=this.buildParams.get("metrics")
        )

        return model


    def fit_normalization_params(this, df):
        params = {}
        feature_cols = this.groceryMLCore.get_feature_col_names(df)
        cyc_cols = [c for c in feature_cols if c.endswith("_cyc_feat")]
        num_cols = [c for c in feature_cols if c not in cyc_cols]

        for col in num_cols:
            if this.groceryMLCore.is_binary_column(df, col):
                continue

            params[col] = {
                "mean": df[col].mean(),
                "std": df[col].std()
            }

        for col in cyc_cols:
            params[col] = {
                "period": TemporalFeatures.get_period_for_column(col)
            }

        return params

  #==========================================================================================#
  def normalize_features(this, df, norm_params):
      this.logger.info("normalize_features()")

      normalized_df = df.copy()

      for col, cfg in norm_params.items():

          if col.endswith("_cyc_feat"):
              sin_col, cos_col = TemporalFeatures.encode_sin_cos(
                  df[col],
                  cfg["period"]
              )

              normalized_df[f"{col}_sin_norm"] = sin_col
              normalized_df[f"{col}_cos_norm"] = cos_col
              normalized_df.drop(columns=[col], inplace=True)

          else:
              mean_val = cfg["mean"]
              std_val = cfg["std"]
              norm_col = col.replace("_feat", "_norm")

              if std_val == 0:
                  normalized_df[norm_col] = 0.0
              else:
                  normalized_df[norm_col] = (df[col] - mean_val) / std_val

              normalized_df.drop(columns=[col], inplace=True)

      return normalized_df

  #==========================================================================================#
  # def get_model_feature_col_names(this, df):
  #     """
  #     Returns all feature columns used for model input:
  #     - normalized continuous features (*_norm)
  #     - binary features (*_feat, not normalized)
  #     """
  #     norm_cols = [c for c in df.columns if c.endswith("_norm")]
  #
  #     binary_feat_cols = []
  #     for c in df.columns:
  #         if c.endswith("_feat") and this.groceryMLCore.is_binary_column(df, c):
  #             binary_feat_cols.append(c)
  #
  #     return norm_cols + binary_feat_cols
  #
  #==========================================================================================#






































  #==========================================================================================#
  # def create_tensorboard(this, log_dir):
  #
  #     tensorboard = tf.keras.callbacks.TensorBoard(
  #         log_dir=log_dir,
  #         histogram_freq=1,
  #         write_graph=True,
  #         write_images=True,
  #         embeddings_freq=1,
  #         embeddings_metadata=f"{log_dir}/embeddingslabels.tsv"
  #     )
  #     return tensorboard;
  #
  #==========================================================================================#
  def train_model(this, model, df, feature_cols, target_col, train_params):
      this.logger.info("train_model()");
      callbacks = []
      x_feat = df[this.fe].to_numpy(np.float32)
      x_item = df["itemId"].to_numpy(np.int32)
      y = df[target_col].to_numpy(np.float32)

      x_feat_tr, x_feat_te, x_item_tr, x_item_te, y_tr, y_te = train_test_split(
          x_feat, x_item, y, test_size=0.2, random_state=42
      )

      callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True))
      callbacks.append(this.tensorboard)

      history = model.fit(
          [x_feat_tr, x_item_tr],
          y_tr,
          validation_split=0.1,
          epochs=train_params["epochs"],
          batch_size=train_params.get("batch_size", 32),
          verbose=0,
          callbacks=callbacks
      )

      return history
  #==========================================================================================#
