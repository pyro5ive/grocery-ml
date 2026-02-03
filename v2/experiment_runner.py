import logging
import pandas as pd
from datetime import datetime

from feature_builders.payday_prox_feature_builder import PaydayProximity_FeatureBuilder
from feature_builders.weather_history_feat_builder import WeatherHistory_FeatureBuilder
from feature_builders.item_supply_level_feature_builder import ItemSupplyLevel_FeatureBuilder
from feature_builders.school_schedule_feat_builder import SchoolSchedule_FeatureBuilder
from feature_builders.days_since_last_purchase_feat_builder import DaysSinceLastPurchase_FeatBuilder
from feature_builders.item_id_feature_builder import ItemIdFeatureBuilder
from purchase_event_builders.winn_dixie_events_df_builder import WinnDixieEventsDfBuilder

class ExperimentRunner:

    trainingSources  = {
        "walmart": r"data\training\walmart",
        "winndixie": r"..\data\training\winndixie\txt",
        "winndixieAdditional" : r"..\data\training\winndixie\additionalTxtRcpts",
        "weather": r"data\weather\VisualCrossing-70062 2000-01-01 to 2026-23-1.csv"
    }
    
    liveSources  = {
        "walmart": r"data\live\walmart",
        "winndixie": r"data\live\winndixie\txt",
        "winndixieAdditional" : r"data\live\winndixie\additionalTxtRcpts",
        "weather": r"data\weather\VisualCrossing-70062 2000-01-01 to 2026-23-1.csv"
    }
    
    def __init__(this):
        this.logger = logging.getLogger(this.__class__.__name__);
        this.itemIdFeatureBuilder = ItemIdFeatureBuilder();
        this.weatherHistoryFeatureBuilder = WeatherHistory_FeatureBuilder()
        this.payDayFeatueBuilder_steve = PaydayProximity_FeatureBuilder("sjm", pd.Timestamp("2026-01-30", tz="US/Central"));
        this.payDayFeatueBuilder_angie = PaydayProximity_FeatureBuilder("ang", pd.Timestamp("2026-01-23", tz="US/Central"));
        this.itemSupplyFeatureBuilder = ItemSupplyLevel_FeatureBuilder();
        this.schoolSchedule_featureBuidler = SchoolSchedule_FeatureBuilder();
        this.daysSinceLastPurchaseFeatureuBuilder = DaysSinceLastPurchase_FeatBuilder();
        this.winndixieDfBuilder = WinnDixieEventsDfBuilder(this.trainingSources);
    ###########################################################################
    
    def run(this):
        this._build_event_dfs();
        this._build_targetCol();
        this._build_featues();
        this.trainingDf.info();
        this._export_df_for_debug();
    ###########################################################################

    def _build_event_dfs(this):
        this.trainingDf = this.winndixieDfBuilder.build_df()
    ###########################################################################
    
    def _build_targetCol(this):
        this.trainingDf["didBuy_target"] = 1;
    ###########################################################################

    def _export_df_for_debug(this):
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        this.trainingDf.to_csv(f"trainingDf-{timeStamp}.csv");
    ###########################################################################

    def _build_featues(this):
        this.trainingDf = this.weatherHistoryFeatureBuilder.build_feature(this.trainingDf);
        this.trainingDf = this.schoolSchedule_featureBuidler.build_feature(this.trainingDf);
        this.trainingDf = this.payDayFeatueBuilder_angie.buildAll(this.trainingDf);
        this.trainingDf = this.payDayFeatueBuilder_steve.buildAll(this.trainingDf);
        this.trainingDf = this.itemIdFeatureBuilder.build_feature(this.trainingDf);
        this.trainingDf = this.daysSinceLastPurchaseFeatureuBuilder.build_feature(this.trainingDf);
        this.trainingDf = this.ex
        this.trainingDf.info();
        pass;

    #------------------------------------------------------------------------#
        
        