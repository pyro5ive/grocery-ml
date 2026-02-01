import logging
import pandas as pd

class ManualEntryEventsDfBuilder:


    def _init_(this):
        thisClassName = this.__class__.__name__
        self.logger = logging.getLogger(thisClassName)
    ###########################################
    def build_df(this):
        this.logger.info("Manual Entry Events Df builder starting")
        eventsDf = None;


        this.logger.info("Manual Entry Events Df builder is finsihed")
        return eventsDf;
    ###########################################
