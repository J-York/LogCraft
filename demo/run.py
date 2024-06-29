import sys
sys.path.append("../")
from logafe.preprocess import preprocess
from model_select.select import model_select
from anomaly_detect.detect import detect
import os
import requests


logname = "test"

# data preprocess and feature enhancement
preprocess(logname, format_index=0)

# meta learner construction and model recommandation
config_dict=model_select(logname)

# model trainging and inference
detect(logname,config_dict)