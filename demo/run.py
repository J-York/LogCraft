import sys
sys.path.append("../")
from logafe.preprocess import preprocess
from model_select.select import model_select
from anomaly_detect.detect import detect
import os
import requests


if __name__ == "__main__":
    logname = "BGL"

    pid = os.getpid()
    url = 'http://10.10.1.210/api/v1/job/create'
    data = {
        'student_id': '1913179',
        'password': 'ljqdx666@',
        'description': 'AutoLogAD_run',
        'server_ip': '10.10.1.219',
        'duration': '几天',
        'pid': pid,
        'server_user': 'zhangshenglin',
        'command': 'python',
        'use_gpu': 1,
    }
    # r = requests.post(url, data=data)
    # print('210 reqeat: ', r.text)

    # data preprocess and feature enhancement
    preprocess(logname, format_index=0)

    # meta learner construction and model recommandation
    config_dict=model_select(logname)

    # model trainging and inference
    detect(logname,config_dict)