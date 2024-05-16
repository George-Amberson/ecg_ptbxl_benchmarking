from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *
import pandas as pd
import os


def main():
    
    datafolder = './data/ptbxl/'
    outputfolder = './output/'

    models = [
        conf_fastai_xresnet1d101,
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        ('exp0', 'all'),
       ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        e.evaluate()
        for m in sorted(os.listdir(outputfolder+name+'/models')):
            rpath = outputfolder+name+'/models/'+m+'/results/'
            me_res = pd.read_csv(rpath + 'te_results.csv', index_col=0)
            metric = me_res.loc['point']['macro_auc']
            print(f"\n Macro-AUC for {m}: {metric}")
            if m == "fastai_xresnet1d101" and metric > 0.92:
                print("TEST PASSED")
            else:
                print("TEST NOT PASSED")
            if m == "naive" and metric >= 0.5:
                print("TEST PASSED")
            else:
                print("TEST NOT PASSED")


if __name__ == "__main__":
    main()