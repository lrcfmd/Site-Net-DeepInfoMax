import pytorch_lightning as pl
import sys
from matminer.featurizers.site import *
import matminer

site_feauturizers_dict = matminer.featurizers.site.__dict__
from lightning_module import (
    basic_callbacks,
    DIM_h5_Data_Module,
    SiteNet,
    SiteNet_DIM
)
from lightning_module import basic_callbacks
import yaml
from pytorch_lightning.callbacks import *
import argparse
import os
os.environ["export MKL_NUM_THREADS"] = "1"
os.environ["export NUMEXPR_NUM_THREADS"] = "1"
os.environ["export OMP_NUM_THREADS"] = "1"
os.environ["export OPENBLAS_NUM_THREADS"] = "1"
import torch
import pandas as pd
from scipy import stats
import numpy as np
import sys, os
from modules import SiteNetAttentionBlock,SiteNetEncoder,k_softmax
from tqdm import tqdm
from lightning_module import collate_fn
from lightning_module import af_dict as lightning_af_dict
import torch
from torch_scatter import segment_coo,segment_csr
from torch import nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression,ElasticNet
import pickle as pk
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#monkeypatches

compression_alg = "gzip"

import pickle as pk

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="ml options")
    parser.add_argument("-w", "--number_of_worker_processes", default=1,type=int)
    parser.add_argument("-u", "--cell_size_limit", default = None )
    args = parser.parse_args()
    config_and_model = [["Initial_eform","config/compact_dim_klnorm.yaml",None,"e_form"],
                        ["Initial_egap","config/compact_dim_klnorm.yaml",None,"e_gap"],
                        ["nocomp_klnorm_moremultiloss_eform","config/compact_dim_nocomp_klnorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nocomp_klnorm_DIM-v2.ckpt","e_form"],
                        ["nocomp_klnorm_moremultiloss_egap","config/compact_dim_nocomp_klnorm.yaml","Data/Matbench/matbench_mp_gap_cubic_50_train_1.hdf5_best_compact_dim_nocomp_klnorm_DIM.ckpt","e_gap"],
                        ["nocomp_klnorm_eform","config/compact_dim_nocomp_klnorm_noextrafalse.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nocomp_klnorm_noextrafalse_DIM-v1.ckpt","e_form"],
                        ["nocomp_klnorm_egap","config/compact_dim_nocomp_klnorm_noextrafalse.yaml","Data/Matbench/matbench_mp_gap_cubic_50_train_1.hdf5_best_compact_dim_nocomp_klnorm_noextrafalse_DIM.ckpt","e_gap"]
                        ]

    #config_and_model = [["klnorm_multiloss","config/compact_dim_klnorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_klnorm_DIM-v2.ckpt"]]

    limits = [50, 100, 250, 1000]
    repeats = [100,100,100,100]
    #repeats = [1000, 250, 100, 25, 10, 5]

    results_dataframe = pd.DataFrame(columns = ["rf_R2","rf_MAE","rf_MSE","nn_R2","nn_MAE","nn_MSE","lin_R2","lin_MAE","lin_MSE","model","limit","measure"])

    train_data_dict = {"e_form":"Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5","e_gap":"Data/Matbench/matbench_mp_gap_cubic_50_train_1.hdf5"}
    test_data_dict = {"e_form":"Data/Matbench/matbench_mp_e_form_cubic_50_test_1.hdf5","e_gap":"Data/Matbench/matbench_mp_gap_cubic_50_test_1.hdf5"}

    for cm in config_and_model:
        print("Model type is " + cm[0])

        torch.set_num_threads(args.number_of_worker_processes)
        try:
            print("config file is " + cm[1])
            with open(str(cm[1]), "r") as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
        except Exception as e:
            raise RuntimeError(
                "Config not found or unprovided, a path to a configuration yaml must be provided with -c"
            )
        results_list = []
        model_name = cm[2]
        dataset_name = train_data_dict[cm[3]] #Get train dataset according to training target
        #config["Max_Samples"] = 100
        config["h5_file"] = dataset_name
        config["dynamic_batch"] = False
        config["Batch_Size"] = 128
        #config["Max_Samples"] = 1000
        if args.cell_size_limit != None:
            args.cell_size_limit = int(args.cell_size_limit)
        Dataset = DIM_h5_Data_Module(
            config,
            max_len=args.cell_size_limit,
            ignore_errors=True,
            overwrite=False,
            cpus=args.number_of_worker_processes,
            chunk_size=32,
        )

        dataset_name = test_data_dict[cm[3]] #Get test dataset according to training target
        config["h5_file"] = dataset_name
        #config["Max_Samples"] = 1000
        Dataset_Test = DIM_h5_Data_Module(
            config,
            max_len=args.cell_size_limit,
            ignore_errors=True,
            overwrite=False,
            cpus=args.number_of_worker_processes,
            chunk_size=32,
        )

        #for limit,repeat in zip(limits,repeats):
        torch.cuda.empty_cache()
        model = SiteNet_DIM(config)
        model.to(device)
        if model_name != None:
            print("DIM PARAMETERS")
            model.load_state_dict(torch.load(model_name,map_location=torch.device("cpu"))["state_dict"], strict=True)
        else:
            print("INITIAL PARAMETERS")
        results = model.forward(Dataset.Dataset,batch_size=128).detach().cpu().numpy()
        results_y = np.array([Dataset.Dataset[i]["target"] for i in range(len(Dataset.Dataset))])
        results_Test = model.forward(Dataset_Test.Dataset,batch_size=128).detach().cpu().numpy()
        results_test_y = np.array([Dataset_Test.Dataset[i]["target"] for i in range(len(Dataset_Test.Dataset))])

        results_test_tsne = TSNE(init="pca",perplexity=1000,learning_rate="auto").fit_transform(results_Test)
        results_test_tsne = pd.DataFrame(results_test_tsne)
        results_test_tsne = results_test_tsne
        results_test_tsne["structure"] = [pk.dumps(Dataset_Test.Dataset[i]["structure"]) for i in range(len(Dataset_Test.Dataset))]
        results_test_tsne["target"] = [Dataset_Test.Dataset[i]["target"] for i in range(len(Dataset_Test.Dataset))]
        results_test_tsne.to_csv("TSNE_" + cm[0] + ".csv")

        for limit,repeat in zip(limits,repeats):
            print("Limit is " + str(limit))
            samples = [np.random.choice(np.arange(len(Dataset.Dataset)), size=min(limit,len(Dataset.Dataset)), replace=False) for i in range(repeat)]
            rows = pd.DataFrame(columns = ["rf_R2","rf_MAE","rf_MSE","nn_R2","nn_MAE","nn_MSE","lin_R2","lin_MAE","lin_MSE"])
            for i,sample in enumerate(samples):
                if cm[2] is None:
                    print("Normalizing untrained model inputs")
                    rf = make_pipeline(StandardScaler(),RandomForestRegressor()).fit(results[sample,:], results_y[sample])
                    nn = make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=[64], max_iter=5000)).fit(results[sample,:], results_y[sample])
                    lin = make_pipeline(StandardScaler(),LinearRegression()).fit(results[sample,:], results_y[sample])
                else:
                    print("Do not normalize trained latent space")
                    rf = RandomForestRegressor().fit(results[sample,:], results_y[sample])
                    nn = MLPRegressor(hidden_layer_sizes=[64], max_iter=5000).fit(results[sample,:], results_y[sample])
                    lin = LinearRegression().fit(results[sample,:], results_y[sample])
                rows = rows.append(pd.DataFrame({
                    "rf_R2": rf.score(results_Test, results_test_y),
                    "rf_MAE":np.mean(np.absolute(rf.predict(results_Test)-results_test_y)),
                    "rf_MSE":np.mean(np.array(rf.predict(results_Test)-results_test_y)**2),
                    "nn_R2": nn.score(results_Test, results_test_y),
                    "nn_MAE":np.mean(np.absolute(nn.predict(results_Test)-results_test_y)),
                    "nn_MSE":np.mean(np.array(nn.predict(results_Test)-results_test_y)**2),
                    "lin_R2": lin.score(results_Test, results_test_y),
                    "lin_MAE":np.mean(np.absolute(lin.predict(results_Test)-results_test_y)),
                    "lin_MSE":np.mean(np.array(lin.predict(results_Test)-results_test_y)**2),
                },
                index=[str(i)]))
            
            rows["model"] = cm[0]
            rows["limit"] = limit
            results_dataframe = results_dataframe.append(rows, ignore_index=True)
            results_dataframe.to_csv("Downstream_DIM.csv")  