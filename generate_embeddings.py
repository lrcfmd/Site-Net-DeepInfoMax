#!/usr/bin/env python3
import argparse
import os
import torch
import yaml
import pandas as pd

from lightning_module import DIM_h5_Data_Module, SiteNet_DIM

# Optional environment variable settings
os.environ["export MKL_NUM_THREADS"] = "1"
os.environ["export NUMEXPR_NUM_THREADS"] = "1"
os.environ["export OMP_NUM_THREADS"] = "1"
os.environ["export OPENBLAS_NUM_THREADS"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DIM embeddings for a given dataset")
    parser.add_argument("-c", "--config", required=True, help="Path to the config YAML file")
    parser.add_argument("-m", "--model", required=True, help="Path to the model checkpoint file")
    parser.add_argument("-f", "--file", required=True, help="Path to the HDF5 dataset file")
    parser.add_argument("-w", "--number_of_worker_processes",default = 1,type=int)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load hyperparameter configuration
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Update config with the provided dataset file and batch settings
    config["h5_file"] = args.file
    config["dynamic_batch"] = False
    config["Batch_Size"] = 128

    # Create the data module (assumes HDF5 format as in the original script)
    Dataset = DIM_h5_Data_Module(
        config,
        max_len=None,
        ignore_errors=True,
        overwrite=False,
        cpus=args.number_of_worker_processes,
        chunk_size=256,
    )

    ids = [i["database_ID"] for i in Dataset.Dataset]

    # Initialize and load the model
    model = SiteNet_DIM(config)
    model.to(device)
    checkpoint = torch.load(args.model, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # Generate embeddings using the model's forward pass
    model.eval()
    with torch.no_grad():
        embeddings = model.forward(Dataset.Dataset, batch_size=128)
        embeddings = embeddings.detach().cpu().numpy()

    # Save the embeddings to a CSV file
    df = pd.DataFrame(embeddings)
    df["database_ID"] = ids
    df.to_csv("DIM_embeddings.csv", index=False)
    print("Embeddings saved to DIM_embeddings.csv")
