import pandas as pd
import optuna
import torch.nn as nn
import torch

from config import TrainConfig
from model_config import ModelConfig
from data_setup import DataSetup
from models import SLR_CNN
from train import Train
from validate import Validation
from torchvision import transforms

from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler

import os
os.environ["RAY_DASHBOARD_DISABLE"] = "1"

print(torch.cuda.is_available())

config = TrainConfig()

SEARCH_SPACE = {
    "in_channels": 6,
    "input_height": 48,
    "input_width": 24,
    "channel_list": tune.choice([[16, 32], [32, 64], [64, 128]]),
    "kernel_size": tune.choice([3, 5, 7]),
    "padding": tune.choice([0, 1]),
    "pool_kernel": tune.choice([2, 3]),
    "dropout_rate": tune.uniform(0.1, 0.5),
    "fc_hidden_dim": tune.choice([64, 128, 256]),
    "lr": tune.loguniform(1e-4, 1e-2),
    "weight_decay": tune.loguniform(1e-6, 1e-2),
    "optimizer": tune.choice(["adam", "sgd"]),
    "activation": tune.choice(["relu", "leakyrelu"]),
    "epochs": tune.choice([5, 10, 20]),
    "batch_size": tune.choice([32, 64, 128])
}

'''
def tune_model(config):

    # Initialize model
    model = SLR_CNN(
        in_channels = config["in_channels"],
        input_height = config["input_height"],
        input_width = config["input_width"],
        channel_list = config["channel_list"],
        kernel_size = config["kernel_size"],
        padding = config["padding"],
        pool_kernel = config["pool_kernel"],
        dropout_rate = config["dropout_rate"],
        fc_hidden_dim = config["fc_hidden_dim"],
        activation = config["activation"],
    )

    # Setup model
    c = ModelConfig(model)
    c.set_optimizer(
        lr = config["lr"],
        weight_decay = config["weight_decay"],
        optimizer_type = config["optimizer"]
    )
    c.set_criterion()
    c.to_device()

    # Data setup
    data = data_setup.DataSetup(config["batch_size"], path = config.PT_SAVE_DIR)

    val_best_loss = 0.0
    for epoch in range(config["epochs"]):
        c.model.train()
        t = train.Train(
            data,
            epoch,
            config["epochs"],
            c
        )
        t.run()
        c.model.eval()
        val = validate.Validation(
            data,
            epoch,
            config["epochs"],
            c
        )
        val_best_loss = val.run()
        tune.report(loss = val.epoch_loss)
        
def main():
    tuner = Tuner(
        tune_model,
        param_space = SEARCH_SPACE,
        tune_config = tune.TuneConfig(
            num_samples = 500,
            scheduler = ASHAScheduler(metric = "loss", mode = "min")
        ),
    )
    result = tuner.fit()
    
    #Log results to .csv
    df = result.get_dataframe()
    df.to_csv(config.HPARAMS_SEARCH_SAVE_DIR + 'test_run.csv', index = False)
'''

def check_trial_prune(model, search_params, trial):
    with torch.no_grad():
        dummy_input = torch.zeros(
            1,
            search_params["in_channels"],
            search_params["input_height"],
            search_params["input_width"]
        )
        try:
            dummy_output = model.conv_layers(dummy_input)
        except Exception as e:
            print(f"[Trial Pruned] Model forward failed: {e}")
            raise optuna.exceptions.TrialPruned()

        h, w = dummy_output.shape[2], dummy_output.shape[3]
        if h <= 0 or w <= 0:
            print(f"[Trial Pruned] Invalid conv output size: ({h}, {w})")
            raise optuna.exceptions.TrialPruned()
        

# Check if the model is invalid
#check_trial_prune(model, search_params, trial)

def objective(trial):

    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    padding = kernel_size // 2  # Ensures "same" padding behavior

    search_params = {
        "in_channels": 6,
        "input_height": 48,
        "input_width": 24,
        "channel_list": trial.suggest_categorical("channel_list", [[16, 32, 64], [32, 64, 128], [16, 32, 64, 128]]), #  [16, 32, 64, 128]
        "kernel_size": kernel_size,
        "padding":  trial.suggest_categorical("padding", [0, 1]),
        "pool_kernel": trial.suggest_categorical("pool_kernel", [2, 3]),
        "dropout_rate": trial.suggest_categorical("dropout_rate", [0, 0.1, 0.5]),
        "fc_hidden_dim": trial.suggest_categorical("fc_hidden_dim", [64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log = True), 
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log = True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
        "activation": trial.suggest_categorical("activation", [nn.ReLU, nn.LeakyReLU]),
        "batchnorm":trial.suggest_categorical("batchnorm", [False, True]),
        "epochs": trial.suggest_categorical("epochs", [5, 10, 12, 15, 20, 25, 30]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "flip_prob": trial.suggest_categorical("flip_prob", [0.0, 0.25, 0.5, 0.75, 1])
    }

    # Build model
    try:
        model = SLR_CNN(
            in_channels = search_params["in_channels"],
            input_height = search_params["input_height"],
            input_width = search_params["input_width"],
            channel_list = search_params["channel_list"],
            kernel_size = search_params["kernel_size"],
            padding = search_params["padding"],
            pool_kernel = search_params["pool_kernel"],
            dropout_rate = search_params["dropout_rate"],
            fc_hidden_dim = search_params["fc_hidden_dim"],
            activation = search_params["activation"],
            batchnorm = search_params["batchnorm"]
        )
    except RuntimeError as e:
        print(
            f"[Trial {trial.number}] Skipped due to model construction error: {e}"
        )
        raise optuna.exceptions.TrialPruned()
    


    c = ModelConfig(model)
    c.set_optimizer(
        lr = search_params["lr"],
        weight_decay = search_params["weight_decay"],
        optimizer_type = search_params["optimizer"]
    )
    c.set_criterion()
    c.to_device()

    # Path to data
    channels_str = "_".join(config.INPUT_CHANNELS)
    data_path = config.PT_SAVE_DIR + "train_val_data_with_%s_channels_%s.pt" % (channels_str, config.PT_SAVE_STR)
    data = DataSetup(
        search_params["batch_size"], 
        data_path = data_path
    )

    # Transform train data
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p = search_params["flip_prob"]),
    ])
    data.get_loaders(transform = train_transform)
    data.create_dataloaders()

    val_best_loss = float("inf")
    for epoch in range(search_params["epochs"]):

        # Train
        c.model.train()
        t = Train(
            data, 
            epoch, 
            search_params["epochs"], 
            c
        )
        t.run()

        # Validate
        c.model.eval()
        val = Validation(
            data, 
            epoch, 
            search_params["epochs"], 
            val_best_loss, 
            c
        )
        
        val_best_loss = val.run()
        print('Val Best Loss: %01f' % val_best_loss)
        print('Epoch loss: %01f' % val.epoch_loss)

    return val_best_loss  # Optuna minimizes this
'''
def objective(trial):
    search_params = {
        "in_channels": 6,
        "input_height": 48,
        "input_width": 24,
        "channel_list": trial.suggest_categorical("channel_list", [[16, 32, 64], [32, 64, 128], [64, 128, 256], [16, 32, 64, 128], [32, 64, 128, 256]]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "padding": trial.suggest_categorical("padding", [0, 1]),
        "pool_kernel": trial.suggest_categorical("pool_kernel", [2, 3]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0, 0.1, 0.5),
        "fc_hidden_dim": trial.suggest_categorical("fc_hidden_dim", [64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log = True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
        "activation": trial.suggest_categorical("activation", [nn.ReLU, nn.LeakyReLU]),
        "epochs": trial.suggest_categorical("epochs", [5, 10, 12, 15, 20]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "flip_prob": trial.suggest_categorical("flip_prob", [0.0, 0.25, 0.5, 0.75, 1])
    }

    
    # Build model
    model = SLR_CNN(
        in_channels=search_params["in_channels"],
        input_height=search_params["input_height"],
        input_width=search_params["input_width"],
        channel_list=search_params["channel_list"],
        kernel_size=search_params["kernel_size"],
        padding=search_params["padding"],
        pool_kernel=search_params["pool_kernel"],
        dropout_rate=search_params["dropout_rate"],
        fc_hidden_dim=search_params["fc_hidden_dim"],
        activation=search_params["activation"],
    )

    # Model on Juypter notebook
    model = SLR_CNN(
        in_channels = 6,
        input_height = 48,
        input_width = 24,
        channel_list = [16, 32, 64, 128],
        kernel_size = 3,
        padding = 1,
        pool_kernel=2,
        dropout_rate=0,
        fc_hidden_dim = 64,
        activation = nn.ReLU,


    # Check if the model is invalid
    check_trial_prune(model, search_params, trial)

    c = ModelConfig(model)
    c.set_optimizer(
        lr = 0.001,
        weight_decay = 0,
        optimizer_type = "adam"
    )
    c.set_criterion()
    c.to_device()

    # Path to data
    channels_str = "_".join(config.INPUT_CHANNELS)
    data_path = config.PT_SAVE_DIR + "train_val_data_with_%s_channels_%s.pt" % (
        channels_str, config.PT_SAVE_STR
    )
    batch_size = 64
    data = DataSetup(
        batch_size, 
        data_path = data_path
    )
    #print(data)
    #print(data.shape)

    # Transform train data
    train_transform = transforms.RandomHorizontalFlip(p = 0.25)
    data.get_loaders(transform = train_transform)
    data.create_dataloaders()

    val_best_loss = float("inf")
    n_epochs = 15
    for epoch in range(n_epochs):

        # Train
        c.model.train()
        t = Train(data, epoch, n_epochs, c)
        t.run()

        # Validate
        c.model.eval()
        val = Validation(data, epoch, n_epochs, val_best_loss, c)
        val_best_loss = val.run()
        print('Val Best Loss: %01f' % val_best_loss)
        print('Epoch loss: %01f' % val.epoch_loss)

    return val_best_loss  # Optuna minimizes this
'''
def main():
    study = optuna.create_study(direction = "minimize")
    study.optimize(objective, n_trials = 500)  # Try 50 combinations

    print("Best trial:")
    print(study.best_trial)

    # Save results
    df = study.trials_dataframe()
    df.to_csv(config.HPARAMS_SEARCH_SAVE_DIR + "optuna_results.csv", index=False)

if __name__ == "__main__":
    main()