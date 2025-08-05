import torch.nn as nn
import pandas as pd
import optuna
import time
import os

from config import TrainConfig
from model_config import ModelConfig
from data_setup import DataSetup
from models import SLR_ViT_scalar
from train import Train
from validate import Validation
from torchvision import transforms
from datetime import datetime

os.environ["RAY_DASHBOARD_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("CUDA version:", torch.version.cuda)
print("Torch version:", torch.__version__)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {gpu_name}")
else:
    print("CUDA not available. Using CPU.")

config = TrainConfig()

# Create directory for each unique run
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_SAVE_DIR = os.path.join(config.HPARAMS_SEARCH_SAVE_DIR, "ViT", f"run_{run_id}")
os.makedirs(RUN_SAVE_DIR, exist_ok = True)

def objective(trial):

    """
    
    """

    # Setup search parameter space
    SEARCH_HPARAMS = {
        "HP_PATCH_SIZE": trial.suggest_categorical("patch_size", [4, 6]),
        "HP_EMB_DIM": trial.suggest_categorical("emb_dim", [64, 128, 256]),
        "HP_DEPTH": trial.suggest_int("depth", 2, 6),
        "HP_N_HEADS": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "HP_MLP_DIM": trial.suggest_categorical("mlp_dim", [128, 256, 512]),
        "HP_DROPOUT": trial.suggest_categorical("dropout_rate", [0]),
        "HP_LR": trial.suggest_float("lr", 1e-4, 1e-2, log = True),
        "HP_WEIGHT_DECAY": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "HP_OPTIMIZER": trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop", "adamw"]),
        "HP_ACTIVATION": trial.suggest_categorical("activation", [nn.GELU, nn.ReLU]),
        "HP_EPOCHS": trial.suggest_categorical("epochs", [100]),
        "HP_BATCH_SIZE": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "HP_FLIP_PROB": trial.suggest_categorical("flip_prob", [0.0, 0.25, 0.5])
    }

    # Build model
    try:
        model = SLR_ViT_scalar(
            in_channels = len(config.INPUT_CHANNELS),
            img_height = config.INPUT_HEIGHT,
            img_width = config.INPUT_WIDTH,
            patch_size=SEARCH_HPARAMS["HP_PATCH_SIZE"],
            emb_dim=SEARCH_HPARAMS["HP_EMB_DIM"],
            depth=SEARCH_HPARAMS["HP_DEPTH"],
            n_heads=SEARCH_HPARAMS["HP_N_HEADS"],
            mlp_dim=SEARCH_HPARAMS["HP_MLP_DIM"],
            n_scalars=len(config.SCALARS),
            activation=SEARCH_HPARAMS["HP_ACTIVATION"],
            dropout=SEARCH_HPARAMS["HP_DROPOUT"]
        )
    except RuntimeError as e:
        print(
            f"[Trial {trial.number}] skipped due to model construction error: {e}"
        )
        raise optuna.exceptions.TrialPruned()
    
    # Model config
    c = ModelConfig(model) 
    c.set_optimizer(
        lr = SEARCH_HPARAMS["HP_LR"],
        weight_decay = SEARCH_HPARAMS["HP_WEIGHT_DECAY"],
        optimizer_type = SEARCH_HPARAMS["HP_OPTIMIZER"]
    )
    c.set_criterion(criterion = "MAE")
    c.to_device()

    # Path to data
    channels_str = "_".join(config.INPUT_CHANNELS)
    data_path = config.PT_SAVE_DIR + "train_val_test_data_%s_channels_%s.pt" % (channels_str, config.PT_SAVE_STR)
    data = DataSetup(
        SEARCH_HPARAMS["HP_BATCH_SIZE"], 
        data_path = data_path
    )

    # Transform train data
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p = SEARCH_HPARAMS["HP_FLIP_PROB"]),
    ])
    data.get_loaders(transform = train_transform)
    data.create_dataloaders()

    # Define empties
    best_val_loss = float("inf")
    model_path = None
    train_losses = []
    val_losses = []
    epoch_times = []
    EPOCHS_NO_IMPROVE = 0
    try:
        for epoch in range(SEARCH_HPARAMS["HP_EPOCHS"]):
            epoch_start = time.time()
            # Train
            c.model.train()
            t = Train(
                data,  
                epoch, 
                SEARCH_HPARAMS["HP_EPOCHS"], 
                c
            )
            t.run()
            train_losses.append(t.epoch_loss)

            # Validate
            c.model.eval()
            val = Validation(
                data, 
                epoch, 
                SEARCH_HPARAMS["HP_EPOCHS"], 
                best_val_loss, 
                c
            )
            current_val_loss = val.run()
            val.reduce_lr()
            val_losses.append(val.epoch_loss)

            epoch_end = time.time()
            epoch_times.append(epoch_end - epoch_start)

            if torch.isnan(torch.tensor(current_val_loss)):
                print(f"[Trial {trial.number}] Pruned due to NaN loss at epoch {epoch + 1}")
                raise optuna.exceptions.TrialPruned()

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_model_state_dict = c.model.state_dict()

                # Save model to unique path per trial
                model_path = os.path.join(RUN_SAVE_DIR, f"ViT_model_trial_{trial.number}.pt")
                torch.save(best_model_state_dict, model_path)
                trial.set_user_attr("best_model_path", model_path)
                EPOCHS_NO_IMPROVE = 0
            else:
                EPOCHS_NO_IMPROVE += 1

            # Early stopping check
            if EPOCHS_NO_IMPROVE >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in val loss.")
                break

        trial.set_user_attr("epoch_times", epoch_times)
        trial.set_user_attr("train_losses", train_losses)
        trial.set_user_attr("val_losses", val_losses)
        if model_path is not None and best_model_state_dict is not None:
            trial.set_user_attr("best_model_path", model_path)
            trial.set_user_attr("best_model_state_dict", best_model_state_dict)
        else:
            print(
            f"[Trial {trial.number}] skipped due to invalid model parameters and/or NaNs for losses"
        )
            raise optuna.exceptions.TrialPruned()

        return best_val_loss
    
    except (ValueError, RuntimeError) as e:
        print(
            f"[Trial {trial.number}] skipped due to invalid model parameters and/or NaNs for losses: {e}"
        )
        raise optuna.exceptions.TrialPruned()

def main():

    start_time = time.time()
    study = optuna.create_study(direction = "minimize")
    study.optimize(objective, n_trials = 500)

    end_time = time.time()
    elapsed = end_time - start_time

    # Print elapsed time in a human-readable format
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"\nStudy completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (hh:mm:ss)"
    )

    print("Best trial: %s" % study.best_trial.number)

    channels = config.INPUT_CHANNELS
    channels_str = "_".join(channels)

    # Save results
    df = study.trials_dataframe()
    df["best_model_path"] = [t.user_attrs.get("best_model_path", "") for t in study.trials]
    df.to_csv(os.path.join(RUN_SAVE_DIR, "ViT_optuna_results_%s_channels_with_scalars.csv" % channels_str), index = False)

    # Retrieve user attributes
    best_trial = study.best_trial
    best_model_state_dict = best_trial.user_attrs["best_model_state_dict"]
    best_model_path = os.path.join(RUN_SAVE_DIR, f"ViT_best_model_trial_{best_trial.number}.pt")
    train_losses = best_trial.user_attrs.get("train_losses", None)
    val_losses = best_trial.user_attrs.get("val_losses", None)

    # Model settings from best trial
    best_model = SLR_ViT_scalar(
        in_channels = len(config.INPUT_CHANNELS),
        n_scalars = len(config.SCALARS),
        input_height = config.INPUT_HEIGHT,
        input_width = config.INPUT_WIDTH,
        channel_list = best_trial.params["channels_list"],
        kernel_size = best_trial.params["kernel_size"],
        padding = best_trial.params["padding"],
        pool_kernel = best_trial.params["pool_kernel"],
        dropout_rate = best_trial.params["dropout_rate"],
        lstm_hidden_dim = best_trial.params["lstm_hidden_dim"],
        lstm_layers = best_trial.params["lstm_layers"],
        fc_hidden_dim = best_trial.params["fc_hidden_dim"],
        activation = best_trial.params["activation"],
        batchnorm = best_trial.params["batchnorm"]
    )

    best_model.load_state_dict(best_model_state_dict)
    torch.save(best_model.state_dict(), best_model_path)

    BEST_TRIAL_MODEL_PATH = os.path.join(RUN_SAVE_DIR, "ViT_overall_best_model_%s_channels_with_scalars.pt" % channels_str)
    torch.save(best_model.state_dict(), BEST_TRIAL_MODEL_PATH)

    all_epoch_times = []
    for t in study.trials:
        times = t.user_attrs.get("epoch_times", [])
        all_epoch_times.extend(times)

    if all_epoch_times:
        avg_epoch_time = sum(all_epoch_times) / len(all_epoch_times)
        print(f"\nAverage epoch time across all trials: {avg_epoch_time:.2f} seconds")
        with open(os.path.join(RUN_SAVE_DIR, "average_epoch_time.txt"), "w") as f:
            f.write(f"{avg_epoch_time:.2f} seconds\n")
    else:
        print("\nNo epoch timing data found in trials.")

    if train_losses is not None and val_losses is not None:
        loss_df = pd.DataFrame({
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses
        })
        loss_df.to_csv(
            os.path.join(
                RUN_SAVE_DIR, 
                "ViT_best_trial_losses_%s_channels_with_scalars.csv" % channels_str
            ), 
            index = False
        )

if __name__ == "__main__":
    main()