"""
Builds train, validation, and testing datasets
given .pickle file
"""

import pandas as pd
import numpy as np
import sys
import os
import argparse
import torch

from functools import partial
from typing import Callable, List
from datetime import timedelta
from sklearn import preprocessing

sys.path.append("./config.py")
from config import TrainConfig

config = TrainConfig()

pd.set_option('display.max_rows', 100)

def allocate_datasets(
    df: pd.DataFrame,
    obs_time_col: str,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
    buffer_days = 7,
    fsave: bool = False,
    save_dir: str = "./dataset_splits"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Splits input pandas DataFrame into training, validation, and testing sets based
    on unique observation times and user defined split fractions. Also removes the last
    week from the training and validation datasets so that autocorrelation does not 
    occur during validation and testing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing observations, indexed or labeled by observation time
    
    obs_time_col : str
        Name of column that contains the observation time for grouping
    
    train_frac : float, default = 0.7
        Fraction of unique times to allocate to the training set
    
    val_frac : float, default = 0.2
        Fraction of unique times to allocate to the validation set
    
    test_frac : float, default = 0.1
        Fraction of unique times to allocate to the test set

    Returns
    -------
    tuple of pd.DataFrame
        A tuple containing (train_df, val_df, test_df), each sorted by observation time.
    """
    assert np.isclose(train_frac + val_frac + test_frac, 1.0), "Fractions must sum to 1."
    print(
        "Splitting data into these fractions: Train: %s, Validate: %s, Test: %s" % (train_frac, val_frac, test_frac)
    )

    # Remove Alaska and Canada sites
    df = df[~df["site"].str.contains(r"\b(AK|CAN)\b", regex = True)]

    # Create column for numeric sites for saving as tensors
    df["site_numeric"] = preprocessing.LabelEncoder().fit_transform(list(df["site"]))
    # Map lat/lon to x, y, and z coords 
    # (https://datascience.stackexchange.com/questions/13567/ways-to-deal-with-longitude-latitude-feature/13575#13575)
    df["sin_lat"] = np.sin(np.radians(df["site_lat"]))

    # Create column from index because we don't want to remove it
    df = df.copy()
    df.index.name = None
    if obs_time_col not in df.columns:
        df[obs_time_col] = df.index

    # Find unique ob times
    times = pd.to_datetime(df[obs_time_col].unique())
    unique_times = pd.Series(np.sort(times))

    # Define test set as last 10%
    n_total = len(unique_times)
    n_test = int(n_total * test_frac)
    test_times = unique_times[-n_test:]
    test_start_date = test_times.min()
    val_test_buffer_start = test_start_date - pd.Timedelta(days = buffer_days)
    
    # All times before the val-test buffer
    before_val_test_buffer = unique_times[unique_times < val_test_buffer_start]

    # Now split what's left into train and val with another 7-day buffer
    # Find index of buffer split
    val_start_idx = int(len(before_val_test_buffer) * (train_frac / (train_frac + val_frac)))
    val_start_time = before_val_test_buffer[val_start_idx]
    train_val_buffer_start = val_start_time
    train_val_buffer_end = train_val_buffer_start + pd.Timedelta(days = buffer_days)

    # Create train and val times with buffer
    train_times = before_val_test_buffer[before_val_test_buffer < train_val_buffer_start]
    val_times = before_val_test_buffer[
        (before_val_test_buffer >= train_val_buffer_end) & 
        (before_val_test_buffer < val_test_buffer_start)
    ]

    # Assign to dict
    splits = {
        "train": train_times,
        "val": val_times,
        "test": test_times
    }

    # Allocate datasets
    datasets = {
        key: df[df[obs_time_col].isin(splits[key])].sort_values(by = obs_time_col)
        for key in ["train", "val", "test"]
    }

    return datasets["train"], datasets["val"], datasets["test"]

def get_input_images(
    df: pd.DataFrame,
    obs_time_col: str, 
    site_col: str,
    analysis_time_col: str,
    channels_1d: List[str],
    input_height: int,
    levels: List[str],
    channel: str
) -> np.ndarray:
    """
    Get NxM (time-height) and N (time) images from ERA5 pd.DataFrame

    Parameters:
    ----------
    df : pd.DataFrame
        pandas DataFrame with time, site, and channel information
    obs_time_col : str
        Column with date and time when observation was collected
    site : str
        Column with site names
    analysis_time_col : str
        Column with date and time of analysis or forecast hour
    levels : list
        Levels to be used in the images
    channel : str
        Channel (variable) used to generate image (e.g.,
        temperature, relative humidity profiles)

    Returns:
    -------
    np.ndarray
        Processed images (shape = samples x time x levels)
    """

    images = []
    df_group = df.groupby([obs_time_col, site_col])
    print("Creating %s images..." % channel)
    for (_, _), group in df_group:
        group = group.sort_values(by = analysis_time_col)
        if len(group) < 24:
            print("Invalid group length of %s; must be 24" % len(group))
            continue
        # For 1-d channels (precipitation, refreezing energy, etc.)
        if channel in channels_1d:
            image_1d = group[[channel]].to_numpy()
            # Log scale precipitation
            if channel == "swe_mm_model":
                image_1d = np.log1p(image_1d)
            # Broadcast 1d channels to the input heights
            image = np.repeat(
                image_1d,
                input_height,
                axis = 0
            ).reshape(config.INPUT_HEIGHT, config.INPUT_WIDTH)[::-1, :]
        # 2-d channels (temp, RH, wind speed, etc.)
        else:
            image = group[
                ["%s%02dK" % (channel, level) for level in levels]
            ].to_numpy().T[::-1, :]
        images.append(image)
    return np.stack(images)

def get_scalars(
    df: pd.DataFrame,
    obs_time_col: str, 
    site_col: str,
    analysis_time_col: str,
    scalar_cols: List[str]
) -> np.array:
    """
    Extract scalar values for grouped images

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing all image data and labels
    obs_time_col : str
        Column name for observation time (used for grouping)
    site_col : str
        Column name for site ID (used for grouping)
    analysis_time_col : str
        Column with date and time of analysis or forecast hour
    scalar_col : str
        Name of the column containing the scalars

    Returns:
    --------
    np.ndarray
        Array of shape (n_samples, n_scalar_features)
    """

    scalars = []
    df_group = df.groupby([obs_time_col, site_col])
    print("Getting scalars for columns: %s" % scalar_cols)

    for (_, _), group in df_group:
        group = group.sort_values(by = analysis_time_col)
        if len(group) < 24:
            continue
        scalar_values = [group[col].iloc[0] for col in scalar_cols]
        scalars.append(scalar_values)

    return np.array(scalars, dtype = np.float32)

def get_labels(
    df: pd.DataFrame,
    obs_time_col: str,
    site_col: str,
    analysis_time_col: str,
    label_col: str
) -> np.ndarray:
    """
    Extract labels and time to match the grouped image samples.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing all image data and labels
    obs_time_col : str
        Column name for observation time (used for grouping)
    site_col : str
        Column name for site ID (used for grouping)
    analysis_time_col : str
        Column with date and time of analysis or forecast hour
    label_col : str
        Name of the column containing the labels (target variable)

    Returns:
    -------
    np.ndarray
        Array of labels (samples,)
    """
    
    labels = []
    df_group = df.groupby([obs_time_col, site_col])
    print("Getting labels using %s column" % label_col)
    for (_, site_col), group in df_group:
        group = group.sort_values(by = analysis_time_col)
        if len(group) < 24:
            #print("Invalid group length of %s; must be 24. Check" % len(group))
            continue
        label = group[label_col].iloc[0]
        labels.append(label)
    return np.array(labels)

def get_metadata(
    df: pd.DataFrame,
    obs_time_col: str, 
    site_col: str,
    analysis_time_col: str,
    metadata_cols: List[str]
) -> np.array:
    """
    Extract metadata for grouped images

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing all image data and labels
    obs_time_col : str
        Column name for observation time (used for grouping)
    site_col : str
        Column name for site ID (used for grouping)
    analysis_time_col : str
        Column with date and time of analysis or forecast hour
    metadata_col : str
        Name of the column containing the metadata

    Returns:
    --------
    np.ndarray
        Array of shape (n_samples, n_metadata_cols)
    """

    # Convert time to timestamp, site name to number because you
    # cannot store strings as tensors
    df["obs_collect_time_utc"] = df["obs_collect_time_utc"].apply(lambda x: x.timestamp())
    df_group = df.groupby([obs_time_col, site_col])
    print("Getting metadata for columns: %s" % metadata_cols)

    metadatas = []
    for (_, _), group in df_group:
        group = group.sort_values(by = analysis_time_col)
        if len(group) < 24: 
            continue
        metadata_values = [group[col].iloc[0] for col in metadata_cols]
        metadatas.append(metadata_values)

    return np.array(metadatas, dtype = np.float32)

def combine_channels(
    get_input_images_fn: Callable[[str], np.ndarray], 
    channels: List[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine single channels images into multi-channel images
    
    Parameters:
    ----------
    get_image_fn : function
        Uses get_input_images that uses channel as input
    channels : list of str
        List of channels to combine
    
    Returns:
    -------
    np.ndarray
        Combined image array (samples, channels, height, width)
    """
    print(f"Combining these channels: {", ".join(channels)}")
    img_list = [
        get_input_images_fn(channel = channel) for channel in channels
    ]
    return np.stack(img_list, axis = -1)

def clean_data(
    df: pd.DataFrame, 
    columns: List[str],
    obs_time_col: str,
    site_col: str,
    fsave: bool,
    save_dir: str,
    save_str: str
) -> pd.DataFrame:
    
    df_group = df.groupby([obs_time_col, site_col])
    def clean_group(group):
        return np.isfinite(group[columns]).all().all()

    df_clean = pd.concat(
        [group for _, group in df_group if clean_group(group)],
        ignore_index = True
    )

    if fsave:
        df_clean.to_pickle(save_dir + save_str)

    # Cleaned dataframe
    return df_clean

def normalize_images(images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize images by channel using training data mean and standard deviation

    Parameters:
    ----------
    images: np.ndarray
        Input images of shape (N, H, W, C)
    
    Returns:
    -------
    normalized_images : np.ndarray
        Standardized images with input shape
    mean : np.ndarray
        Per-channel mean used for normalization
    std : np.ndarray
        Per-channel std used for normalization
    """
    print("Normalizing train images...")
    mean = images.mean(axis = (0, 1, 2), keepdims = True)
    std = images.std(axis = (0, 1, 2), keepdims = True)
    normalized_images = (images - mean) / std

    return normalized_images, mean.squeeze(), std.squeeze()

def normalize_scalars(scalars: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize scalars (e.g., lat/lon/elev) using training data mean and std.

    Parameters:
    ----------
    scalars : np.ndarray
        Array of shape (N, D), where D is the number of scalar features (e.g., 3 for lat/lon/elev)

    Returns:
    -------
    normalized_scalars : np.ndarray
        Standardized scalars
    mean : np.ndarray
        Per-feature mean
    std : np.ndarray
        Per-feature std
    """
    print("Normalizing train scalars...")
    mean = scalars.mean(axis = 0, keepdims = True)
    std = scalars.std(axis = 0, keepdims = True)
    normalized_scalars = (scalars - mean) / std

    return normalized_scalars, mean.squeeze(), std.squeeze()

def apply_img_norm(
    images: np.ndarray, 
    mean: List[float], 
    std: List[float]
) -> np.ndarray:
    return (images - mean[None, None, None, :]) / std[None, None, None, :]

def apply_scalar_norm(
    scalars: np.ndarray, 
    mean: np.ndarray,
    std: np.ndarray
):
    return (scalars - mean[None, :]) / std[None, :]

def to_tensor(x: np.ndarray) -> torch.Tensor:
    """
    Convert images and labels from np.ndarray to torch.Tensor

    Parameters:
    ----------
    x : np.ndarray
        Numpy array to be converted

    Returns:
    -------
    Torch.tensor
        Converted tensor with torch.float32 dtype
    """
    return torch.from_numpy(x).float()

def main() -> None:

    """
    Run functions for building datasets
    """
    
    # Add parser for .pickle file for input
    parser = argparse.ArgumentParser(
        description = ".pickle file with hourly ERA5 data and labels for training"
    )
    parser.add_argument(
        "pkl_file",
        help = "Path to .pickle file with ERA5 data and labels for training"
    )
    args = parser.parse_args()
    pickle_file = pd.read_pickle(args.pkl_file)

    obs_time_col = "obs_collect_time_utc"
    site_col = "site"

    # Split data
    train_df, val_df, test_df = allocate_datasets(pickle_file, obs_time_col = obs_time_col)

    print("Original train size:", len(train_df))
    print("Original validation size:", len(val_df))
    print("Original test size:", len(test_df))

    # Remove NaNs and infs from features due to problematic ERA5 data
    features_2d = ["T", "SPD", "Q", "W", "R"]
    cols_to_check = (
        config.CHANNELS_1D +
        config.SCALARS +
        [f"{feat}{lev:02d}K" for feat in features_2d for lev in range(1, config.INPUT_HEIGHT)]
    )
    # Clean data
    train_df = clean_data(
        train_df, 
        cols_to_check, 
        obs_time_col, 
        site_col, 
        fsave = False,
        save_dir = config.SPLIT_DATA_SAVE_DIR,
        save_str = "train_split_data.pkl"
    )
    val_df = clean_data(
        val_df, 
        cols_to_check, 
        obs_time_col, 
        site_col, 
        fsave = False,
        save_dir = config.SPLIT_DATA_SAVE_DIR,
        save_str = "val_split_data.pkl"
    )
    test_df  = clean_data(
        test_df, 
        cols_to_check, 
        obs_time_col, 
        site_col,
        fsave = False,
        save_dir = config.SPLIT_DATA_SAVE_DIR,
        save_str = "test_split_data.pkl"
    )

    print("Cleaned train size:", len(train_df))
    print("Cleaned validation size:", len(val_df))
    print("Cleaned test size:", len(test_df))

    #train_df_events = 

    # Config settings
    channels = config.INPUT_CHANNELS
    n_channels = len(channels)
    levels = np.arange(1, config.INPUT_HEIGHT + 1, 1)

    # Get images for training, validation, and testing datasets
    print("Creating images for training, validation, and testing datasets...")
    train_images, val_images, test_images = [
        combine_channels(
            partial(
                get_input_images,
                df = df,
                obs_time_col = obs_time_col,
                site_col = "site",
                analysis_time_col = "time",
                channels_1d = config.CHANNELS_1D,
                input_height = config.INPUT_HEIGHT,
                levels = levels
            ),
            channels
        )
        for df in [train_df, val_df, test_df]
    ]
    print("Finished creating images")

    # Get labels for training, validation, and testing datasets
    train_labels, val_labels, test_labels = [
        get_labels(
            df,
            obs_time_col = obs_time_col,
            site_col = "site",
            analysis_time_col = "time",
            label_col = "slr"
        )
        for df in [train_df, val_df, test_df]
    ]
    print("Finished getting labels")

    # Get scalars for training, validation, and testing datasets
    train_scalars, val_scalars, test_scalars = [
        get_scalars(
            df,
            obs_time_col = obs_time_col,
            site_col = "site",
            analysis_time_col = "time",
            scalar_cols = config.SCALARS
        )
        for df in [train_df, val_df, test_df]
    ]
    print("Finished getting scalars")

    # Get site metadata
    train_metadata, val_metadata, test_metadata = [
        get_metadata(
            df,
            obs_time_col = obs_time_col,
            site_col = "site",
            analysis_time_col = "time",
            metadata_cols = config.METADATA
        )
        for df in [train_df, val_df, test_df]
    ]
    print("Finished getting metadata")
    print(train_metadata)

    # Normalize images based on training dataset
    train_images, mean, std = normalize_images(train_images)
    val_images = apply_img_norm(val_images, mean, std)
    test_images = apply_img_norm(test_images, mean, std)

    # Normalize scalars based on training dataset
    train_scalars, mean, std = normalize_scalars(train_scalars)
    val_scalars = apply_scalar_norm(val_scalars, mean, std)
    test_scalars = apply_scalar_norm(test_scalars, mean, std)

    # Convert to tensors 
    train_images_tensor = to_tensor(train_images)
    train_labels_tensor = to_tensor(train_labels)
    train_scalars_tensor = to_tensor(train_scalars)
    train_metadata_tensor = to_tensor(train_metadata)

    val_images_tensor = to_tensor(val_images)
    val_labels_tensor = to_tensor(val_labels)
    val_scalars_tensor = to_tensor(val_scalars)
    val_metadata_tensor = to_tensor(val_metadata)

    test_images_tensor = to_tensor(test_images)
    test_labels_tensor = to_tensor(test_labels)
    test_scalars_tensor = to_tensor(test_scalars)
    test_metadata_tensor = to_tensor(test_metadata)

    # Create dicts to save data
    data_to_save = {
        "train": {
            "images": train_images_tensor,
            "labels": train_labels_tensor,
            "scalars": train_scalars_tensor,
            "metadata": train_metadata_tensor,
        },
        "val": {
            "images": val_images_tensor,
            "labels": val_labels_tensor,
            "scalars": val_scalars_tensor,
            "metadata": val_metadata_tensor,
        },
        "test": {
            "images": test_images_tensor,
            "labels": test_labels_tensor,
            "scalars": test_scalars_tensor,
            "metadata": test_metadata_tensor,
        }
    }

    # Save data
    print("Saving datasets")
    channels_str = "_".join(channels)
    save_path = os.path.join(
        config.PT_SAVE_DIR,
        f"train_val_test_data_{channels_str}_channels_{config.PT_SAVE_STR}.pt"
    )
    torch.save(data_to_save, save_path)
    
if __name__ == "__main__":
    main()