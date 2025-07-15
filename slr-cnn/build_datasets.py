"""
Builds train, validation, and testing datasets
given .pickle file
"""

import pandas as pd
import numpy as np
import sys
import argparse
import torch

from functools import partial
from typing import Callable, List

sys.path.append("./config.py")
from config import TrainConfig

config = TrainConfig()
'''
def allocate_datasets(
    df: pd.DataFrame,
    obs_time_col: str,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    test_frac: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert np.isclose(train_frac + val_frac + test_frac, 1.0), "Fractions must sum to 1."

    """
    Splits input pandas DataFrame into training, validation, and testing sets based
    on unique observation times and user defined split fractions. Also remove the last
    week from the training and validation datasets so that 

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

    # Create column from index because we don't want to remove it
    df = df.copy()
    df["obs_collect_time_utc"] = df.index

    # Find unique ob times
    times = df[obs_time_col] if obs_time_col in df.columns else df.index.to_series()
    unique_times = np.sort(times.unique())

    # Determine length of each dataset
    n_total = len(unique_times)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    # Splitting occurs here
    splits = {
        "train" : unique_times[:n_train], # Training
        "val" : unique_times[n_train:n_train + n_val], # Validation
        "test" : unique_times[n_train + n_val:] # Testing
    }
    df.index.name = None
    print(len(df[df[obs_time_col].isin(splits["train"])].sort_values(by = obs_time_col)))
    print(len(df[df[obs_time_col].isin(splits["val"])].sort_values(by = obs_time_col)))
    print(len(df[df[obs_time_col].isin(splits["test"])].sort_values(by = obs_time_col)))
   
    return tuple(
        df[df[obs_time_col].isin(splits[key])].sort_values(by = obs_time_col)
        for key in ["train", "val", "test"]
    )
'''
def allocate_datasets(
    df: pd.DataFrame,
    obs_time_col: str,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
    buffer_days = 7
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert np.isclose(train_frac + val_frac + test_frac, 1.0), "Fractions must sum to 1."

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
    train_val_buffer_end = train_val_buffer_start + pd.Timedelta(days=buffer_days)

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

    # Return split dataframes
    return tuple(
        df[df[obs_time_col].isin(splits[key])].sort_values(by = obs_time_col)
        for key in ["train", "val", "test"]
    )

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
            #print("Invalid group length of %s; must be 24" % len(group))
            continue
        # For 1-d channels (precipitation, refreezing energy, etc.)
        if channel in channels_1d:
            image_1d = group[[channel]].to_numpy()
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
        print(image.shape)
        images.append(image)
    return np.stack(images)

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

def combine_channels(
    get_input_images_fn: Callable[[str], np.ndarray], 
    channels: List[str]
) -> np.ndarray:
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

def clean_data(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Remove NaNs or infinite values from data

    Parameters:
    ----------
    images : np.ndarray
        Images to be input to CNN [assumes shape of (batch_size, channels, height, width)]
    labels : np.ndarray
        Labels to be input to CNN
    
    Returns:
    -------
    np.ndarray
        Cleaned images and labels
    """
    valid_data = (
        ~np.isnan(images).any(axis = (1, 2, 3)) &
        ~np.isinf(images).any(axis = (1, 2, 3)) &
        ~np.isnan(labels) &
        ~np.isinf(labels)
    ) 

    images_cleaned, labels_cleaned = images[valid_data], labels[valid_data]

    print(
        "Removed %s invalid samples" % str((len(images)) - len(images_cleaned))
    )

    return images_cleaned, labels_cleaned

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
    print("Normalizing images...")
    mean = images.mean(
        axis = (0, 1, 2),
        keepdims = True
    )
    std = images.std(
        axis = (0, 1, 2),
        keepdims = True
    )
    normalized_images = (images - mean) / std

    return normalized_images, mean.squeeze(), std.squeeze()

def apply_normalization(
    images: np.ndarray, 
    mean: List[float], 
    std: List[float]
) -> np.ndarray:
    return (images - mean[None, None, None, :]) / std[None, None, None, :]

def to_tensor(x: np.ndarray) -> torch.Tensor:
    """
    Convert images and labels from np.ndarray to torch.Tensor

    Parameters:
    ----------
    x : np.ndarray
        Numpy array to be convert

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

    # Split data
    train_df, val_df, test_df = allocate_datasets(
        pickle_file,
        obs_time_col = obs_time_col
    )

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
    print("Finished creating images")

    # Clean data
    train_images, train_labels = clean_data(train_images, train_labels)
    val_images, val_labels = clean_data(val_images, val_labels)
    test_images, test_labels = clean_data(test_images, test_labels)

    # Normalize images based on training dataset
    train_images, mean, std = normalize_images(train_images)
    val_images = apply_normalization(val_images, mean, std)
    test_images = apply_normalization(test_images, mean, std)
    
    # Convert to tensors 
    train_images_tensor = to_tensor(train_images)
    train_labels_tensor = to_tensor(train_labels)

    val_images_tensor = to_tensor(val_images)
    val_labels_tensor = to_tensor(val_labels)

    test_images_tensor = to_tensor(test_images)
    test_labels_tensor = to_tensor(test_labels)

    # Save train/validation datasets
    print("Saving datasets")
    channels_str = "_".join(channels)
    torch.save(
        {
            "train_images": train_images_tensor,
            "train_labels": train_labels_tensor,
            "val_images": val_images_tensor,
            "val_labels": val_labels_tensor,
        }, 
        config.PT_SAVE_DIR + "train_val_data_with_%s_channels_%s.pt" % (channels_str, config.PT_SAVE_STR)
    )

    # Save testing dataset
    torch.save(
        {
            "test_images": test_images_tensor,
            "test_labels": test_labels_tensor,
        }, 
        config.PT_SAVE_DIR + "test_dataset_with_%s_channels.pt" % channels_str
    )
    
if __name__ == "__main__":
    main()