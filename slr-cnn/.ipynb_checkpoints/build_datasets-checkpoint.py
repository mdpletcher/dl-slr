"""
PyTorch custom data loaders
"""

import pandas as pd
import numpy as np
import sys
import argparse

from functools import partial
from typing import Callable, List

sys.path.append("./config.py")
from config import TrainConfig

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
    on unique observation times and user defined split fractions

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

    # Create column from index because we don't want to remove it
    df = df.copy()
    df['obs_collect_time_utc'] = df.index

    # Find unique ob times
    times = df[obs_time_col] if obs_time_col in df.columns else df.index.to_series()
    unique_times = np.sort(times.unique())

    # Determine length of each dataset
    n_total = len(unique_times)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    # Splitting occurs here
    splits = {
        'train': unique_times[:n_train], # Training
        'val': unique_times[n_train:n_train + n_val], # Validation
        'test': unique_times[n_train + n_val:] # Testing
    }

    df.index.name = None
    return tuple(
        df[df[obs_time_col].isin(splits[key])].sort_values(by = obs_time_col)
        for key in ['train', 'val', 'test']
    )

def get_input_images(
    df: pd.DataFrame,
    obs_time_col: str, 
    site_col: str,
    analysis_time_col: str,
    levels: list,
    channel: str
) -> np.ndarray:
    """
    Get NxM (time-height) images from ERA5 pd.DataFrame. Also get
    info needed to match images with labels.

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
    for (obs_time_col, site_col), group in df_group:
        group = group.sort_values(by = analysis_time_col)
        if len(group) < 24:
            print('Invalid group length of %s; must be 24' % len(group))
        image = group[['%s%s' % (channel, level) for level in levels]].to_numpy()
        images.append(image.T)
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
    for (_, site_col), group in df_group:
        group = group.sort_values(by = analysis_time_col)
        if len(group) < 24:
            print('Invalid group length of %s; must be 24' % len(group))
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
    img_list = [
        get_input_images_fn(channel) for channel in channels
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

    
'''
def get_input_vectors(
    df
) -> np.ndarray:
    
'''

def main():
    
    # Add parser for .pickle file for input
    parser = argparse.ArgumentParser(
        description = ".pickle file with hourly ERA5 data and labels for training"
    )
    parser.add_argument(
        "pkl_file",
        help = "Path tp .pickle file with ERA5 data and labels for training"
    )
    args = parser.parse_args()
    pickle_file = args.pkl_file

    # Split data
    train_df, val_df, test_df = allocate_datasets(pickle_file)

    # Config settings
    config = TrainConfig()
    channels = config.input_channels
    n_channels = len(channels)
    levels = np.arange(1, config.input_height, 1)

    # Get images for training, validation, and testing datasets
    train_images, val_images, test_images = [
        combine_channels(
            partial(
                get_input_images,
                df = df,
                obs_time_col = 'obs_collect_time_utc',
                site_col = 'site',
                analysis_time_col = 'time',
                levels = levels
            ),
            channels
        )
        for df in [train_df, val_df, test_df]
    ]

    # Get labels for training, validation 










if __name__ == "__main__":
    main()