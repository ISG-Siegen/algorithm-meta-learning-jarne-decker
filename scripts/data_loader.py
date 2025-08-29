import logging
import os
import pandas as pd
import numpy as np
from config import *


logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("load-data")


def load_and_preprocess_data(dataset_name):
    """Loads and preprocesses the specified dataset."""
    log.info(f"Loading and preprocessing dataset: {dataset_name}")
    if dataset_name == "movielens":
        dataset_path = os.path.join(DATA_DIR, "ml-1m/ratings.csv")
        data = pd.read_csv(dataset_path, sep=",", header=None, names=["user", "item", "rating", "timestamp"])

    elif dataset_name == "retailrocket":
        dataset_path = os.path.join(DATA_DIR, "retailrocket/events.csv")
        data = pd.read_csv(dataset_path, sep=",", header=None, names=["user", "item", "rating", "timestamp"])
        if 'user' in data.columns:
            data['user'] = data['user'].astype(str)
            log.info("Converted 'user' column to string type for consistency.")
        if 'item' in data.columns:
            data['item'] = data['item'].astype(str)
            log.info("Converted 'item' column to string type for consistency.")
        if 'rating' in data.columns:
            data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
            original_len = len(data)
            data.dropna(subset=['rating'], inplace=True)
        if len(data) < original_len:
            log.warning(f"Dropped {original_len - len(data)} rows with non-numeric ratings.")
        log.info("Ensured 'rating' column is numeric (float).")
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_numeric(data['timestamp'], errors='coerce')
            data.dropna(subset=['timestamp'], inplace=True)
            data['timestamp'] = data['timestamp'].astype(np.int64)
            log.info("Ensured 'timestamp' column is numeric (int64).")

    elif dataset_name == "steam":
        dataset_path = os.path.join(DATA_DIR, "steam/ratings.csv")
        data = pd.read_csv(dataset_path, sep=",", header=None, names=["user", "item", "rating", "timestamp"])
        if 'user' in data.columns:
            data['user'] = data['user'].astype(str)
            log.info("Converted 'user' column to string type for consistency.")
        if 'item' in data.columns:
            data['item'] = data['item'].astype(str)
            log.info("Converted 'item' column to string type for consistency.")
        if 'rating' in data.columns:
            data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
            original_len = len(data)
            data.dropna(subset=['rating'], inplace=True)
        if len(data) < original_len:
            log.warning(f"Dropped {original_len - len(data)} rows with non-numeric ratings.")
        log.info("Ensured 'rating' column is numeric (float).")
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_numeric(data['timestamp'], errors='coerce')
            data.dropna(subset=['timestamp'], inplace=True)
            data['timestamp'] = data['timestamp'].astype(np.int64)
            log.info("Ensured 'timestamp' column is numeric (int64).")

    elif dataset_name == "lastfm":
        dataset_path = os.path.join(DATA_DIR, "lastfm/user_artists.csv")
        data = pd.read_csv(dataset_path, sep=",", header=0, names=["user", "item", "rating"])
        if 'timestamp' not in data.columns:
            data['timestamp'] = np.arange(len(data))

    elif dataset_name == "bookcrossing":
        dataset_path = os.path.join(DATA_DIR, "bookcrossing/book_history.csv")
        data = pd.read_csv(dataset_path, sep=",", header=0, names=["user", "item", "rating"])
        if 'timestamp' not in data.columns:
            data['timestamp'] = np.arange(len(data))

    elif dataset_name in ["restaurants", "yelp", "amazon-books", "online-retail"]:
        dataset_path = os.path.join(DATA_DIR, dataset_name, "ratings.csv")
        data = pd.read_csv(dataset_path, sep=",", header=None, names=["user", "item", "rating", "timestamp"])
        if 'user' in data.columns:
            data['user'] = data['user'].astype(str)
            log.info("Converted 'user' column to string type for consistency.")
        if 'item' in data.columns:
            data['item'] = data['item'].astype(str)
            log.info("Converted 'item' column to string type for consistency.")
        if 'rating' in data.columns:
            data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
            original_len = len(data)
            data.dropna(subset=['rating'], inplace=True)
        if len(data) < original_len:
            log.warning(f"Dropped {original_len - len(data)} rows with non-numeric ratings.")
        log.info("Ensured 'rating' column is numeric (float).")
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_numeric(data['timestamp'], errors='coerce')
            data.dropna(subset=['timestamp'], inplace=True)
            data['timestamp'] = data['timestamp'].astype(np.int64)
            log.info("Ensured 'timestamp' column is numeric (int64).")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # For consistency, rename columns
    data.rename(columns={'user_id': 'user', 'item_id': 'item'}, inplace=True)
    
    # user filtering
    user_counts = data.groupby('user')['item'].count()
    data = data[data['user'].isin(user_counts[user_counts >= 10].index)]
    
    log.info(f"Loaded {len(data)} interactions for {data['user'].nunique()} users.")
    return data

def split_user_data_temporally(data, test_ratio=0.2):
    """Splits each user's interaction data based on timestamps."""
    log.info(f"Splitting data temporally with test ratio {test_ratio}...")
    data = data.sort_values(['user', 'timestamp'])
    
    # Get the split point for each user
    def get_split_point(df):
        return df.head(int(len(df) * (1 - test_ratio)))
        
    train_data = data.groupby('user', group_keys=False).apply(get_split_point)
    
    # Test data is whatever is not in train data
    test_data = data.loc[~data.index.isin(train_data.index)]
    
    log.info(f"Train data size: {len(train_data)}, Test data size: {len(test_data)}")
    return train_data, test_data


def split_users_for_meta_features(data, test_ratio=0.2):
    """Splits each user's interaction data based on timestamps.
       Ensures both train and test are non-empty for users present in both.
    """
    if data.empty:
        log.warning("Cannot split empty data.")
        return pd.DataFrame(columns=data.columns), pd.DataFrame(columns=data.columns)

    log.info(f"Splitting data temporally with test ratio {test_ratio}...")
    data = data.sort_values(['user', 'timestamp'])
    
    train_list = []
    test_list = []

    for user_id, group in data.groupby('user'):
        n_items = len(group)
        if n_items < 2:
            log.debug(f"User {user_id} has < 2 items ({n_items}), cannot split. Assigning all to train if n_items=1.")
            if n_items == 1:
                train_list.append(group)
            continue 

        # Ensure at least one item in test, and at least one in train if possible
        n_test_items = max(1, int(n_items * test_ratio))
        n_train_items = n_items - n_test_items
        
        if n_train_items == 0 and n_items > 0 :
            n_train_items = 1
            n_test_items = n_items - 1
            if n_test_items == 0 and n_items > 1:
                n_test_items = 1
                n_train_items = n_items -1


        train_list.append(group.iloc[:n_train_items])
        test_list.append(group.iloc[n_train_items:])

    train_data = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame(columns=data.columns)
    test_data = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame(columns=data.columns)
    
    log.info(f"Train data size: {len(train_data)} ({train_data['user'].nunique()} users), Test data size: {len(test_data)} ({test_data['user'].nunique()} users)")
    return train_data, test_data