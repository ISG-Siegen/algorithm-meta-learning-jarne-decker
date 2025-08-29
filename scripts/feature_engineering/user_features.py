import pandas as pd
import numpy as np
import os
import logging
from scipy.stats import entropy

from config import *
from data_loader import load_and_preprocess_data, split_users_for_meta_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("user-feature-generator")


def generate_user_meta_features(user_train_data: pd.DataFrame, full_train_data_for_global_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Generates meta-features for each user based on their training interaction data.
    """

    log.info(f"Generating user meta-features from user_train_data with {len(user_train_data)} interactions.")
    if user_train_data.empty:
        log.warning("Input user_train_data is empty. Returning empty DataFrame for meta-features.")
        return pd.DataFrame(columns=['user_id'])

    # Calculate global item popularity 
    item_pop_counts = full_train_data_for_global_stats['item'].value_counts().rename('item_pop_count')

    meta_features_list = []
    for user_id, group in user_train_data.groupby('user'):
        features = {'user_id': user_id} 

        # --- Activity-based ---
        features['num_interactions'] = len(group)
        features['num_unique_items'] = group['item'].nunique()

        # --- Rating-based  ---
        if 'rating' in group.columns and group['rating'].notna().any():
            features['avg_rating'] = group['rating'].mean()
            features['std_rating'] = group['rating'].std(ddof=0)
            features['min_rating'] = group['rating'].min()
            features['max_rating'] = group['rating'].max()
            features['median_rating'] = group['rating'].median()
            rating_counts = group['rating'].value_counts()
            features['rating_entropy'] = entropy(rating_counts) if len(rating_counts) > 1 else 0.0
        else:  # Fill with defaults if no rating data
            features.update({
                'avg_rating': 0.0, 'std_rating': 0.0, 'min_rating': 0.0,
                'max_rating': 0.0, 'median_rating': 0.0, 'rating_entropy': 0.0
            })
            
        # --- Temporal ---
        if 'timestamp' in group.columns and pd.api.types.is_numeric_dtype(group['timestamp']) and group['timestamp'].notna().any():
            min_ts, max_ts = group['timestamp'].min(), group['timestamp'].max()
            features['history_duration_seconds'] = max_ts - min_ts if max_ts > min_ts else 0
            features['first_interaction_ts'] = min_ts
            features['last_interaction_ts'] = max_ts
            features['avg_time_diff_interactions'] = group['timestamp'].sort_values().diff().mean() if len(group) > 1 else 0.0
        else:  # Fill with defaults if no timestamp data
            features.update({
                'history_duration_seconds': 0.0, 'first_interaction_ts': 0.0,
                'last_interaction_ts': 0.0, 'avg_time_diff_interactions': 0.0
            })

        # --- Item-based ---
        user_items_pop_series = group['item'].map(item_pop_counts).dropna()
        if not user_items_pop_series.empty:
            features['avg_item_pop_interacted'] = user_items_pop_series.mean()
            features['median_item_pop_interacted'] = user_items_pop_series.median()
            features['std_item_pop_interacted'] = user_items_pop_series.std(ddof=0)
        else:
            features.update({
                'avg_item_pop_interacted': 0.0, 'median_item_pop_interacted': 0.0,
                'std_item_pop_interacted': 0.0
            })
            
        meta_features_list.append(features)

    df_user_meta_features = pd.DataFrame(meta_features_list).fillna(0)
    
    if 'user_id' in df_user_meta_features.columns:
        cols = ['user_id'] + [col for col in df_user_meta_features.columns if col != 'user_id']
        df_user_meta_features = df_user_meta_features[cols]

    log.info(f"Generated {df_user_meta_features.shape[1]-1} meta-features for {len(df_user_meta_features)} users.")
    return df_user_meta_features


if __name__ == '__main__':
    """
    This block allows running the script directly to generate and save
    user meta-feature files for all datasets defined in config.py.
    """
    log.info("--- Running User Meta-Feature Generation Standalone ---")
    
    for dataset_name in DATASET_NAMES:
        log.info(f"\n{'='*20} Processing dataset: {dataset_name.upper()} {'='*20}")
        
        output_path = os.path.join(META_DATA_DIR, f"user_meta_features_{dataset_name}.csv")
        
        if os.path.exists(output_path):
            log.info(f"User meta-features file already exists for {dataset_name} at {output_path}. Skipping.")
            continue

        full_data = load_and_preprocess_data(dataset_name)
        if full_data.empty:
            log.error(f"Cannot generate features for {dataset_name} as no data was loaded.")
            continue
        
        train_data_for_features, _ = split_users_for_meta_features(full_data, test_ratio=0.2)
        if train_data_for_features.empty:
            log.error(f"Cannot generate features for {dataset_name} as training split is empty.")
            continue

        df_features = generate_user_meta_features(train_data_for_features, train_data_for_features)
        
        if not df_features.empty:
            os.makedirs(META_DATA_DIR, exist_ok=True)
            df_features.to_csv(output_path, index=False)
            log.info(f"User meta-features for {dataset_name} saved to: {os.path.abspath(output_path)}")
        else:
            log.warning(f"No user features were generated for {dataset_name}. No file saved.")