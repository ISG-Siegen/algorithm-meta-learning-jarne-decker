import pandas as pd
import os
import logging
import numpy as np

from config import *
from data_loader import load_and_preprocess_data, split_user_data_temporally

from evaluation.lenskit_evaluator import evaluate_lenskit_per_user_with_features
from evaluation.recbole_evaluator import evaluate_recbole_per_user_with_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("performance-feature-extractor")


def generate_performance_features() -> pd.DataFrame:
    """
    Generates dynamic features for each algorithm by evaluating them on a set
    of probe datasets.
    """
    log.info(f"Starting dynamic feature generation using probe datasets: {PROBE_DATASET_NAMES}")
    
    all_records = []

    for dataset_name in PROBE_DATASET_NAMES:
        log.info(f"--- Evaluating all algorithms on probe dataset: {dataset_name} ---")
        try:
            # 1. Load and split the probe data once per dataset
            full_data = load_and_preprocess_data(dataset_name)
            if full_data.empty: continue
            
            train_data, test_data = split_user_data_temporally(full_data)
            if train_data.empty or test_data.empty: continue
            
            # 2. Evaluate the entire portfolio for each library
            log.info(f"  Running LensKit portfolio on {dataset_name}...")
            lenskit_results_df = evaluate_lenskit_per_user_with_features(train_data, test_data, K_RECOMMENDATIONS)
            
            log.info(f"  Running RecBole portfolio on {dataset_name}...")
            recbole_results_df = evaluate_recbole_per_user_with_features(train_data, test_data, dataset_name, K_RECOMMENDATIONS)

            # 3. Combine the results from both libraries
            combined_results_df = pd.concat([lenskit_results_df, recbole_results_df], ignore_index=True)
            if combined_results_df.empty:
                log.warning(f"No performance results generated for {dataset_name}.")
                continue

            # 4. Aggregate all metrics for each algorithm
            aggregation_rules = {
                'ndcg': 'mean',     
                'traintime': 'first', 
                'predtime': 'first',  
            }
            cols_to_agg = {k: v for k, v in aggregation_rules.items() if k in combined_results_df.columns}
            
            aggregated_metrics = combined_results_df.groupby('model').agg(cols_to_agg)

            # 5. Store each metric in the long-format list
            for algo_identifier, row in aggregated_metrics.iterrows():
                for metric_name, value in row.items():
                    all_records.append({
                        'algorithm_identifier': algo_identifier,
                        'dataset': dataset_name,
                        'metric': metric_name,
                        'value': value
                    })
            log.info(f"  Successfully aggregated dynamic features for {dataset_name}.")

        except Exception as e:
            log.error(f"  Failed to process probe dataset {dataset_name}: {e}", exc_info=True)

    if not all_records:
        log.error("No dynamic features were generated.")
        return pd.DataFrame()

    # 6. Pivot the long-format data into the final wide-format feature table
    log.info("Pivoting records into final feature matrix...")
    long_df = pd.DataFrame(all_records)
    
    # 1. Pivot the data. 'algorithm_identifier' becomes the index.
    wide_df = long_df.pivot_table(
        index='algorithm_identifier',
        columns=['dataset', 'metric'],
        values='value'
    )

    # 2. Flatten the MultiIndex columns into a single string.
    wide_df.columns = [f"{metric}_on_{dataset}" for dataset, metric in wide_df.columns]
    
    # 3. Reset the index to turn 'algorithm_identifier' back into a regular column.
    wide_df.reset_index(inplace=True)
    
    return wide_df.fillna(0.0)