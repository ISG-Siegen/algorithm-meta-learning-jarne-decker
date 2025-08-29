import pandas as pd
import os
import logging

from config import *

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("analysis-module")




def calculate_sbs_vbs(df_meta_ground_truth: pd.DataFrame, dataset_name: str) -> tuple:
    """
    Calculates Single Best Solver (SBS), Virtual Best Solver (VBS),
    and the average performance of ALL individual solvers.

    Returns:
        tuple: A tuple containing (sbs_algorithm, sbs_performance, vbs_performance, all_algos_avg_perf_series).
               Returns (None, None, None, None) on failure.
    """
    if df_meta_ground_truth.empty:
        log.warning(f"Input DataFrame for {dataset_name} is empty. Cannot calculate baselines.")
        return None, None, None, None

    performance_cols = [col for col in df_meta_ground_truth.columns if col.startswith(PERFORMANCE_METRIC_PREFIX)]
    
    if not performance_cols:
        log.warning(f"No performance columns (starting with '{PERFORMANCE_METRIC_PREFIX}') found for {dataset_name}.")
        return None, None, None, None

    if USER_ID_COLUMN not in df_meta_ground_truth.columns:
        log.warning(f"User ID column '{USER_ID_COLUMN}' not found for {dataset_name}.")
        return None, None, None, None
        
    log.info(f"Calculating baselines for {dataset_name} using columns: {performance_cols}")

    # --- Calculate SBS---
    avg_perf_per_algo = df_meta_ground_truth[performance_cols].mean()
    
    if avg_perf_per_algo.empty:
        log.warning(f"Could not calculate average performance per algorithm for {dataset_name}.")
        return None, None, None, None

    sbs_performance = avg_perf_per_algo.max()
    sbs_algorithm_col_name = avg_perf_per_algo.idxmax()
    sbs_algorithm_clean_name = sbs_algorithm_col_name.replace(PERFORMANCE_METRIC_PREFIX, "")
    log.info(f"SBS for {dataset_name}: Algorithm='{sbs_algorithm_clean_name}', Performance={sbs_performance:.4f}")

    # --- Calculate VBS ---
    oracle_perf_per_user = df_meta_ground_truth[performance_cols].max(axis=1)
    vbs_performance = oracle_perf_per_user.mean()
    log.info(f"VBS/Oracle for {dataset_name}: Performance={vbs_performance:.4f}")

    return sbs_algorithm_clean_name, sbs_performance, vbs_performance, avg_perf_per_algo

if __name__ == '__main__':
    """
    This block allows running the script directly to generate a summary of
    baseline performances for all datasets defined in config.py.
    """
    log.info("--- Running Baseline Performance Calculation Standalone ---")
    all_baseline_results = []

    for dataset_name in DATASET_NAMES:
        log.info(f"\n--- Processing dataset: {dataset_name} ---")
        input_file_path = os.path.join(META_DATA_DIR, f"meta_ground_truth_{dataset_name}.csv")

        if not os.path.exists(input_file_path):
            log.warning(f"Meta ground truth file not found for {dataset_name} at {input_file_path}. Skipping.")
            continue
        
        try:
            df_meta_ground_truth = pd.read_csv(input_file_path)
            log.info(f"Loaded meta ground truth for {dataset_name}, shape: {df_meta_ground_truth.shape}")
            
            baseline_stats = calculate_sbs_vbs(df_meta_ground_truth, dataset_name)
            if baseline_stats:
                all_baseline_results.append(baseline_stats)
        except Exception as e:
            log.error(f"Error processing file {input_file_path} for dataset {dataset_name}: {e}", exc_info=True)

    if not all_baseline_results:
        log.info("No baseline results were calculated. Exiting.")
    else:
        df_final_baselines = pd.DataFrame(all_baseline_results)
        
        log.info("\n--- Compiled Baseline Performances ---")
        print(df_final_baselines.to_string())

        try:
            os.makedirs(ANALYSIS_DIR, exist_ok=True)
            output_csv_path = os.path.join(ANALYSIS_DIR, "baseline_algorithm_performances.csv")
            df_final_baselines.to_csv(output_csv_path, index=False, float_format='%.4f')
            log.info(f"Baseline performance results saved to: {os.path.abspath(output_csv_path)}")
        except Exception as e:
            log.error(f"Error saving baseline results to CSV: {e}", exc_info=True)