import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import KFold
from config import *

from data_loader import load_and_preprocess_data, split_user_data_temporally
from evaluation.lenskit_evaluator import evaluate_lenskit_per_user
from evaluation.recbole_evaluator import evaluate_recbole_per_user
from feature_engineering.static_features import extract_all_code_features
from feature_engineering.user_features import generate_user_meta_features
from meta_learning.analysis import calculate_sbs_vbs
from meta_learning.model import train_evaluate_meta_learner_user_features, train_evaluate_meta_learner_user_algo_features
from feature_engineering.performance_features import generate_performance_features
from utils import create_algo_identifier, get_agg_results

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("main-pipeline")


def run_step_1_generate_ground_truth():
    """Step 1: Generate per-user performance data for all base algorithms."""
    log.info("### STEP 1: GENERATING META-LEARNING GROUND TRUTH ###")
    for dataset_name in DATASET_NAMES:
        log.info(f"--- Processing dataset: {dataset_name} for Ground Truth ---")
        output_path = os.path.join(META_DATA_DIR, f"meta_ground_truth_{dataset_name}.csv")
        if os.path.exists(output_path) and not OVERWRITE_EXISTING_FILES:
            log.info(f"Ground truth for {dataset_name} already exists. Skipping.")
            continue

        full_data = load_and_preprocess_data(dataset_name)
        if full_data.empty: continue

        train_data, test_data = split_user_data_temporally(full_data)
        if train_data.empty or test_data.empty: continue
        
        lenskit_perf = evaluate_lenskit_per_user(train_data, test_data, K_RECOMMENDATIONS)
        recbole_perf = evaluate_recbole_per_user(train_data, test_data, dataset_name, K_RECOMMENDATIONS)
        
        all_perf_long = pd.concat([lenskit_perf, recbole_perf], ignore_index=True)
        if all_perf_long.empty: continue

        meta_dataset = all_perf_long.pivot_table(index='user', columns='model', values='ndcg').reset_index().fillna(0.0)
        meta_dataset.columns = [f"{PERFORMANCE_METRIC_PREFIX}{col}" if col != 'user' else USER_ID_COLUMN for col in meta_dataset.columns]
        
        os.makedirs(META_DATA_DIR, exist_ok=True)
        meta_dataset.to_csv(output_path, index=False)
        log.info(f"Ground truth for {dataset_name} saved to {os.path.abspath(output_path)}")


def run_step_2_generate_features():
    """
    Step 2: Generates all algorithm and user meta-features.
    """
    log.info("### STEP 2: GENERATING META-FEATURES ###")
    
    # --- Algorithm Features ---
    final_algo_features_path = os.path.join(ALGO_FEATURES_DIR, "algorithm_full_features.csv")
    
    # Check if the final combined file already exists to skip the whole process
    if os.path.exists(final_algo_features_path) and not OVERWRITE_EXISTING_FILES:
        log.info(f"Full algorithm features file already exists at {final_algo_features_path}. Skipping generation.")
    else:
        log.info("--- Generating full set of algorithm features ---")
        
        # 1. Generate Source Code Features 
        code_features_path = os.path.join(ALGO_FEATURES_DIR, "algorithm_code_metrics.csv")
        if not os.path.exists(code_features_path) or OVERWRITE_EXISTING_FILES:
            extract_all_code_features(ALGORITHM_PORTFOLIO, code_features_path)
        else:
            log.info("Source code features file already exists. Loading it.")
        df_code_features = pd.read_csv(code_features_path)
        # Create the standard identifier for merging
        df_code_features['algorithm_identifier'] = df_code_features.apply(
            lambda row: create_algo_identifier(row['library'], row['algorithm_identifier']), axis=1
        )
        
        # 2. Generate Performance Features 
        perf_features_path = os.path.join(ALGO_FEATURES_DIR, "algorithm_performance_features.csv")
        if not os.path.exists(perf_features_path) or OVERWRITE_EXISTING_FILES:
            # This function runs all algorithms on all probe datasets. This can be time-consuming.
            df_perf_features = generate_performance_features()
            df_perf_features.to_csv(perf_features_path, index=False)
            log.info(f"Performance features generated and saved to {perf_features_path}")
        else:
            log.info("Performance features file already exists. Loading it.")
            df_perf_features = pd.read_csv(perf_features_path)

        # 3. Load Conceptual Features ---
        log.info("Loading conceptual algorithm features...")
        conceptual_features_path = os.path.join(ALGO_FEATURES_DIR, "conceptual_features.csv")
        try:
            df_conceptual_features = pd.read_csv(conceptual_features_path)
        except FileNotFoundError:
            log.error(f"Conceptual features file not found at {conceptual_features_path}. Please create it. Proceeding without them.")
            df_conceptual_features = pd.DataFrame({'algorithm_identifier': []})


        # 4. Merge feature sets into a single file
        log.info("Merging source code, performance, and conceptual features...")
        
        # Start with code features, drop redundant columns
        df_full_algo_features = df_code_features
        
        # Merge performance features
        if not df_perf_features.empty:
            df_full_algo_features = pd.merge(df_full_algo_features, df_perf_features, on='algorithm_identifier', how='left')
        
        # Merge conceptual features
        if not df_conceptual_features.empty:
            df_full_algo_features = pd.merge(df_full_algo_features, df_conceptual_features, on='algorithm_identifier', how='left')
        
        # Save the final combined feature file
        df_full_algo_features.to_csv(final_algo_features_path, index=False)
        log.info(f"Full, combined algorithm features saved to: {os.path.abspath(final_algo_features_path)}")

    # --- User Features ---
    for dataset_name in DATASET_NAMES:
        log.info(f"--- Generating user features for dataset: {dataset_name} ---")
        user_feature_path = os.path.join(META_DATA_DIR, f"user_meta_features_{dataset_name}.csv")
        if os.path.exists(user_feature_path) and not OVERWRITE_EXISTING_FILES:
            log.info(f"User features for {dataset_name} already exist. Skipping generation.")
            continue
        
        full_data = load_and_preprocess_data(dataset_name)
        if full_data.empty: continue
        train_data, _ = split_user_data_temporally(full_data)
        if train_data.empty: continue
        df_user_features = generate_user_meta_features(train_data, train_data)
        
        os.makedirs(META_DATA_DIR, exist_ok=True)
        df_user_features.to_csv(user_feature_path, index=False)
        log.info(f"User features for {dataset_name} saved to {os.path.abspath(user_feature_path)}")


def run_step_3_meta_learning_and_analysis():
    """
    Step 3: Loads all data, calculates baselines, runs k-fold CV for meta-learners,
    and saves three separate summary report files.
    """
    log.info("### STEP 3: RUNNING META-LEARNING AND ANALYSIS (WITH K-FOLD CV) ###")
    
    # --- Load Algorithm Features ---
    try:
        df_algo_features_raw = pd.read_csv(os.path.join(ALGO_FEATURES_DIR, "algorithm_full_features.csv"))
        
        # defines which algorithm features are utilized 
        feature_cols_to_keep = [
            'sloc', 'lloc',  'average_cc_file',
            'num_complexity_blocks', 'hal_volume', 'hal_difficulty', 'hal_effort',
            'perf_on_amazon-books', 'perf_on_online-retail', 'perf_on_yelp',
            'predtime_on_amazon-books', 'traintime_on_amazon-books', 'predtime_on_online-retail',
            'traintime_on_online-retail', 'predtime_on_yelp', 'traintime_on_yelp',
            'ast_node_count', 'ast_edge_count', 'ast_avg_degree', 'ast_max_degree',
            'ast_transitivity', 'ast_avg_clustering', 'ast_depth',
            'family', 'learning_paradigm', 'handles_cold_start'
        ]
        existing_feature_cols = [col for col in feature_cols_to_keep if col in df_algo_features_raw.columns]
        df_algo_features_prepared = df_algo_features_raw[['algorithm_identifier'] + existing_feature_cols].copy()
        log.info(f"Algorithm features loaded and prepared. Shape: {df_algo_features_prepared.shape}")
    except Exception as e:
        log.error(f"Error loading algorithm features: {e}", exc_info=True)
        df_algo_features_prepared = None

    overall_results_list = []
    for dataset_name in DATASET_NAMES:
        log.info(f"\n{'='*20} Analyzing Dataset: {dataset_name.upper()} {'='*20}")


        # Load and merge data for this dataset
        try:
            df_meta_ground_truth = pd.read_csv(os.path.join(META_DATA_DIR, f"meta_ground_truth_{dataset_name}.csv"))
            df_user_meta_features = pd.read_csv(os.path.join(META_DATA_DIR, f"user_meta_features_{dataset_name}.csv"))
            df_meta_ground_truth[USER_ID_COLUMN] = df_meta_ground_truth[USER_ID_COLUMN].astype(str)
            df_user_meta_features[USER_ID_COLUMN] = df_user_meta_features[USER_ID_COLUMN].astype(str)
            df_merged_full = pd.merge(df_user_meta_features, df_meta_ground_truth, on=USER_ID_COLUMN, how='inner')
        except FileNotFoundError as e:
            log.warning(f"Missing required data file for {dataset_name}. Skipping analysis. Error: {e}")
            continue
        if df_merged_full.empty: continue

        # Calculate Baselines
        sbs_algo, sbs_perf, vbs_perf, all_algos_perf = calculate_sbs_vbs(df_meta_ground_truth, dataset_name)
        current_results = {"dataset": dataset_name, "sbs_algorithm": sbs_algo, "sbs_performance": sbs_perf, "vbs_performance": vbs_perf}
        if all_algos_perf is not None:
            prefixed_algos_perf = all_algos_perf.copy()
            prefixed_algos_perf.index = "avg_" + prefixed_algos_perf.index
            current_results.update(prefixed_algos_perf.to_dict())

        # K-Fold Cross-Validation for all configured meta-learners
        unique_users = df_merged_full[USER_ID_COLUMN].unique()
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        for model_config_name, model_config in META_LEARNER_GRID.items():
            log.info(f"--- Starting {K_FOLDS}-Fold CV for '{model_config_name}' on {dataset_name} ---")
            fold_perfs_uf, fold_accs_uf1, fold_accs_uf3 = [], [], []
            fold_perfs_uf_af, fold_accs_uf_af1, fold_accs_uf_af3 = [], [], []

            for fold_idx, (train_user_indices, test_user_indices) in enumerate(kf.split(unique_users)):
                log.info(f"  Running Fold {fold_idx + 1}/{K_FOLDS} for '{model_config_name}'...")
                train_users, test_users = unique_users[train_user_indices], unique_users[test_user_indices]

                # --- CONDITIONAL EXECUTION FOR USER-ONLY MODEL ---
                if model_config.get('run_on_user_only', False):
                    uf_perf, uf_acc1, uf_acc3 = train_evaluate_meta_learner_user_features(df_merged_full, train_users, test_users, dataset_name, model_config)
                    fold_perfs_uf.append(uf_perf); fold_accs_uf1.append(uf_acc1); fold_accs_uf3.append(uf_acc3)
                
                # --- CONDITIONAL EXECUTION FOR USER+ALGO MODEL ---
                if model_config.get('run_on_user_algo', False):
                    if df_algo_features_prepared is not None:
                        uf_af_perf, uf_af_acc1, uf_af_acc3 = train_evaluate_meta_learner_user_algo_features(df_merged_full, df_algo_features_prepared, train_users, test_users, dataset_name, model_config)
                        fold_perfs_uf_af.append(uf_af_perf); fold_accs_uf_af1.append(uf_af_acc1); fold_accs_uf_af3.append(uf_af_acc3)
        
            
            # Aggregate results for this model config

            uf_p_mean, uf_p_ci = get_agg_results(fold_perfs_uf)
            uf_a1_mean, uf_a1_ci = get_agg_results(fold_accs_uf1)
            uf_a3_mean, uf_a3_ci = get_agg_results(fold_accs_uf3)
            current_results.update({f'perf_user_only_{model_config_name}': uf_p_mean, f'perf_ci_user_only_{model_config_name}': uf_p_ci,
                                    f'acc1_user_only_{model_config_name}': uf_a1_mean, f'acc1_ci_user_only_{model_config_name}': uf_a1_ci,
                                    f'acc3_user_only_{model_config_name}': uf_a3_mean, f'acc3_ci_user_only_{model_config_name}': uf_a3_ci})
            
            if df_algo_features_prepared is not None and fold_perfs_uf_af:
                uf_af_p_mean, uf_af_p_ci = get_agg_results(fold_perfs_uf_af)
                uf_af_a1_mean, uf_af_a1_ci = get_agg_results(fold_accs_uf_af1)
                uf_af_a3_mean, uf_af_a3_ci = get_agg_results(fold_accs_uf_af3)
                current_results.update({f'perf_user_algo_{model_config_name}': uf_af_p_mean, f'perf_ci_user_algo_{model_config_name}': uf_af_p_ci,
                                        f'acc1_user_algo_{model_config_name}': uf_af_a1_mean, f'acc1_ci_user_algo_{model_config_name}': uf_af_a1_ci,
                                        f'acc3_user_algo_{model_config_name}': uf_af_a3_mean, f'acc3_ci_user_algo_{model_config_name}': uf_af_a3_ci})
        
        overall_results_list.append(current_results)

    # --- Final Processing and Saving of Multiple Report Files ---
    if not overall_results_list:
        log.error("No final results were generated. Exiting analysis."); return

    df_full_summary = pd.DataFrame(overall_results_list)
    if len(df_full_summary) > 1:
        numeric_cols = df_full_summary.select_dtypes(include=np.number).columns
        average_row = df_full_summary[numeric_cols].mean().to_dict()
        average_row['dataset'], average_row['sbs_algorithm'] = 'OVERALL_AVERAGE', 'N/A'
        df_full_summary = pd.concat([df_full_summary, pd.DataFrame([average_row])], ignore_index=True)

    log.info("\n" + "="*25 + " FULL SUMMARY OF PERFORMANCES " + "="*25)
    print(df_full_summary.to_string())

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # --- 1. Main Performance Summary Table ---
    log.info("Generating Main Performance Summary Table...")
    
    # --- Find the best User+Algo model config independently ---
    user_algo_perf_cols = [col for col in df_full_summary.columns if col.startswith('perf_user_algo_')]
    best_ua_model_name = ''
    if user_algo_perf_cols:
        best_ua_perf_col = df_full_summary[user_algo_perf_cols].mean().idxmax()
        best_ua_model_name = best_ua_perf_col.replace('perf_user_algo_', '')
        log.info(f"Identified '{best_ua_model_name}' as the best User+Algo meta-learner config.")

    # --- Find the best User-Only model config independently ---
    user_only_perf_cols = [col for col in df_full_summary.columns if col.startswith('perf_user_only_')]
    best_uo_model_name = ''
    if user_only_perf_cols:
        best_uo_perf_col = df_full_summary[user_only_perf_cols].mean().idxmax()
        best_uo_model_name = best_uo_perf_col.replace('perf_user_only_', '')
        log.info(f"Identified '{best_uo_model_name}' as the best User-Only meta-learner config.")
    
    # --- Assemble the final summary by picking the best of each type ---
    df_main_summary = df_full_summary[['dataset', 'sbs_algorithm', 'sbs_performance', 'vbs_performance']].copy()

    if best_uo_model_name:
        df_main_summary['ml_perf_user_only'] = df_full_summary[f'perf_user_only_{best_uo_model_name}']
        df_main_summary['ml_acc1_user_only'] = df_full_summary[f'acc1_user_only_{best_uo_model_name}']
        df_main_summary['ml_acc3_user_only'] = df_full_summary[f'acc3_user_only_{best_uo_model_name}']
        df_main_summary['ml_perf_ci_user_only'] = df_full_summary.get(f'perf_ci_user_only_{best_uo_model_name}')
        df_main_summary['ml_acc1_ci_user_only'] = df_full_summary.get(f'acc1_ci_user_only_{best_uo_model_name}')
        df_main_summary['ml_acc3_ci_user_only'] = df_full_summary.get(f'acc3_ci_user_only_{best_uo_model_name}')

    if best_ua_model_name:
        df_main_summary['ml_perf_user_algo'] = df_full_summary[f'perf_user_algo_{best_ua_model_name}']
        df_main_summary['ml_acc1_user_algo'] = df_full_summary[f'acc1_user_algo_{best_ua_model_name}']
        df_main_summary['ml_acc3_user_algo'] = df_full_summary[f'acc3_user_algo_{best_ua_model_name}']
        df_main_summary['ml_perf_ci_user_algo'] = df_full_summary.get(f'perf_ci_user_algo_{best_ua_model_name}')
        df_main_summary['ml_acc1_ci_user_algo'] = df_full_summary.get(f'acc1_ci_user_algo_{best_ua_model_name}')
        df_main_summary['ml_acc3_ci_user_algo'] = df_full_summary.get(f'acc3_ci_user_algo_{best_ua_model_name}')
    
    # Calculate gain and gap closed metrics, using .get() for safety
    df_main_summary['gain_over_sbs'] = df_main_summary.get('ml_perf_user_algo', np.nan) - df_main_summary.get('sbs_performance', np.nan)
    df_main_summary['gain_over_user_only'] = df_main_summary.get('ml_perf_user_algo', np.nan) - df_main_summary.get('ml_perf_user_only', np.nan)
    vbs_gap = df_main_summary.get('vbs_performance', np.nan) - df_main_summary.get('sbs_performance', np.nan)
    ml_gain = df_main_summary.get('ml_perf_user_algo', np.nan) - df_main_summary.get('sbs_performance', np.nan)
    df_main_summary['vbs_gap_closed_perc'] = (ml_gain / vbs_gap.replace(0, np.nan) * 100).fillna(0)

    df_main_summary.to_csv(os.path.join(ANALYSIS_DIR, "1_main_performance_summary.csv"), index=False, float_format='%.4f')
    log.info(f"Main summary saved. Path: {os.path.abspath(os.path.join(ANALYSIS_DIR, '1_main_performance_summary.csv'))}")

    # --- 2. Meta-Learner Model Comparison Table ---
    log.info("Generating Meta-Learner Model Comparison Table...")
    ml_comp_cols = ['dataset'] + sorted([col for col in df_full_summary.columns if col.startswith(('perf_user_', 'acc1_user_', 'acc3_user_'))])
    df_ml_comparison = df_full_summary[[col for col in ml_comp_cols if col in df_full_summary.columns]]
    df_ml_comparison.to_csv(os.path.join(ANALYSIS_DIR, "2_meta_learner_model_comparison.csv"), index=False, float_format='%.4f')
    log.info(f"Meta-learner comparison saved. Path: {os.path.abspath(os.path.join(ANALYSIS_DIR, '2_meta_learner_model_comparison.csv'))}")

    # 3. Dataset Statistics Table
    log.info("Generating Dataset Statistics Table...")
    dataset_stats_list = []
    for dataset_name in DATASET_NAMES:
        full_data = load_and_preprocess_data(dataset_name)
        if full_data.empty: continue
        num_users = full_data['user'].nunique()
        num_items = full_data['item'].nunique()
        num_interactions = len(full_data)
        sparsity = 1 - (num_interactions / (num_users * num_items)) if (num_users * num_items) > 0 else 0
        dataset_stats_list.append({'dataset': dataset_name, 'users': num_users, 'items': num_items, 'interactions': num_interactions, 'sparsity': sparsity})
    
    df_dataset_stats = pd.DataFrame(dataset_stats_list)
    df_dataset_stats.to_csv(os.path.join(ANALYSIS_DIR, "3_dataset_statistics.csv"), index=False, float_format='%.4f')
    log.info(f"Dataset statistics saved. Path: {os.path.abspath(os.path.join(ANALYSIS_DIR, '3_dataset_statistics.csv'))}")


if __name__ == "__main__":
    log.info("Starting Experiment Pipeline...")
    # If False, the pipeline uses existing meta-data like ground truth, user and algo features if present
    OVERWRITE_EXISTING_FILES = False

    run_step_1_generate_ground_truth()
    run_step_2_generate_features()
    run_step_3_meta_learning_and_analysis()

    log.info("Experiment Pipeline Finished.")