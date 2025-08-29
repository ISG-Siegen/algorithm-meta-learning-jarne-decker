import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from config import *

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("feature-analyzer-cv")

def analyze_feature_importance_cv(dataset_name="movielens"):
    """
    Performs a k-fold cross-validated feature importance analysis.
    """
    log.info(f"--- Starting Cross-Validated Feature Importance Analysis on: {dataset_name} ---")

    # --- 1. Load and Prepare Data ---
    log.info("Loading and preparing data...")
    df_meta_ground_truth = pd.read_csv(os.path.join(META_DATA_DIR, f"meta_ground_truth_{dataset_name}.csv"))
    df_user_meta_features = pd.read_csv(os.path.join(META_DATA_DIR, f"user_meta_features_{dataset_name}.csv"))
    df_algo_features_raw = pd.read_csv(os.path.join(ALGO_FEATURES_DIR, "algorithm_full_features.csv"))

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
    relevant_feature_cols = [col for col in feature_cols_to_keep if col in df_algo_features_raw.columns]
    df_algo_features = df_algo_features_raw[['algorithm_identifier'] + relevant_feature_cols].copy()


    df_merged_full = pd.merge(df_user_meta_features, df_meta_ground_truth, on=USER_ID_COLUMN)
    
    performance_cols = [col for col in df_merged_full.columns if col.startswith(PERFORMANCE_METRIC_PREFIX)]

    user_meta_feature_cols = [col for col in df_user_meta_features.columns if col != USER_ID_COLUMN]
    
    # Get algorithm features by excluding metadata columns
    non_feature_algo_cols = ['algorithm_identifier', 'library', 'model', 'file_path']
    algo_feature_cols = [col for col in df_algo_features.columns if col not in non_feature_algo_cols]
    
    feature_cols = user_meta_feature_cols + algo_feature_cols
    id_vars_for_melt = [USER_ID_COLUMN] + user_meta_feature_cols
    df_melted = pd.melt(df_merged_full, id_vars=id_vars_for_melt, value_vars=performance_cols, var_name='algorithm_identifier_perf_col', value_name='actual_performance')
    df_melted['algorithm_identifier'] = df_melted['algorithm_identifier_perf_col'].str.replace(f'^{PERFORMANCE_METRIC_PREFIX}', '', regex=True)
    df_meta_final = pd.merge(df_melted, df_algo_features, on='algorithm_identifier', how='left').dropna()

    # --- 2. Define Feature Columns and Model Config ---
    categorical_features = [col for col in ['family', 'learning_paradigm', 'handles_cold_start'] if col in feature_cols]
    numerical_features = [col for col in feature_cols if col not in categorical_features]

    X = df_meta_final[feature_cols]
    y = df_meta_final['actual_performance']

    model_config = META_LEARNER_GRID['LightGBM_Tuned']
    base_model = model_config['model']
    param_dist = model_config['params']
    
    # --- 3. Cross-Validation Loop for Feature Importance ---
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    all_fold_importances = []
    feature_names = None

    log.info(f"Starting {K_FOLDS}-fold CV for feature importance extraction...")
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        log.info(f"--- Running Fold {fold_idx + 1}/{K_FOLDS} ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Build pipeline for this fold
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
        full_pipeline = make_pipeline(preprocessor, base_model)
        
        # Set up and run HPO for this fold
        model_step_name = base_model.__class__.__name__.lower()
        prefixed_param_dist = {f"{model_step_name}__{key}": value for key, value in param_dist.items()}
        random_search = RandomizedSearchCV(
            estimator=full_pipeline, param_distributions=prefixed_param_dist,
            n_iter=50, cv=3, random_state=RANDOM_SEED, n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        
        # Extract importances from the best model of this fold
        best_pipeline = random_search.best_estimator_
        final_model = best_pipeline.named_steps[model_step_name]
        
        # Get feature names from the fitted preprocessor
        if feature_names is None:
            feature_names = best_pipeline.named_steps['columntransformer'].get_feature_names_out()

        all_fold_importances.append(final_model.feature_importances_)

    # --- 4. Aggregate and Process Importances ---
    log.info("Aggregating feature importances across all folds...")
    avg_importances = np.mean(all_fold_importances, axis=0)
    std_importances = np.std(all_fold_importances, axis=0)
    
    df_importances = pd.DataFrame({
        'feature': feature_names,
        'mean_importance': avg_importances,
        'std_importance': std_importances
    }).sort_values(by='mean_importance', ascending=False)

    # Save the aggregated importances to a CSV file
    importance_path = os.path.join(ANALYSIS_DIR, f"feature_importance_cv_{dataset_name}.csv")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    df_importances.to_csv(importance_path, index=False)
    log.info(f"Aggregated feature importances saved to: {os.path.abspath(importance_path)}")

    # --- 5. Create and Save the Plot ---
    log.info("Generating feature importance plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 12))
    
    top_features = df_importances.head(25)
    
    # Plot the average importance with error bars for standard deviation
    top_features.plot(
        kind='barh', x='feature', y='mean_importance', xerr='std_importance',
        ax=ax, color='skyblue', ecolor='gray', capsize=2
    )
    ax.invert_yaxis() # Display most important feature at the top
    ax.set_title(f'Cross-Validated Feature Importance on {dataset_name.capitalize()}', fontsize=16)
    ax.set_xlabel('Mean Importance (Gini)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.get_legend().remove()
    plt.tight_layout()
    
    plot_path = os.path.join(ANALYSIS_DIR, f"feature_importance_cv_{dataset_name}.png")
    fig.savefig(plot_path, dpi=300)
    log.info(f"Feature importance plot saved to: {os.path.abspath(plot_path)}")
    plt.show()

if __name__ == '__main__':
    analyze_feature_importance_cv()