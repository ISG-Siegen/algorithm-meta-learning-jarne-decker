import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

from config import *

log = logging.getLogger("meta-learning-model")

def train_evaluate_meta_learner_user_features(
        df_merged_meta_data: pd.DataFrame, 
        train_user_ids: np.ndarray, 
        test_user_ids: np.ndarray, 
        dataset_name: str,
        meta_learner_config: dict
    ):

    # Get Model Name out of the configuration
    model_name = "Unknown Model"
    for name, config in META_LEARNER_GRID.items():
        if isinstance(meta_learner_config['model'], config['model'].__class__):
            model_name = name; break

    log.info(f"Training/Evaluating Meta-Learner '{model_name}' (User Features Only) for {dataset_name}...")
    
    performance_cols = [col for col in df_merged_meta_data.columns if col.startswith(PERFORMANCE_METRIC_PREFIX)]
    user_meta_feature_cols = [col for col in df_merged_meta_data.columns if col not in performance_cols and col != USER_ID_COLUMN]

    # Split data based on the provided user ID lists
    train_indices = df_merged_meta_data[USER_ID_COLUMN].isin(train_user_ids)
    test_indices = df_merged_meta_data[USER_ID_COLUMN].isin(test_user_ids)

    X_train_uf = df_merged_meta_data.loc[train_indices, user_meta_feature_cols]
    Y_train_uf = df_merged_meta_data.loc[train_indices, performance_cols]
    X_test_uf = df_merged_meta_data.loc[test_indices, user_meta_feature_cols]
    Y_test_uf = df_merged_meta_data.loc[test_indices, performance_cols]
    
    log.info(f"Meta-dataset (user features) split: Train shape X: {X_train_uf.shape}, Test shape X: {X_test_uf.shape}")

    # --- FEATURE SCALING ---
    scaler = StandardScaler()
    X_train_uf_scaled = scaler.fit_transform(X_train_uf)
    X_test_uf_scaled = scaler.transform(X_test_uf)


    # Model-Training with Hyperparameter Optimization
    base_model = meta_learner_config['model']
    param_dist = meta_learner_config['params']

    # Model creation and HPO
    multi_output_model = MultiOutputRegressor(base_model)

    if param_dist:
        log.info(f"Starting Hyperparameter Optimization for {model_name} (User Features Only)...")
        
        prefixed_param_dist = {f"estimator__{key}": value for key, value in param_dist.items()}

        random_search = RandomizedSearchCV(
            estimator=multi_output_model,
            param_distributions=prefixed_param_dist,
            n_iter=50, cv=3, random_state=RANDOM_SEED, n_jobs=-1,
            scoring='neg_mean_squared_error'
        )
        random_search.fit(X_train_uf_scaled, Y_train_uf)
        
        log.info(f"Best parameters found for {model_name} (User Features Only): {random_search.best_params_}")
        final_meta_learner = random_search.best_estimator_
    else:
        log.info(f"Training {model_name} (User Features Only) with default parameters...")
        final_meta_learner = multi_output_model
        final_meta_learner.fit(X_train_uf_scaled, Y_train_uf)

    # Evaluation
    Y_pred_uf_all_algos = final_meta_learner.predict(X_test_uf_scaled)
    
    actual_performances_ml_uf = []
    correct_predictions_count = 0
    correct_top3_predictions_count = 0 
    
    for i in range(len(X_test_uf)):
        predicted_performances_for_user = Y_pred_uf_all_algos[i]

        top3_predicted_indices = np.argsort(predicted_performances_for_user)[-3:][::-1]
        selected_algo_idx = top3_predicted_indices[0]

        actual_performance_of_chosen_algo = Y_test_uf.iloc[i, selected_algo_idx]
        actual_performances_ml_uf.append(actual_performance_of_chosen_algo)

        actual_best_algo_idx = np.argmax(Y_test_uf.iloc[i].to_numpy())
        if selected_algo_idx == actual_best_algo_idx:
            correct_predictions_count += 1

        if actual_best_algo_idx in top3_predicted_indices:
            correct_top3_predictions_count += 1
    
    num_test_users = len(X_test_uf)

    ml_uf_performance = np.mean(actual_performances_ml_uf) if actual_performances_ml_uf else 0.0
    ml_uf_accuracy = correct_predictions_count / num_test_users  if num_test_users  > 0 else 0.0
    ml_uf_acc_top3 = correct_top3_predictions_count / num_test_users if num_test_users > 0 else 0.0
    
    log.info(f"  Final Performance for '{model_name}' (User Features Only) on {dataset_name}: Avg_Perf={ml_uf_performance:.4f}, Top-1_Acc={ml_uf_accuracy:.2%}")
    
    return ml_uf_performance, ml_uf_accuracy, ml_uf_acc_top3

def train_evaluate_meta_learner_user_algo_features(
        df_merged_meta_data_full: pd.DataFrame, 
        df_algo_features_prepared: pd.DataFrame, 
        train_user_ids: pd.Series, 
        test_user_ids: pd.Series, 
        dataset_name: str,
        meta_learner_config: dict
    ):
    """
    Trains and evaluates a meta-learner (User + Algo Features), optionally with HPO.
    Returns both the average actual performance and the Top-1 selection accuracy.
    """
    model_name = "Unknown Model"
    for name, config in META_LEARNER_GRID.items():
        if isinstance(meta_learner_config['model'], config['model'].__class__):
            model_name = name
            break
            
    log.info(f"Training/Evaluating Meta-Learner '{model_name}' (User + Algo Features) for {dataset_name}...")

    # --- Data Preparation and Splitting ---
    performance_cols = [col for col in df_merged_meta_data_full.columns if col.startswith(PERFORMANCE_METRIC_PREFIX)]
    user_meta_feature_cols = [col for col in df_merged_meta_data_full.columns if col not in performance_cols and col != USER_ID_COLUMN]
    algo_source_feature_cols = [col for col in df_algo_features_prepared.columns if col != 'algorithm_identifier']
    id_vars_for_melt = [USER_ID_COLUMN] + user_meta_feature_cols
    df_melted = pd.melt(df_merged_meta_data_full, id_vars=id_vars_for_melt, value_vars=performance_cols, var_name='algorithm_identifier_perf_col', value_name='actual_performance')
    df_melted['algorithm_identifier'] = df_melted['algorithm_identifier_perf_col'].str.replace(f'^{PERFORMANCE_METRIC_PREFIX}', '', regex=True)


    df_meta_phase2 = pd.merge(df_melted, df_algo_features_prepared, on='algorithm_identifier', how='left')
    df_meta_phase2.dropna(subset=algo_source_feature_cols, inplace=True)
    
    if df_meta_phase2.empty:
        log.error(f"Meta-dataset for Phase 2 is empty after merge/dropna for {dataset_name}.")
        return None, None, None

    train_indices = df_meta_phase2[USER_ID_COLUMN].isin(train_user_ids)
    test_indices = df_meta_phase2[USER_ID_COLUMN].isin(test_user_ids)
    X_meta_phase2_cols = user_meta_feature_cols + algo_source_feature_cols
    
    X_train_uf_af = df_meta_phase2.loc[train_indices, X_meta_phase2_cols]
    Y_train_uf_af = df_meta_phase2.loc[train_indices, 'actual_performance']
    X_test_uf_af = df_meta_phase2.loc[test_indices, X_meta_phase2_cols]
    Y_test_uf_af_ground_truth_table = df_meta_phase2.loc[test_indices, [USER_ID_COLUMN, 'algorithm_identifier', 'actual_performance']]

    if X_train_uf_af.empty or X_test_uf_af.empty:
        log.error(f"Training or testing split for Phase 2 is empty for {dataset_name}.")
        return None, None, None
    
    base_model = meta_learner_config['model']
    param_dist = meta_learner_config['params']
    model_step_name = base_model.__class__.__name__.lower() # e.g., 'ridge', 'lgbmregressor'

    # 1. Define which features are categorical vs. numerical
    categorical_features = ['family', 'learning_paradigm', 'handles_cold_start']
    existing_categorical = [f for f in categorical_features if f in X_train_uf_af.columns]
    numerical_features = [f for f in X_train_uf_af.columns if f not in existing_categorical]
    
    # 2. Create the preprocessor for scaling and encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), existing_categorical)
        ],
        remainder='passthrough'
    )

    # 3. Build the appropriate full pipeline
    if isinstance(base_model, Ridge):
        log.info(f"Building Factorization Machine (PolynomialFeatures + Ridge) pipeline...")
        full_pipeline = make_pipeline(
            preprocessor,
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            base_model
        )
        prefixed_param_dist = {f"ridge__{key}": value for key, value in param_dist.items()}
    else:
        log.info(f"Building standard pipeline for {model_step_name}...")
        full_pipeline = make_pipeline(
            preprocessor,
            base_model
        )
    
    # --- Model Training with HPO ---
    if param_dist:
        prefixed_param_dist = {f"{model_step_name}__{key}": value for key, value in param_dist.items()}
        random_search = RandomizedSearchCV(
            estimator=full_pipeline, 
            param_distributions=prefixed_param_dist,
            n_iter=50, cv=3, random_state=RANDOM_SEED, n_jobs=-1
        )
        random_search.fit(X_train_uf_af, Y_train_uf_af)
        final_meta_learner = random_search.best_estimator_
    else:
        final_meta_learner = full_pipeline
        final_meta_learner.fit(X_train_uf_af, Y_train_uf_af)
    
    # --- Evaluation Loop ---
    actual_performances_ml = []
    correct_top1_predictions_count = 0
    correct_top3_predictions_count = 0
    
    unique_test_user_ids = np.unique(test_user_ids)
    for test_user_id in unique_test_user_ids:
        user_specific_ground_truth_rows = Y_test_uf_af_ground_truth_table[Y_test_uf_af_ground_truth_table[USER_ID_COLUMN] == test_user_id]
        if user_specific_ground_truth_rows.empty: continue
        
        user_specific_rows_in_X_test = X_test_uf_af.loc[user_specific_ground_truth_rows.index]
        
        predicted_perfs = final_meta_learner.predict(user_specific_rows_in_X_test)
        
        # --- ACCURACY CALCULATION ---
        top3_predicted_indices = np.argsort(predicted_perfs)[-3:][::-1]
        best_predicted_idx = top3_predicted_indices[0]
        
        actual_best_idx = np.argmax(user_specific_ground_truth_rows['actual_performance'].to_numpy())
        
        if best_predicted_idx == actual_best_idx:
            correct_top1_predictions_count += 1

        if actual_best_idx in top3_predicted_indices:
            correct_top3_predictions_count += 1

        actual_performance_of_chosen_algo = user_specific_ground_truth_rows.iloc[best_predicted_idx]['actual_performance']
        actual_performances_ml.append(actual_performance_of_chosen_algo)

    # Calculate metrics
    num_test_users = len(unique_test_user_ids)
    ml_performance = np.mean(actual_performances_ml) if actual_performances_ml else 0.0
    ml_accuracy = correct_top1_predictions_count / num_test_users if num_test_users > 0 else 0.0
    ml_accuracy_top3 = correct_top3_predictions_count / num_test_users if num_test_users > 0 else 0.0
    
    log.info(f"  Final Performance for '{model_name}' on {dataset_name}: Avg_Perf={ml_performance:.4f}, Top-1_Acc={ml_accuracy:.2%}")
    
    return ml_performance, ml_accuracy, ml_accuracy_top3