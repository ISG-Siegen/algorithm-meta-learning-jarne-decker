import logging
import pandas as pd
from config import *
from lenskit.batch import recommend
from lenskit.data import from_interactions_df
from utils import calculate_ndcg_for_recs
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("lenskit-evaluator")




def evaluate_lenskit_per_user(train_data, test_data, k):
    """Evaluates LensKit algorithms and returns per-user nDCG scores, using topn_pipeline."""

    log.info("Starting per-user evaluation for LensKit models...")

    algorithms_pipeline = ALGORITHM_PORTFOLIO["LensKit"]

    all_user_scores = []

    if 'rating' not in train_data.columns: 
        log.debug("No 'rating' column in train_data for LensKit, proceeding with implicit interpretation.")

    lk_train_data = from_interactions_df(train_data)


    for name, algo_config in algorithms_pipeline.items():
        log.info(f"Training LensKit model: {name}")
        algo_pipe_template = algo_config['pipeline']
        algo_pipe_fitted = algo_pipe_template.clone()
        
        # Fit the pipeline using the LensKit data object
        algo_pipe_fitted.train(lk_train_data) # Use the converted lk_train_data
        
        log.info(f"Generating recommendations for {name}...")
        users_to_rec = test_data['user'].unique()
        
        recs = recommend(algo_pipe_fitted, users_to_rec, k)

        log.info(f"Calculating per-user nDCG for {name}...")
        per_user_ndcg_df = calculate_ndcg_for_recs(recs, test_data, k)
        per_user_ndcg_df['model'] = f"LK_{name}"
        all_user_scores.append(per_user_ndcg_df)

    if not all_user_scores:
        return pd.DataFrame(columns=['user', 'ndcg', 'model'])
        
    return pd.concat(all_user_scores, ignore_index=True)

def evaluate_lenskit_per_user_with_features(train_data, test_data, k):
    """
    Evaluates LensKit algorithms and returns per-user nDCG scores, as well as
    per-algorithm dynamic features (timings).
    """
    log.info("Starting per-user evaluation for LensKit models...")

    algorithms_pipeline = ALGORITHM_PORTFOLIO["LensKit"]

    all_user_scores = []

    lk_train_data = from_interactions_df(train_data)

    for name, algo_config in algorithms_pipeline.items():
        log.info(f"Training LensKit model: {name}")
        algo_pipe_template = algo_config['pipeline']
        algo_pipe_fitted = algo_pipe_template.clone()
        
        # --- 1. Measure Training Time ---
        start_time = time.time()
        algo_pipe_fitted.train(lk_train_data)
        training_time = time.time() - start_time
        
        log.info(f"Generating recommendations for {name}...")
        users_to_rec = test_data['user'].unique()
        
        # --- 2. Measure Prediction Time ---
        start_time = time.time()
        recs = recommend(algo_pipe_fitted, users_to_rec, k)
        prediction_time = time.time() - start_time

        # Calculate per-user NDCG
        log.info(f"Calculating per-user nDCG for {name}...")
        per_user_ndcg_df = calculate_ndcg_for_recs(recs, test_data, k)
        per_user_ndcg_df['model'] = f"LK_{name}"

        
        # --- 4. Add the timing features to the output DataFrame ---
        per_user_ndcg_df['traintime'] = training_time
        per_user_ndcg_df['predtime'] = prediction_time
        
        all_user_scores.append(per_user_ndcg_df)

    if not all_user_scores:
        return pd.DataFrame(columns=['user', 'ndcg', 'model', 'traintime', 'predtime'])
        
    return pd.concat(all_user_scores, ignore_index=True)