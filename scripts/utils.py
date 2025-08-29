import logging
import pandas as pd
import numpy as np
from scipy import stats
from config import *
from sklearn.metrics import ndcg_score
from lenskit.data import ItemListCollection

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("evaluation-utils")


def calculate_ndcg_for_recs(recommendations, test_data, k_val):
    """
    Calculates the per-user nDCG@k score for a set of recommendations.
    """

    test_user_items_dict = test_data.groupby('user')['item'].apply(list).to_dict()

    recommendations_df = recommendations
    if isinstance(recommendations, ItemListCollection):
        log.debug("Input to calculate_ndcg_for_recs is ListILC, converting to DataFrame.")
        recommendations_df = recommendations.to_df()

    if not isinstance(recommendations_df, pd.DataFrame):
        log.error(f"Recommendations object is not a DataFrame. Type: {type(recommendations_df)}")
        user_ndcgs_fallback = {uid: 0.0 for uid in test_data['user'].unique()}
        return pd.Series(user_ndcgs_fallback, name='ndcg').to_frame().reset_index().rename(columns={'index':'user'})

    if 'user_id' in recommendations_df.columns and 'user' not in recommendations_df.columns:
        recommendations_df = recommendations_df.rename(columns={'user_id': 'user'})
    elif 'user' not in recommendations_df.columns:
        log.error(f"'user' column not found. Columns: {str(recommendations_df.columns)}")
        user_ndcgs_fallback = {uid: 0.0 for uid in test_data['user'].unique()}
        return pd.Series(user_ndcgs_fallback, name='ndcg').to_frame().reset_index().rename(columns={'index':'user'})

    user_ndcgs = {}
    for user_id, user_recs_group_df in recommendations_df.groupby('user'):
        if user_id in test_user_items_dict:
            true_items_set = set(test_user_items_dict[user_id])

            pred_items_list = []
            item_col_name = 'item' if 'item' in user_recs_group_df.columns else 'item_id'

            if item_col_name not in user_recs_group_df.columns:
                user_ndcgs[user_id] = 0.0
                continue

            if 'score' in user_recs_group_df.columns:
                pred_items_list = user_recs_group_df.sort_values('score', ascending=False)[item_col_name].tolist()
            elif 'rank' in user_recs_group_df.columns:
                pred_items_list = user_recs_group_df.sort_values('rank', ascending=True)[item_col_name].tolist()
            else:
                pred_items_list = user_recs_group_df[item_col_name].tolist() # Assume already ordered

            if len(pred_items_list) < 2:
                user_ndcgs[user_id] = 0.0
                continue

            relevance_scores = [1.0 if item in true_items_set else 0.0 for item in pred_items_list]

            mock_prediction_scores = [float(len(pred_items_list) - i) for i in range(len(pred_items_list))]

            if not relevance_scores:
                 user_ndcgs[user_id] = 0.0
                 continue

            score = ndcg_score(np.asarray([relevance_scores]), np.asarray([mock_prediction_scores]), k=k_val)
            user_ndcgs[user_id] = score
        else:
            user_ndcgs[user_id] = 0.0

    for user_id_test in test_data['user'].unique():
        if user_id_test not in user_ndcgs:
            user_ndcgs[user_id_test] = 0.0

    return pd.Series(user_ndcgs, name='ndcg').to_frame().reset_index().rename(columns={'index':'user'})

def create_algo_identifier(library_str, model_str):
    """Helper to create consistent algorithm identifiers like 'LK_Pop'."""
    lib_prefix = "LK" if str(library_str).lower() == "lenskit" else "RB" if str(library_str).lower() == "recbole" else str(library_str)
    return f"{lib_prefix}_{model_str}"

def get_agg_results(scores):
    """Calculates the mean and 95% confidence interval half-width from a list of scores."""
    if not scores or np.all(np.isnan(scores)):
        return np.nan, np.nan
    
    n = len(scores)
    mean = np.mean(scores)
    
    # Cannot calculate a confidence interval for a single value
    if n < 2:
        return mean, np.nan
        
    # Calculate the standard error of the mean
    std_dev = np.std(scores, ddof=1) # Use ddof=1 for sample standard deviation
    standard_error = std_dev / np.sqrt(n)
    
    # Calculate the 95% confidence interval half-width using the t-distribution
    ci_half_width = stats.t.ppf(0.975, df=n-1) * standard_error
    
    return mean, ci_half_width