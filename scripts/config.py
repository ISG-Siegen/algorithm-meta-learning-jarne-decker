import os
from lenskit.basic import PopScorer
from lenskit.knn import ItemKNNScorer
import lenskit.als
from lenskit.implicit import BPR as LenskitBPR
from recbole.model.general_recommender import Pop, ItemKNN, BPR as RecboleBPR, EASE
from lenskit import topn_pipeline
from lightgbm import LGBMRegressor
from recbole.model.sequential_recommender import FPMC
from lenskit.knn import UserKNNScorer
from lenskit.als import BiasedMFScorer
from recbole.model.general_recommender import LINE, FISM

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
META_DATA_DIR = os.path.join(RESULTS_DIR, "meta_learning_data")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "meta_learning_analysis")
ALGO_FEATURES_DIR = os.path.join(RESULTS_DIR, "code_features")

# --- General Experiment Constants ---
RANDOM_SEED = 29
K_RECOMMENDATIONS = 10
PERFORMANCE_METRIC_PREFIX = 'perf_'
USER_ID_COLUMN = 'user_id'
K_FOLDS = 5

# --- Datasets to Process ---
DATASET_NAMES = ["movielens", "lastfm", "bookcrossing", "retailrocket", "steam"]

# Datasets for generating performance features
PROBE_DATASET_NAMES = ["amazon-books", "online-retail", "yelp"]

# --- Algorithm Portfolio Definition ---
ALGORITHM_PORTFOLIO = {
    "LensKit": {
        "Pop": {
            "class": PopScorer,
            "pipeline": topn_pipeline(PopScorer())
        },
        "ItemKNN": {
            "class": ItemKNNScorer,
            "pipeline": topn_pipeline(ItemKNNScorer(k=20))
        },
        "ImplicitMF": {
            "class": lenskit.als.ImplicitMFScorer,
            "pipeline": topn_pipeline(lenskit.als.ImplicitMFScorer())
        },
        "BPR": {
            "class": LenskitBPR,
            "pipeline": topn_pipeline(LenskitBPR())
        },
        "UserUser": {
            "class": UserKNNScorer, 
            "pipeline": topn_pipeline(UserKNNScorer(max_nbrs=20))
            },
        "BiasedMF": {
            "class": BiasedMFScorer, 
            "pipeline": topn_pipeline(BiasedMFScorer(embedding_size=50))
            }
    },
    "RecBole": {
        "Pop": {"class": Pop},
        "ItemKNN": {"class": ItemKNN},
        "BPR": {"class": RecboleBPR},
        "EASE": {"class": EASE},
        "FISM": {"class": FISM},
        "LINE": {"class": LINE},
        "FPMC": {"class": FPMC}
    }
}

# --- Meta-Learner Configuration ---
META_LEARNER_TEST_SIZE = 0.2

META_LEARNER_GRID = {

    'LightGBM_Tuned': {
        'model': LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 400, 600, 800, 1000],
            'num_leaves': [20, 30, 40, 50, 60],
            'max_depth': [-1, 10, 20, 30],

            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0, 0.1, 0.5, 1],
        },
        'run_on_user_only': True,
        'run_on_user_algo': True
    },
}

META_LEARNER_FIXED_MODELS = {
        'LightGBM_Fixed_UserAlgo': {
        'model': LGBMRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            n_estimators=1000,
            num_leaves=31,
            max_depth=20,
            learning_rate=0.001
        ),
        'params': None
    },
    'LightGBM_Fixed_UserOnly': {
        'model': LGBMRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            n_estimators=1000,
            num_leaves=20,
            max_depth=30,
            learning_rate=0.01
        ),
        'params': None
    },
}
