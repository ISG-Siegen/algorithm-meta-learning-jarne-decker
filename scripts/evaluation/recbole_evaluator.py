
import pandas as pd
import numpy as np
import os
import shutil
import logging
import time
from config import *
from utils import calculate_ndcg_for_recs
from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils.case_study import full_sort_topk


logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("recbole-evaluator")

def evaluate_recbole_per_user(train_data, test_data, dataset_name_original, k):
    if train_data.empty:
        log.warning(f"RecBole: Train data is empty for {dataset_name_original}, skipping.")
        return pd.DataFrame(columns=['user', 'ndcg', 'model'])

    log.info(f"Starting per-user evaluation for RecBole models for dataset: {dataset_name_original}")
    
    recbole_internal_dataset_id = f"{dataset_name_original}_custom_meta"

    recbole_run_base_path = os.path.join(META_DATA_DIR, f"{dataset_name_original}_recbole_run_temp")

    dataset_files_dir = os.path.join(recbole_run_base_path, "data", recbole_internal_dataset_id)

    if os.path.exists(recbole_run_base_path):
        log.info(f"Cleaning up existing RecBole run base directory: {recbole_run_base_path}")
        try: shutil.rmtree(recbole_run_base_path)
        except Exception as e: log.warning(f"Could not remove {recbole_run_base_path}: {e}")
    os.makedirs(dataset_files_dir, exist_ok=True)
    
    df_for_inter_and_train_cols = {}
    load_col_config_inter = []
    if 'user' in train_data.columns and 'item' in train_data.columns:
        df_for_inter_and_train_cols['user_id'] = train_data['user']
        df_for_inter_and_train_cols['item_id'] = train_data['item']
        load_col_config_inter.extend(['user_id:token', 'item_id:token'])
    else:
        log.error("Train_data missing user/item for RecBole.")
        return pd.DataFrame(columns=['user', 'ndcg', 'model'])

    if 'rating' in train_data.columns:
        df_for_inter_and_train_cols['rating'] = train_data['rating']
        load_col_config_inter.append('rating:float')
    if 'timestamp' in train_data.columns:
        df_for_inter_and_train_cols['timestamp'] = train_data['timestamp']
        load_col_config_inter.append('timestamp:float')

    df_for_inter_and_train = train_data[['user', 'item', 'rating', 'timestamp']].rename(columns={
        'user': 'user_id:token',
        'item': 'item_id:token',
        'rating' : 'rating:float',
        'timestamp': 'timestamp:float'
    })
    df_for_test_file = test_data[['user', 'item']].rename(columns={
        'user': 'user_id:token',
        'item': 'item_id:token'
    })

    inter_file_path = os.path.join(dataset_files_dir, f'{recbole_internal_dataset_id}.inter')
    train_file_path = os.path.join(dataset_files_dir, f'{recbole_internal_dataset_id}.train')
    test_file_path = os.path.join(dataset_files_dir, f'{recbole_internal_dataset_id}.test')

    df_for_inter_and_train.to_csv(inter_file_path, sep='\t', index=False)
    df_for_inter_and_train.to_csv(train_file_path, sep='\t', index=False)
    df_for_test_file.to_csv(test_file_path, sep='\t', index=False)

    log.info(f"RecBole data files created in {dataset_files_dir}")
    log.info(f"RecBole load_col for 'inter' will be: {load_col_config_inter}")

    models_to_run = ALGORITHM_PORTFOLIO["RecBole"]
    all_user_scores_for_dataset = []
    
    for model_name_alias, recbole_model_config in models_to_run.items():
        log.info(f"Processing RecBole model: {model_name_alias} for {recbole_internal_dataset_id}")

        model_class = recbole_model_config['class']

        recbole_model_name_str = model_name_alias

        model_checkpoint_dir = os.path.join(recbole_run_base_path, 'checkpoints', recbole_internal_dataset_id, model_name_alias)
        os.makedirs(model_checkpoint_dir, exist_ok=True)

        config_dict_train = {
            'model': model_class, 
            'dataset': recbole_internal_dataset_id,
            'data_path': os.path.join(recbole_run_base_path, "data"),
            'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']}, 
            'USER_ID_FIELD': 'user_id', 'ITEM_ID_FIELD': 'item_id',
            'RATING_FIELD': 'rating' if any('rating:' in col for col in load_col_config_inter) else None,
            'TIME_FIELD': 'timestamp' if any('timestamp:' in col for col in load_col_config_inter) else None,
            'LABEL_FIELD': None,
            'eval_args': { 'group_by': 'user', 'order':'RO', 'mode': 'full'},
            'metrics': ['NDCG', 'MRR'], 'topk': [10, 20, k], 
            'reproducibility': True, 'seed': RANDOM_SEED, 
            'use_gpu': True, 'gpu_id':'0',
            'train_batch_size': 1024, 'eval_batch_size': 2048,
            'checkpoint_dir': model_checkpoint_dir, 
            'show_progress': False,
        }
        
        if recbole_model_name_str in ["Pop", "ItemKNN", "EASE"]:
            config_dict_train['epochs'] = 1
            config_dict_train['stopping_step'] = 0 
        else: 
            config_dict_train['epochs'] = 50
            config_dict_train['stopping_step'] = 10
        if recbole_model_name_str == "ItemKNN": config_dict_train['k'] = 20 
        if recbole_model_name_str in ["BPR", "LINE", "FISM"]:
            config_dict_train['learning_rate'] = 0.001
            config_dict_train['embedding_size'] = 64

        if recbole_model_name_str == "FPMC":
            config_dict_train['embedding_size'] = 64
            config_dict_train['eval_args']['order'] = 'TO'
            config_dict_train['MAX_ITEM_LIST_LENGTH'] = 50 
        
        keys_to_remove_if_none = ['RATING_FIELD', 'TIME_FIELD', 'LABEL_FIELD']
        for key_to_remove in keys_to_remove_if_none:
            if key_to_remove in config_dict_train and config_dict_train[key_to_remove] is None:
                del config_dict_train[key_to_remove]
        
        log.debug(f"Training config_dict for model {model_name_alias}:\n{config_dict_train}")

        model_pth_file_found = None
        try:
            # 1: Train the model and save checkpoint
            log.info(f"Training RecBole model {model_name_alias}...")
            run_recbole(model=recbole_model_name_str, dataset=recbole_internal_dataset_id, config_dict=config_dict_train)
            
            # Find the saved .pth file
            pth_files = sorted(
                [f for f in os.listdir(model_checkpoint_dir) if f.endswith(".pth")],
                key=lambda x: os.path.getmtime(os.path.join(model_checkpoint_dir, x)),
                reverse=True
            )
            if not pth_files:
                raise FileNotFoundError(f"No .pth model file found in {model_checkpoint_dir} for {model_name_alias}")
            model_pth_file_found = os.path.join(model_checkpoint_dir, pth_files[0])
            log.info(f"Trained model for {model_name_alias} saved at: {model_pth_file_found}")

            # 2: Load the trained model and generate recommendations
            log.info(f"Loading trained model {model_name_alias} from {model_pth_file_found} for recommendation...")

            loaded_config, loaded_model, loaded_dataset, loaded_train_data, loaded_valid_data, loaded_test_data = \
                load_data_and_model(model_file=model_pth_file_found)


            external_user_ids_to_rec_for = test_data['user'].unique().tolist()
            print(external_user_ids_to_rec_for[:20])
            external_user_id_strings = [str(uid) for uid in external_user_ids_to_rec_for]

            internal_uid_series = loaded_dataset.token2id(loaded_dataset.uid_field, external_user_id_strings)

            log.info(f"Generating top-{k} recommendations for {len(internal_uid_series)} users using {model_name_alias}...")
            topk_scores_tensor, topk_iid_list_tensor = full_sort_topk(
                uid_series=internal_uid_series, 
                model=loaded_model,
                test_data=loaded_test_data,
                k=k,
                device=loaded_config['device']
            )

            external_topk_item_list = []
            for i in range(len(internal_uid_series)):
                user_topk_iids_internal = topk_iid_list_tensor[i].cpu().numpy()
                user_topk_items_external = loaded_dataset.id2token(loaded_dataset.iid_field, user_topk_iids_internal)
                external_topk_item_list.append(user_topk_items_external)

            recs_for_calc_list = []

            for i, user_id_original in enumerate(external_user_ids_to_rec_for):
 
                user_top_items = external_topk_item_list[i]
                for rank_idx, item_id_external in enumerate(user_top_items):
                    recs_for_calc_list.append({
                        'user': user_id_original, 
                        'item': item_id_external, 
                        'rank': rank_idx + 1
                        })
            recs_for_calc_df = pd.DataFrame(recs_for_calc_list)
            
            recs_for_calc_df['item'] = recs_for_calc_df['item'].astype(test_data['item'].dtype)


            log.info(f"Calculating per-user nDCG for RecBole model {model_name_alias} (programmatic recs)...")
            per_user_ndcg_df = calculate_ndcg_for_recs(recs_for_calc_df, test_data, k)
            per_user_ndcg_df['model'] = f"RB_{model_name_alias}"
            all_user_scores_for_dataset.append(per_user_ndcg_df)

        except Exception as e:
            log.error(f"Error processing RecBole model {model_name_alias} for {dataset_name_original} (programmatic recs): {e}", exc_info=True)
            empty_ndcg = {uid: 0.0 for uid in test_data['user'].unique()}
            empty_df = pd.Series(empty_ndcg, name='ndcg').to_frame().reset_index().rename(columns={'index':'user'})
            empty_df['model'] = f"RB_{model_name_alias}"
            all_user_scores_for_dataset.append(empty_df)
            
    # Cleanup
    log.info(f"Attempting to clean up RecBole run base directory: {recbole_run_base_path}")
    time.sleep(1) 
    try:
        if os.path.exists(recbole_run_base_path): shutil.rmtree(recbole_run_base_path)
        top_level_saved_dir = 'saved'
        if os.path.exists(top_level_saved_dir) and not os.listdir(top_level_saved_dir):
            try: shutil.rmtree(top_level_saved_dir)
            except OSError: log.warning(f"Could not remove top-level empty '{top_level_saved_dir}', it might be in use.")
        elif os.path.exists(top_level_saved_dir):
            log.warning(f"Top-level '{top_level_saved_dir}' directory exists and is not empty. Manual check advised.")
        log.info(f"Cleanup potentially completed for {recbole_run_base_path}.")
    except Exception as e_other:
        log.warning(f"Error during cleanup of {recbole_run_base_path}: {e_other}")

    final_recbole_results_df = pd.DataFrame(columns=['user', 'ndcg', 'model'])
    if all_user_scores_for_dataset:
        final_recbole_results_df = pd.concat(all_user_scores_for_dataset, ignore_index=True)
    
    print(f"\nRecBole ({dataset_name_original}) Results Summary:")
    print(final_recbole_results_df.head() if not final_recbole_results_df.empty else "No RecBole results.")
    return final_recbole_results_df


def evaluate_recbole_per_user_with_features(train_data, test_data, dataset_name_original, k):
    if train_data.empty:
        log.warning(f"RecBole: Train data is empty for {dataset_name_original}, skipping.")
        return pd.DataFrame(columns=['user', 'ndcg', 'model'])

    log.info(f"Starting per-user evaluation for RecBole models for dataset: {dataset_name_original}")
    
    recbole_internal_dataset_id = f"{dataset_name_original}_custom_meta"
    recbole_run_base_path = os.path.join(META_DATA_DIR, f"{dataset_name_original}_recbole_run_temp")

    dataset_files_dir = os.path.join(recbole_run_base_path, "data", recbole_internal_dataset_id)

    if os.path.exists(recbole_run_base_path):
        log.info(f"Cleaning up existing RecBole run base directory: {recbole_run_base_path}")
        try: shutil.rmtree(recbole_run_base_path)
        except Exception as e: log.warning(f"Could not remove {recbole_run_base_path}: {e}")
    os.makedirs(dataset_files_dir, exist_ok=True)
    
    df_for_inter_and_train_cols = {}
    load_col_config_inter = []
    if 'user' in train_data.columns and 'item' in train_data.columns:
        df_for_inter_and_train_cols['user_id'] = train_data['user']
        df_for_inter_and_train_cols['item_id'] = train_data['item']
        load_col_config_inter.extend(['user_id:token', 'item_id:token'])
    else: 
        log.error("Train_data missing user/item for RecBole.")
        return pd.DataFrame(columns=['user', 'ndcg', 'model'])

    if 'rating' in train_data.columns:
        df_for_inter_and_train_cols['rating'] = train_data['rating']
        load_col_config_inter.append('rating:float')
    if 'timestamp' in train_data.columns:
        df_for_inter_and_train_cols['timestamp'] = train_data['timestamp']
        load_col_config_inter.append('timestamp:float')

    df_for_inter_and_train = train_data[['user', 'item', 'rating', 'timestamp']].rename(columns={
        'user': 'user_id:token',
        'item': 'item_id:token',
        'rating' : 'rating:float',
        'timestamp': 'timestamp:float'
    })
    df_for_test_file = test_data[['user', 'item']].rename(columns={
        'user': 'user_id:token',
        'item': 'item_id:token'
    })

    inter_file_path = os.path.join(dataset_files_dir, f'{recbole_internal_dataset_id}.inter')
    train_file_path = os.path.join(dataset_files_dir, f'{recbole_internal_dataset_id}.train')
    test_file_path = os.path.join(dataset_files_dir, f'{recbole_internal_dataset_id}.test')

    df_for_inter_and_train.to_csv(inter_file_path, sep='\t', index=False)
    df_for_inter_and_train.to_csv(train_file_path, sep='\t', index=False)
    df_for_test_file.to_csv(test_file_path, sep='\t', index=False)

    log.info(f"RecBole data files created in {dataset_files_dir}")
    log.info(f"RecBole load_col for 'inter' will be: {load_col_config_inter}")

    models_to_run = ALGORITHM_PORTFOLIO["RecBole"]
    all_user_scores_for_dataset = []
    
    for model_name_alias, recbole_model_config in models_to_run.items():
        log.info(f"Processing RecBole model: {model_name_alias} for {recbole_internal_dataset_id}")
        # Extract the actual model class from the configuration dictionary
        model_class = recbole_model_config['class']
        # Get the string name of the model for logging and file paths
        recbole_model_name_str = model_name_alias
        
        model_checkpoint_dir = os.path.join(recbole_run_base_path, 'checkpoints', recbole_internal_dataset_id, model_name_alias)
        os.makedirs(model_checkpoint_dir, exist_ok=True)

        config_dict_train = {
            'model': model_class, 
            'dataset': recbole_internal_dataset_id,
            'data_path': os.path.join(recbole_run_base_path, "data"),
            'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']}, 
            'USER_ID_FIELD': 'user_id', 'ITEM_ID_FIELD': 'item_id',
            'RATING_FIELD': 'rating' if any('rating:' in col for col in load_col_config_inter) else None,
            'TIME_FIELD': 'timestamp' if any('timestamp:' in col for col in load_col_config_inter) else None,
            'LABEL_FIELD': None,
            'eval_args': { 'group_by': 'user', 'order':'RO', 'mode': 'full'},
            'metrics': ['NDCG', 'MRR'], 'topk': [10, 20, k],
            'reproducibility': True, 'seed': RANDOM_SEED, 
            'use_gpu': True, 'gpu_id':'0',
            'train_batch_size': 1024, 'eval_batch_size': 2048,
            'checkpoint_dir': model_checkpoint_dir, 
            'show_progress': False,
        }
        
        if recbole_model_name_str in ["Pop", "ItemKNN", "EASE"]:
            config_dict_train['epochs'] = 1
            config_dict_train['stopping_step'] = 0 
        else: 
            config_dict_train['epochs'] = 50
            config_dict_train['stopping_step'] = 10
        if recbole_model_name_str == "ItemKNN": config_dict_train['k'] = 20 
        if recbole_model_name_str in ["BPR", "LINE", "FISM"]:
            config_dict_train['learning_rate'] = 0.001
            config_dict_train['embedding_size'] = 64

        if recbole_model_name_str == "FPMC":
            config_dict_train['embedding_size'] = 64
            config_dict_train['eval_args']['order'] = 'TO'
            config_dict_train['MAX_ITEM_LIST_LENGTH'] = 50 
        
        keys_to_remove_if_none = ['RATING_FIELD', 'TIME_FIELD', 'LABEL_FIELD']
        for key_to_remove in keys_to_remove_if_none:
            if key_to_remove in config_dict_train and config_dict_train[key_to_remove] is None:
                del config_dict_train[key_to_remove]
        
        log.debug(f"Training config_dict for model {model_name_alias}:\n{config_dict_train}")

        model_pth_file_found = None
        try:
            #1: Train the model and save checkpoint
            log.info(f"Training RecBole model {model_name_alias}...")
            start_time = time.time()
            run_recbole(model=recbole_model_name_str, dataset=recbole_internal_dataset_id, config_dict=config_dict_train)
            training_time = time.time() - start_time
            
            pth_files = sorted(
                [f for f in os.listdir(model_checkpoint_dir) if f.endswith(".pth")],
                key=lambda x: os.path.getmtime(os.path.join(model_checkpoint_dir, x)),
                reverse=True
            )
            if not pth_files:
                raise FileNotFoundError(f"No .pth model file found in {model_checkpoint_dir} for {model_name_alias}")
            model_pth_file_found = os.path.join(model_checkpoint_dir, pth_files[0])
            log.info(f"Trained model for {model_name_alias} saved at: {model_pth_file_found}")

            # 2: Load the trained model and generate recommendations
            log.info(f"Loading trained model {model_name_alias} from {model_pth_file_found} for recommendation...")
            loaded_config, loaded_model, loaded_dataset, loaded_train_data, loaded_valid_data, loaded_test_data = \
                load_data_and_model(model_file=model_pth_file_found)
            
            external_user_ids_to_rec_for = test_data['user'].unique().tolist()
            print(external_user_ids_to_rec_for[:20])
            external_user_id_strings = [str(uid) for uid in external_user_ids_to_rec_for]
            internal_uid_series = loaded_dataset.token2id(loaded_dataset.uid_field, external_user_id_strings)

            log.info(f"Generating top-{k} recommendations for {len(internal_uid_series)} users using {model_name_alias}...")
            start_time = time.time()
            topk_scores_tensor, topk_iid_list_tensor = full_sort_topk(
                uid_series=internal_uid_series,
                model=loaded_model,
                test_data=loaded_test_data,
                k=k,
                device=loaded_config['device']
            )
            prediction_time = time.time() - start_time


            external_topk_item_list = []
            for i in range(len(internal_uid_series)):
                user_topk_iids_internal = topk_iid_list_tensor[i].cpu().numpy()
                user_topk_items_external = loaded_dataset.id2token(loaded_dataset.iid_field, user_topk_iids_internal)
                external_topk_item_list.append(user_topk_items_external)

            recs_for_calc_list = []

            for i, user_id_original in enumerate(external_user_ids_to_rec_for):
                user_top_items = external_topk_item_list[i]
                for rank_idx, item_id_external in enumerate(user_top_items):
                    recs_for_calc_list.append({
                        'user': user_id_original, 
                        'item': item_id_external, 
                        'rank': rank_idx + 1
                        })
            recs_for_calc_df = pd.DataFrame(recs_for_calc_list)
            
            recs_for_calc_df['item'] = recs_for_calc_df['item'].astype(test_data['item'].dtype)


            log.info(f"Calculating per-user nDCG for RecBole model {model_name_alias} (programmatic recs)...")
            per_user_ndcg_df = calculate_ndcg_for_recs(recs_for_calc_df, test_data, k)
            per_user_ndcg_df['model'] = f"RB_{model_name_alias}"

            per_user_ndcg_df['traintime'] = training_time
            per_user_ndcg_df['predtime'] = prediction_time

            all_user_scores_for_dataset.append(per_user_ndcg_df)

        except Exception as e:
            log.error(f"Error processing RecBole model {model_name_alias} for {dataset_name_original} (programmatic recs): {e}", exc_info=True)
            empty_ndcg = {uid: 0.0 for uid in test_data['user'].unique()}
            empty_df = pd.Series(empty_ndcg, name='ndcg').to_frame().reset_index().rename(columns={'index':'user'})
            empty_df['model'] = f"RB_{model_name_alias}"
            empty_df['traintime'] = np.nan
            empty_df['predtime'] = np.nan
            all_user_scores_for_dataset.append(empty_df)
            
    log.info(f"Attempting to clean up RecBole run base directory: {recbole_run_base_path}")
    time.sleep(1) 
    try:
        if os.path.exists(recbole_run_base_path): shutil.rmtree(recbole_run_base_path)
        top_level_saved_dir = 'saved'
        if os.path.exists(top_level_saved_dir) and not os.listdir(top_level_saved_dir):
            try: shutil.rmtree(top_level_saved_dir)
            except OSError: log.warning(f"Could not remove top-level empty '{top_level_saved_dir}', it might be in use.")
        elif os.path.exists(top_level_saved_dir):
            log.warning(f"Top-level '{top_level_saved_dir}' directory exists and is not empty. Manual check advised.")
        log.info(f"Cleanup potentially completed for {recbole_run_base_path}.")
    except Exception as e_other:
        log.warning(f"Error during cleanup of {recbole_run_base_path}: {e_other}")

    final_recbole_results_df = pd.DataFrame(columns=['user', 'ndcg', 'model'])
    if all_user_scores_for_dataset:
        final_recbole_results_df = pd.concat(all_user_scores_for_dataset, ignore_index=True)
    
    print(f"\nRecBole ({dataset_name_original}) Results Summary:")
    print(final_recbole_results_df.head() if not final_recbole_results_df.empty else "No RecBole results.")
    return final_recbole_results_df