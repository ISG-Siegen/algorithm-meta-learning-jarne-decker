import os
import pandas as pd
import numpy as np
import logging
FILE_DIR = "./data/raw/"
OUT_DIR = "./data/processed/"
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("preprocessing")
RANDOM_SEED = 42

def preprocess_movielens(filename="ml-1m/ratings.dat"):
    """Preprocesses the MovieLens-1M dataset."""
    log.info(f"Processing MovieLens-1M from: {filename}")
    input_filepath = os.path.join(FILE_DIR, filename)
    with open(input_filepath, 'r', encoding='latin-1') as file:
        data = "user,item,rating,timestamp\n" + file.read()
        data = data.replace("::", ",")

    outpath = os.path.join(OUT_DIR, "movielens/ratings.csv")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    
    with open(outpath, 'w') as outfile:
        outfile.write(data)
    log.info(f"Saved processed MovieLens-1M data to: {outpath}")

def preprocess_lastfm(filename="lastfm/user_artists.dat"):
    """Preprocesses the LastFM dataset."""
    log.info(f"Processing LastFM from: {filename}")
    input_filepath = os.path.join(FILE_DIR, filename)
    
    df = pd.read_csv(input_filepath, sep='\t')
    df.columns = ['user', 'item', 'rating']
    df['timestamp'] = np.arange(len(df))
    
    outpath = os.path.join(OUT_DIR, "lastfm/ratings.csv")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False)
    log.info(f"Saved processed LastFM data to: {outpath}")


def preprocess_bookcrossing():
    input_filepath = os.path.join(FILE_DIR, "bookcrossing/book_history.dat")
    with open(input_filepath, "r") as file:
        lines = file.read().strip().splitlines()[1:]

    outpath = os.path.join(OUT_DIR, "bookcrossing/book_history.csv")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, 'w') as outfile:
        for line in lines:
            csv_line = line.replace("\t", ",")
            outfile.write(csv_line + "\n")

def preprocess_retailrocket():

    filename = "events.csv"
    input_filepath = os.path.join(FILE_DIR, "retailrocket", filename)
    output_filepath = os.path.join(OUT_DIR, "retailrocket", "events.csv")

    df = pd.read_csv(input_filepath)

    event_weights = {
        'view': 1.0,
        'addtocart': 2.0,
        'transaction': 4.0 
    }
    df['rating'] = df['event'].map(event_weights)

    df.dropna(subset=['rating'], inplace=True)

    df = df.rename(columns={'visitorid': 'user', 'itemid': 'item'})

    aggregation_rules = {
        'rating': 'sum',     
        'timestamp': 'max'  
    }

    df_aggregated = df.groupby(['user', 'item']).agg(aggregation_rules).reset_index()

    final_columns = ['user', 'item', 'rating', 'timestamp']
    df_processed = df_aggregated[final_columns].copy()
    
    df_processed['user'] = df_processed['user'].astype(np.int64)
    df_processed['item'] = df_processed['item'].astype(np.int64)
    df_processed['rating'] = df_processed['rating'].astype(float)
    df_processed['timestamp'] = df_processed['timestamp'].astype(np.int64)
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df_processed.to_csv(output_filepath, index=False)

def preprocess_steam():
    filename = "steam-200k.csv"
    input_filepath = os.path.join(FILE_DIR, "steam", filename)
    output_ratings_filepath = os.path.join(OUT_DIR, "steam", "ratings.csv")
    output_mapping_filepath = os.path.join(OUT_DIR, "steam", "item_mapping.csv")

    column_names = ['user', 'item_name', 'event', 'value', 'ignored']
    df = pd.read_csv(input_filepath, header=None, names=column_names)

    # 2. Filter for 'purchase' and 'play' events
    df = df[df['event'].isin(['purchase', 'play'])].copy()

    # 3. Aggregate events by taking the max value
    df_aggregated = df.groupby(['user', 'item_name'])['value'].max().reset_index()
    df_processed = df_aggregated.rename(columns={'value': 'rating'})
    
    # Create a mapping from unique item names to new, sequential integer IDs
    unique_item_names = df_processed['item_name'].unique()
    item_mapping_df = pd.DataFrame({
        'item_id': np.arange(len(unique_item_names)),
        'item_name': unique_item_names                
    })
    
    os.makedirs(os.path.dirname(output_mapping_filepath), exist_ok=True)
    item_mapping_df.to_csv(output_mapping_filepath, index=False)

    item_name_to_id_dict = pd.Series(item_mapping_df.item_id.values, index=item_mapping_df.item_name).to_dict()

    df_processed['item'] = df_processed['item_name'].map(item_name_to_id_dict)
    
    df_processed = df_processed.drop(columns=['item_name'])

    # Create a dummy timestamp column
    df_processed['timestamp'] = np.arange(len(df_processed))

    final_columns = ['user', 'item', 'rating', 'timestamp']
    df_processed = df_processed[final_columns]
    
    df_processed['user'] = df_processed['user'].astype(np.int64)
    df_processed['item'] = df_processed['item'].astype(np.int64)
    df_processed['rating'] = df_processed['rating'].astype(float)
    df_processed['timestamp'] = df_processed['timestamp'].astype(np.int64)

    os.makedirs(os.path.dirname(output_ratings_filepath), exist_ok=True)
    df_processed.to_csv(output_ratings_filepath, index=False)

def preprocess_amazon_books():

    dataset_folder = "amazon-books"
    filename = "Books_rating.csv"
    input_filepath = os.path.join(FILE_DIR, dataset_folder, filename)
    
    output_dir = os.path.join(OUT_DIR, dataset_folder)
    output_ratings_filepath = os.path.join(output_dir, "ratings.csv")
    output_user_mapping_filepath = os.path.join(output_dir, "user_mapping.csv")
    output_item_mapping_filepath = os.path.join(output_dir, "item_mapping.csv")

    log.info(f"Processing raw Amazon Books data from: {input_filepath}")

    if not os.path.exists(input_filepath):
        log.error(f"Input file not found at: {input_filepath}")
        return
    try:
        df = pd.read_csv(input_filepath)
        df = df[['Id', 'User_id', 'review/score', 'review/time']].copy()
        df.columns = ['item_original', 'user_original', 'rating', 'timestamp']
    except Exception as e:
        log.error(f"Failed to read or process columns from {input_filepath}: {e}")
        return

    log.info(f"Loaded {len(df)} raw rating events.")

    log.info("Aggregating duplicate ratings by taking the mean rating...")
    df_aggregated = df.groupby(['user_original', 'item_original']).agg({
        'rating': 'mean',
        'timestamp': 'max'
    }).reset_index()
    log.info(f"Aggregated data into {len(df_aggregated)} unique user-item ratings.")

    log.info("Sampling 10% of users...")
    unique_users = df_aggregated['user_original'].unique()
    
    np.random.seed(RANDOM_SEED)
    sampled_user_ids = np.random.choice(unique_users, size=int(len(unique_users) * 0.10), replace=False)
    
    df_sampled = df_aggregated[df_aggregated['user_original'].isin(sampled_user_ids)].copy()
    log.info(f"Sampled down to {len(df_sampled)} ratings from {len(sampled_user_ids)} users.")

    if df_sampled.empty:
        log.warning("DataFrame is empty after sampling. No output will be generated.")
        return

    log.info("Converting user and item string IDs to integer IDs...")
    
    final_unique_users = df_sampled['user_original'].unique()
    user_mapping_df = pd.DataFrame({
        'user': np.arange(len(final_unique_users)),
        'user_original': final_unique_users
    })
    
    final_unique_items = df_sampled['item_original'].unique()
    item_mapping_df = pd.DataFrame({
        'item': np.arange(len(final_unique_items)),
        'item_original': final_unique_items
    })

    os.makedirs(output_dir, exist_ok=True)
    user_mapping_df.to_csv(output_user_mapping_filepath, index=False)
    item_mapping_df.to_csv(output_item_mapping_filepath, index=False)
    log.info("Saved user and item ID mappings.")

    user_map_dict = pd.Series(user_mapping_df.user.values, index=user_mapping_df.user_original).to_dict()
    item_map_dict = pd.Series(item_mapping_df.item.values, index=item_mapping_df.item_original).to_dict()

    df_sampled['user'] = df_sampled['user_original'].map(user_map_dict)
    df_sampled['item'] = df_sampled['item_original'].map(item_map_dict)
    
    final_columns = ['user', 'item', 'rating', 'timestamp']
    df_final = df_sampled[final_columns].copy()
    
    df_final['user'] = df_final['user'].astype(np.int64)
    df_final['item'] = df_final['item'].astype(np.int64)
    df_final['rating'] = df_final['rating'].astype(float)
    df_final['timestamp'] = df_final['timestamp'].astype(np.int64)

    df_final.to_csv(output_ratings_filepath, index=False)
    
    log.info(f"Successfully processed Amazon Books data and saved to: {os.path.abspath(output_ratings_filepath)}")

def preprocess_online_retail():

    dataset_folder = "online-retail"
    filename = "OnlineRetail.csv"
    input_filepath = os.path.join(FILE_DIR, dataset_folder, filename)
    
    output_dir = os.path.join(OUT_DIR, dataset_folder)
    output_ratings_filepath = os.path.join(output_dir, "ratings.csv")
    output_user_mapping_filepath = os.path.join(output_dir, "user_mapping.csv")
    output_item_mapping_filepath = os.path.join(output_dir, "item_mapping.csv")

    log.info(f"Processing raw Online Retail data from: {input_filepath}")

    if not os.path.exists(input_filepath):
        log.error(f"Input file not found at: {input_filepath}")
        return

    df = pd.read_csv(input_filepath, encoding='ISO-8859-1')
    log.info(f"Loaded {len(df)} raw events.")

    df.dropna(subset=['CustomerID'], inplace=True)
    log.info(f"Removed rows with missing CustomerID, {len(df)} rows remaining.")
    
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    df = df[df['Quantity'] > 0]
    log.info(f"Removed cancellations and non-positive quantities, {len(df)} rows remaining.")

    df['CustomerID'] = df['CustomerID'].astype(int)

    df['timestamp'] = pd.to_datetime(df['InvoiceDate']).astype('int64') // 10**9

    log.info("Aggregating interactions by summing quantities for each user-item pair...")
    aggregation_rules = {
        'Quantity': 'sum',   
        'timestamp': 'max'     
    }
    df_aggregated = df.groupby(['CustomerID', 'StockCode']).agg(aggregation_rules).reset_index()
    log.info(f"Aggregated data into {len(df_aggregated)} unique user-item interactions.")

    df_processed = df_aggregated.rename(columns={
        'CustomerID': 'user_original',
        'StockCode': 'item_original',
        'Quantity': 'rating'
    })

    log.info("Converting user and item IDs to new integer IDs...")

    user_mapping_df = pd.DataFrame({
        'user': np.arange(df_processed['user_original'].nunique()),
        'user_original': df_processed['user_original'].unique()
    })
    
    item_mapping_df = pd.DataFrame({
        'item': np.arange(df_processed['item_original'].nunique()),
        'item_original': df_processed['item_original'].unique()
    })

    os.makedirs(output_dir, exist_ok=True)
    user_mapping_df.to_csv(output_user_mapping_filepath, index=False)
    item_mapping_df.to_csv(output_item_mapping_filepath, index=False)
    log.info("Saved user and item ID mappings.")

    user_map_dict = pd.Series(user_mapping_df.user.values, index=user_mapping_df.user_original).to_dict()
    item_map_dict = pd.Series(item_mapping_df.item.values, index=item_mapping_df.item_original).to_dict()

    df_processed['user'] = df_processed['user_original'].map(user_map_dict)
    df_processed['item'] = df_processed['item_original'].map(item_map_dict)
    
    final_columns = ['user', 'item', 'rating', 'timestamp']
    df_final = df_processed[final_columns].copy()
    
    df_final['user'] = df_final['user'].astype(np.int64)
    df_final['item'] = df_final['item'].astype(np.int64)
    df_final['rating'] = df_final['rating'].astype(float)
    df_final['timestamp'] = df_final['timestamp'].astype(np.int64)

    df_final.to_csv(output_ratings_filepath, index=False)
    
    log.info(f"Successfully processed Online Retail data and saved to: {os.path.abspath(output_ratings_filepath)}")

def preprocess_yelp():

    dataset_folder = "yelp"
    filename = "yelp_academic_dataset_review.json"
    input_filepath = os.path.join(FILE_DIR, dataset_folder, filename)
    
    output_dir = os.path.join(OUT_DIR, dataset_folder)
    output_ratings_filepath = os.path.join(output_dir, "ratings.csv")
    output_user_mapping_filepath = os.path.join(output_dir, "user_mapping.csv")
    output_item_mapping_filepath = os.path.join(output_dir, "item_mapping.csv")

    log.info(f"Processing raw Yelp data from: {input_filepath}")

    if not os.path.exists(input_filepath):
        log.error(f"Input file not found at: {input_filepath}")
        return

    try:
        df = pd.read_json(input_filepath, lines=True)
    except Exception as e:
        log.error(f"Failed to read JSON file {input_filepath}: {e}")
        return

    df = df[['user_id', 'business_id', 'stars', 'date']].copy()
    df.columns = ['user_original', 'item_original', 'rating', 'timestamp_str']
    log.info(f"Loaded {len(df)} raw review events.")

    df['timestamp'] = pd.to_datetime(df['timestamp_str']).astype('int64') // 10**9

    log.info("Aggregating duplicate ratings...")
    df_aggregated = df.groupby(['user_original', 'item_original']).agg({
        'rating': 'mean',
        'timestamp': 'max'
    }).reset_index()
    log.info(f"Aggregated data into {len(df_aggregated)} unique user-item ratings.")

    # Sample 10% of users to reduce dataset size
    unique_users = df_aggregated['user_original'].unique()
    
    np.random.seed(RANDOM_SEED)
    sampled_user_ids = np.random.choice(unique_users, size=int(len(unique_users) * 0.10), replace=False)
    
    df_sampled = df_aggregated[df_aggregated['user_original'].isin(sampled_user_ids)].copy()
    log.info(f"Sampled down to {len(df_sampled)} ratings from {len(sampled_user_ids)} users.")

    if df_sampled.empty:
        log.warning("DataFrame is empty after sampling. No output will be generated.")
        return

    log.info("Converting user and item string IDs to integer IDs...")

    final_unique_users = df_sampled['user_original'].unique()
    user_mapping_df = pd.DataFrame({
        'user': np.arange(len(final_unique_users)),
        'user_original': final_unique_users
    })
    
    final_unique_items = df_sampled['item_original'].unique()
    item_mapping_df = pd.DataFrame({
        'item': np.arange(len(final_unique_items)),
        'item_original': final_unique_items
    })

    os.makedirs(output_dir, exist_ok=True)
    user_mapping_df.to_csv(output_user_mapping_filepath, index=False)
    item_mapping_df.to_csv(output_item_mapping_filepath, index=False)
    log.info("Saved user and item ID mappings.")

    user_map_dict = pd.Series(user_mapping_df.user.values, index=user_mapping_df.user_original).to_dict()
    item_map_dict = pd.Series(item_mapping_df.item.values, index=item_mapping_df.item_original).to_dict()

    df_sampled['user'] = df_sampled['user_original'].map(user_map_dict)
    df_sampled['item'] = df_sampled['item_original'].map(item_map_dict)
    
    final_columns = ['user', 'item', 'rating', 'timestamp']
    df_final = df_sampled[final_columns].copy()
    
    df_final['user'] = df_final['user'].astype(np.int64)
    df_final['item'] = df_final['item'].astype(np.int64)
    df_final['rating'] = df_final['rating'].astype(float)
    df_final['timestamp'] = df_final['timestamp'].astype(np.int64)

    df_final.to_csv(output_ratings_filepath, index=False)
    
    log.info(f"Successfully processed Yelp data and saved to: {os.path.abspath(output_ratings_filepath)}")

def main():
    """
    Runs all preprocessing functions in sequence.
    """
    log.info("--- STARTING ALL DATASET PREPROCESSING ---")

    # Preprocess datasets for experiments
    preprocess_movielens()
    preprocess_lastfm()
    preprocess_bookcrossing()
    preprocess_retailrocket()
    preprocess_steam()
    
    # Preprocess Probe datasets for performance features
    preprocess_amazon_books()
    preprocess_online_retail()
    preprocess_yelp()

    log.info("--- ALL DATASET PREPROCESSING COMPLETE ---")




if __name__ == "__main__":
    main()