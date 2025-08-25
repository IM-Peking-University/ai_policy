import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import json
import os
import logging
import concurrent.futures
import time
from scipy import stats
import re


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_confidence_interval(proportion, n, confidence=0.95):
    """Wilson score interval for proportion."""
    if n == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2/n
    center = (proportion + z**2/(2*n))/denominator
    halfwidth = z * np.sqrt((proportion*(1-proportion) + z**2/(4*n))/n)/denominator
    return max(0.0, center - halfwidth), min(1.0, center + halfwidth)

def check_abstract_for_keywords(abstract, keyword_pattern):
    """Check if abstract contains keywords."""
    if pd.isna(abstract):
        return False

    return bool(keyword_pattern.search(str(abstract).lower()))

def process_chunk(chunk_df, keyword_pattern):
    """Process chunk for keyword detection."""
    chunk_df = chunk_df.copy()
    chunk_df['has_keyword'] = chunk_df['abstract'].apply(check_abstract_for_keywords, args=(keyword_pattern,))
    return chunk_df

def parallelize_keyword_check(df, keyword_pattern, num_processes):
    """Parallel keyword checking."""
    num_chunks = num_processes * 4
    chunks = np.array_split(df, num_chunks)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:

        future_to_chunk = {executor.submit(process_chunk, chunk, keyword_pattern): i for i, chunk in enumerate(chunks)}
        
        results = []
        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(chunks), desc="Checking for keywords"):
            results.append(future.result())
            
    return pd.concat(results)

def analyze_and_save_group(df, group_by_cols, output_filename):
    """Analyze grouped data and save results."""
    logger.info(f"Running analysis for: {output_filename}")

    # Ensure boolean columns are treated correctly for grouping
    for col in ['has_ai_policy', 'is_oa']:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Perform grouping and aggregation
    grouped = df.groupby(group_by_cols)['has_keyword']
    
    # Calculate count and mean (proportion)
    agg_results = grouped.agg(['count', 'mean']).reset_index()
    
    # Calculate confidence intervals
    ci_results = agg_results.apply(
        lambda row: compute_confidence_interval(row['mean'], row['count']),
        axis=1,
        result_type='expand'
    )
    agg_results[['ci_lower', 'ci_upper']] = ci_results
    
    agg_results.to_csv(output_filename, index=False, encoding='utf-8-sig')
    logger.info(f"Saved results to {output_filename}")

def main():
    start_time = time.time()
    
    # --- Setup ---
    cpu_count = os.cpu_count() or 8
    max_workers = int(cpu_count * 0.8)
    output_dir = 'keyword_res'
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load Keywords ---
    logger.info("Loading feature keywords...")
    with open('ai_feature_words.json', 'r') as f:
        keywords = json.load(f).keys()
    # Create a single, efficient regex pattern to find any of the keywords as whole words
    keyword_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b', re.IGNORECASE)
    logger.info(f"Loaded {len(keywords)} keywords into a regex pattern.")

    # --- Load and Preprocess Data ---
    cache_file = 'preprocessed_papers_with_keywords_filtered.joblib'
    if os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}...")
        df = joblib.load(cache_file)
    else:
        logger.info("Loading and preprocessing new data for keyword analysis...")
        input_file = 'ai_policy\data\paper_2021-2025.csv'
        df = pd.read_csv(input_file)
        logger.info(f"Initial dataframe shape: {df.shape}")
        
        # --- KEY MODIFICATION: Filter out rows with empty abstracts ---
        initial_rows = len(df)
        df.dropna(subset=['abstract'], inplace=True)
        df = df[df['abstract'].str.strip() != '']
        final_rows = len(df)
        logger.info(f"Removed {initial_rows - final_rows} rows with empty abstracts.")
        logger.info(f"Dataframe shape after cleaning: {df.shape}")
        # -----------------------------------------------------------------

        # Date processing
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True) # Also remove rows where date could not be parsed
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['half_year'] = df['date'].dt.month.apply(lambda m: 'H1' if m <= 6 else 'H2')
        

        df = parallelize_keyword_check(df, keyword_pattern, max_workers)
        

        joblib.dump(df, cache_file)
        logger.info(f"Cached preprocessed data with keywords to {cache_file}")


    df['country_list'] = df['country_list'].fillna('').str.split('|')
    df['domain_list'] = df['domain_list'].fillna('').str.split('|')
    

    df_country = df.explode('country_list').copy()
    df_domain = df.explode('domain_list').copy()

    df_country = df_country[df_country['country_list'] != '']
    df_domain = df_domain[df_domain['domain_list'] != '']


    scenarios = [
        (df, ['year', 'half_year'], 'by_half_year.csv'),
        (df, ['year', 'half_year', 'has_ai_policy'], 'by_half_year_policy.csv'),
        (df, ['year', 'half_year', 'is_oa'], 'by_half_year_oa.csv'),
        (df_country, ['year', 'half_year', 'country_list'], 'by_half_year_country.csv'),
        (df_domain, ['year', 'half_year', 'domain_list'], 'by_half_year_domain.csv'),
        (df_country, ['year', 'half_year', 'has_ai_policy', 'country_list'], 'by_half_year_policy_country.csv'),
        (df_domain, ['year', 'half_year', 'has_ai_policy', 'domain_list'], 'by_half_year_policy_domain.csv'),
        (df_country, ['year', 'half_year', 'is_oa', 'country_list'], 'by_half_year_oa_country.csv'),
        (df_domain, ['year', 'half_year', 'is_oa', 'domain_list'], 'by_half_year_oa_domain.csv'),
        (df, ['year', 'month'], 'by_month.csv')
    ]
    
    for data, group_cols, filename in scenarios:
        output_path = os.path.join(output_dir, filename)
        analyze_and_save_group(data, group_cols, output_path)

    logger.info(f"All keyword analyses complete. Total time: {(time.time() - start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    main()