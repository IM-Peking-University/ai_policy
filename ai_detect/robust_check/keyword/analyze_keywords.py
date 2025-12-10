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
from functools import partial
# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Confidence interval calculation function ---

def compute_confidence_interval(proportion, n, confidence=0.95):
    """
    Computes the Wilson score interval for a binomial proportion.
    """
    if n == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2/n
    center = (proportion + z**2/(2*n))/denominator
    halfwidth = z * np.sqrt((proportion*(1-proportion) + z**2/(4*n))/n)/denominator
    return max(0.0, center - halfwidth), min(1.0, center + halfwidth)

# --- Keyword checking and parallel processing function ---

def check_abstract_for_keywords(abstract, keyword_pattern):
    """
    Checks if an abstract contains any of the keywords using a compiled regex pattern.
    """
    if pd.isna(abstract):
        return False
    # Use re.search for efficiency, as it stops at the first match.
    return bool(keyword_pattern.search(str(abstract).lower()))

def process_chunk(chunk_df, keyword_pattern):
    """
    Processes a chunk of the DataFrame to check for keywords.
    """
    # Ensure keyword_pattern is accessible in subprocess
    # Since it's a re.Pattern object, Joblib/multiprocessing can serialize it
    chunk_df = chunk_df.copy()
    chunk_df['has_keyword'] = chunk_df['abstract'].apply(check_abstract_for_keywords, args=(keyword_pattern,))
    return chunk_df

def parallelize_keyword_check(df, keyword_pattern, num_processes):
    """
    Parallelizes the keyword checking across a DataFrame.
    """
    num_chunks = num_processes * 4
    chunks = np.array_split(df, num_chunks)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Use partial to pass keyword_pattern
        func = partial(process_chunk, keyword_pattern=keyword_pattern)
        
        results = list(tqdm(executor.map(func, chunks), total=len(chunks), desc="Checking for keywords"))
            
    return pd.concat(results)

# --- Analyze and save function ---

def analyze_and_save_group(df, group_by_cols, output_filename):
    """
    Analyzes a grouped DataFrame and saves the results to a CSV file.
    Output fields include: grouping fields, count, mean (proportion), ci_lower, ci_upper.
    """
    logger.info(f"Running analysis for: {output_filename}")

    # Ensure boolean columns are properly handled
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
    # Rename columns to match requirements
    agg_results.rename(columns={'mean': 'proportion'}, inplace=True)
    agg_results[['ci_lower', 'ci_upper']] = ci_results
    
    # Determine final output column order
    final_cols = group_by_cols + ['count', 'proportion', 'ci_lower', 'ci_upper']

    agg_results = agg_results[final_cols]
    agg_results.to_csv(output_filename, index=False, encoding='utf-8-sig')
    logger.info(f"Saved results to {output_filename}")


# --- Data preparation and merging function ---

def prepare_data_for_analysis(papers_csv, journal_info_csv, cache_file, max_workers, keyword_pattern):
    """
    Load paper and journal data, merge policy info, process dates, create features, and cache.
    """
    if os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}...")
        return joblib.load(cache_file)

    logger.info("Loading and processing new data...")
    
    # Load data
    papers_df = pd.read_csv(papers_csv, low_memory=False)
    journal_policy_df = pd.read_csv(journal_info_csv, low_memory=False)

    # Data cleaning and feature engineering
    
    # % of Citable OA binning
    journal_policy_df['% of Citable OA'] = pd.to_numeric(
        journal_policy_df['% of Citable OA'], errors='coerce'
    )
    journal_policy_df['oa_bin_50'] = np.where(
        journal_policy_df['% of Citable OA'] >= 50,
        'OA>=50%',
        'OA<50%'
    )
    
    # Merge policy information (exact match on journal name)
    papers_df = papers_df.merge(
        journal_policy_df,
        how='left',
        left_on='journal',
        right_on='journal_name',
        validate='m:1'
    )
    
    # Date processing and time feature creation
    papers_df['publication_date'] = pd.to_datetime(papers_df['publication_date'], errors='coerce')
    # Clean rows missing abstract, date, or journal info
    papers_df.dropna(subset=['publication_date', 'abstract', 'journal'], inplace=True)
    
    papers_df['year'] = papers_df['publication_date'].dt.year
    papers_df['month'] = papers_df['publication_date'].dt.month
    papers_df['half_year'] = papers_df['publication_date'].dt.month.apply(lambda m: 'H1' if m <= 6 else 'H2')
    
    # Keyword checking (most time-consuming step)
    papers_df = parallelize_keyword_check(papers_df, keyword_pattern, max_workers)
    
    # List field preparation (only country has many-to-many)
    # Country field is many-to-many (AU;GB;HK;...), domain and field are single-value
    papers_df['country_list'] = papers_df['country'].fillna('').str.split(';')
    
    # Keep only necessary columns
    cols_to_keep = [
        'abstract', 'has_keyword', 'year', 'month', 'half_year',
        'has_ai_policy', 'policy_category', 'is_oa', 'oa_bin_50',
        'country_list', 'domain', 'field'
    ]
    papers_df = papers_df[cols_to_keep]

    # Cache preprocessed results
    joblib.dump(papers_df, cache_file)
    logger.info(f"Cached preprocessed data to {cache_file}")
    
    return papers_df

# --- Main function ---

def main():
    start_time = time.time()
    
    # Configuration and file paths
    cpu_count = os.cpu_count() or 8
    max_workers = int(cpu_count * 0.8)
    output_dir = 'keyword_results_final'
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths (relative to ai_detect directory)
    PAPAERS_CSV = '../../get_paperInfo/output.csv'
    JOURNAL_INFO_CSV = '../../journal_info.csv'
    KEYWORD_JSON = '../keywords/ai_feature_words.json'
    CACHE_FILE = 'keywords/preprocessed_papers_keyword_merged.joblib'
    
    # Load keywords and compile regex
    logger.info("Loading feature keywords...")
    with open(KEYWORD_JSON, 'r') as f:
        keywords = json.load(f).keys()
    # Create efficient regex pattern
    keyword_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b', re.IGNORECASE)
    logger.info(f"Loaded {len(keywords)} keywords into a regex pattern.")

    # Data preparation and merging
    df = prepare_data_for_analysis(PAPAERS_CSV, JOURNAL_INFO_CSV, CACHE_FILE, max_workers, keyword_pattern)

    # Prepare exploded DataFrames
    # Split country (country_list -> country)
    df_country = df.explode('country_list').rename(columns={'country_list': 'country'})
    df_country.dropna(subset=['country'], inplace=True)
    df_country = df_country[df_country['country'] != '']
    
    # Analysis scenario definitions
    
    # Naming convention: [time_granularity]_[main_grouping]_[policy/other_grouping]
    scenarios = [
        # Base time groupings
        (df, ['year', 'month'], 'by_year_month.csv'),
        (df, ['year', 'half_year'], 'by_half_year.csv'),
        
        # Time + policy groupings (has_ai_policy, policy_category)
        (df, ['year', 'month', 'has_ai_policy'], 'by_year_month_policy_h.csv'),
        (df, ['year', 'half_year', 'has_ai_policy'], 'by_half_year_policy_h.csv'),
        (df, ['year', 'month', 'policy_category'], 'by_year_month_policy_c.csv'),
        (df, ['year', 'half_year', 'policy_category'], 'by_half_year_policy_c.csv'),

        # Time + region/domain/topic (no policy dimension)
        (df_country, ['year', 'month', 'country'], 'by_year_month_country.csv'),
        (df_country, ['year', 'half_year', 'country'], 'by_half_year_country.csv'),
        (df, ['year', 'month', 'domain'], 'by_year_month_domain.csv'),
        (df, ['year', 'half_year', 'domain'], 'by_half_year_domain.csv'),
        (df, ['year', 'month', 'field'], 'by_year_month_field.csv'),
        (df, ['year', 'half_year', 'field'], 'by_half_year_field.csv'),

        # Time + is_oa / oa_bin_50 groupings
        (df, ['year', 'month', 'is_oa'], 'by_year_month_is_oa.csv'),
        (df, ['year', 'half_year', 'is_oa'], 'by_half_year_is_oa.csv'),
        (df, ['year', 'month', 'oa_bin_50'], 'by_year_month_oa_bin.csv'),
        (df, ['year', 'half_year', 'oa_bin_50'], 'by_half_year_oa_bin.csv'),
        
        # Combined groupings (all with has_ai_policy)
        
        # Country + policy
        (df_country, ['year', 'month', 'country', 'has_ai_policy'], 'by_year_month_country_h.csv'),
        (df_country, ['year', 'half_year', 'country', 'has_ai_policy'], 'by_half_year_country_h.csv'),
        
        # Domain + policy
        (df, ['year', 'month', 'domain', 'has_ai_policy'], 'by_year_month_domain_h.csv'),
        (df, ['year', 'half_year', 'domain', 'has_ai_policy'], 'by_half_year_domain_h.csv'),
        
        # Field + policy
        (df, ['year', 'month', 'field', 'has_ai_policy'], 'by_year_month_field_h.csv'),
        (df, ['year', 'half_year', 'field', 'has_ai_policy'], 'by_half_year_field_h.csv'),
        
        # is_oa + policy
        (df, ['year', 'month', 'is_oa', 'has_ai_policy'], 'by_year_month_is_oa_h.csv'),
        (df, ['year', 'half_year', 'is_oa', 'has_ai_policy'], 'by_half_year_is_oa_h.csv'),
        
        # oa_bin_50 + policy
        (df, ['year', 'month', 'oa_bin_50', 'has_ai_policy'], 'by_year_month_oa_bin_h.csv'),
        (df, ['year', 'half_year', 'oa_bin_50', 'has_ai_policy'], 'by_half_year_oa_bin_h.csv'),
    ]

    # Run analysis
    for data, group_cols, filename in scenarios:
        output_path = os.path.join(output_dir, filename)
        analyze_and_save_group(data, group_cols, output_path)

    logger.info(f"All {len(scenarios)} keyword analyses complete. Total time: {(time.time() - start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    main()
