import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import os
import logging
import concurrent.futures
import time
from scipy.optimize import minimize
from scipy import stats
from collections import Counter
import nltk
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_adjectives(document):
    if pd.isna(document) or document == '':
        return []
    words = nltk.word_tokenize(str(document))
    tagged_words = nltk.pos_tag(words)
    return [word for word, tag in tagged_words if tag.startswith('JJ')]

def process_chunk(chunk):
    chunk = chunk.copy()
    chunk['adjectives'] = chunk['abstract'].apply(lambda x: Counter(extract_adjectives(x)))
    return chunk

def parallelize_dataframe_processing(df, func, num_processes):
    num_chunks = num_processes * 4
    chunks = np.array_split(df, num_chunks)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(executor.map(func, chunks), total=len(chunks), desc="Extracting Adjectives"))
    return pd.concat(results)

def document_log_probability(adjectives_counts, dist, indices):
    word_counts = {word: adjectives_counts[word] for word in adjectives_counts if word in indices}
    if not word_counts:
        return np.log(1e-10)

    idx = [indices[word] for word in word_counts]
    counts = np.array(list(word_counts.values()))
    probs = dist[idx]
    return np.sum(counts * np.log(probs + 1e-10))

def compute_log_likelihood(alpha, adjectives_list, human_dist, ai_dist, token_indices):
    log_likelihood = 0.0
    alpha_safe = np.clip(alpha[0], 1e-15, 1.0 - 1e-15)
    log_1_minus_alpha = np.log(1.0 - alpha_safe)
    log_alpha = np.log(alpha_safe)
    
    for adjectives in adjectives_list:
        log_p_human = document_log_probability(adjectives, human_dist, token_indices)
        log_p_ai = document_log_probability(adjectives, ai_dist, token_indices)
        
        term1 = log_1_minus_alpha + log_p_human
        term2 = log_alpha + log_p_ai
        
        max_log = np.maximum(term1, term2)
        log_likelihood += max_log + np.log(np.exp(term1 - max_log) + np.exp(term2 - max_log))
        
    return -log_likelihood

def estimate_alpha(adjectives, human_dist, ai_dist, token_indices):
    result = minimize(
        compute_log_likelihood,
        x0=np.array([0.5]),
        args=(adjectives, human_dist, ai_dist, token_indices),
        bounds=[(0, 1)],
        method='L-BFGS-B'
    )
    return result.x[0]

def compute_confidence_interval(alpha, adjectives_list, human_dist, ai_dist, token_indices, confidence=0.95):
    fisher_info = 0.0
    alpha_safe = np.clip(alpha, 1e-15, 1.0 - 1e-15)
    
    for adjectives in adjectives_list:
        log_p_human = document_log_probability(adjectives, human_dist, token_indices)
        log_p_ai = document_log_probability(adjectives, ai_dist, token_indices)
        
        if abs(log_p_ai - log_p_human) < 1e-6:
            continue
            
        max_log_p = np.maximum(log_p_ai, log_p_human)
        min_log_p = np.minimum(log_p_ai, log_p_human)
        log_diff_squared = 2 * max_log_p + 2 * np.log(abs(1 - np.exp(min_log_p - max_log_p)))
        
        log_1_minus_alpha = np.log(1.0 - alpha_safe)
        log_alpha = np.log(alpha_safe)
        
        log_term1 = log_1_minus_alpha + log_p_human
        log_term2 = log_alpha + log_p_ai
        max_log_mix = np.maximum(log_term1, log_term2)
        log_mixture = max_log_mix + np.log(np.exp(log_term1 - max_log_mix) + np.exp(log_term2 - max_log_mix))
        
        log_fisher_contrib = log_diff_squared - 2 * log_mixture
        
        if log_fisher_contrib > -100:
            fisher_info += np.exp(log_fisher_contrib)
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    if fisher_info > 1e-10: 
        standard_error = 1.0 / np.sqrt(fisher_info)
        
        if 1e-3 < alpha < 1.0 - 1e-3: 
            logit_alpha = np.log(alpha / (1.0 - alpha))
            logit_se = standard_error / (alpha * (1.0 - alpha))
            
            logit_lower = logit_alpha - z * logit_se
            logit_upper = logit_alpha + z * logit_se
            lower = np.exp(logit_lower) / (1 + np.exp(logit_lower))
            upper = np.exp(logit_upper) / (1 + np.exp(logit_upper))
        else:
            margin = z * standard_error
            lower = max(0.0, alpha - margin)
            upper = min(1.0, alpha + margin)
    else:
        lower, upper = 0.0, 1.0
    
    return lower, upper

def analyze_group(group_data, group_columns, human_dist, ai_dist, token_indices):
    if len(group_data) < 10:
        return None
        
    adjectives = group_data['adjectives'].tolist()
    
    alpha = estimate_alpha(adjectives, human_dist, ai_dist, token_indices)
    
    ci_lower, ci_upper = compute_confidence_interval(alpha, adjectives, human_dist, ai_dist, token_indices)
        
    result = {
        'count': len(group_data),
        'alpha': alpha,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }
    
    return pd.DataFrame([result])

def run_analysis(df, group_by_cols, human_dist, ai_dist, token_indices, output_filename, num_processes):
    logger.info(f"Running analysis for: {output_filename}")

    groups = df.groupby(group_by_cols)
    all_results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_group = {
            executor.submit(analyze_group, group_df, group_by_cols, human_dist, ai_dist, token_indices): name
            for name, group_df in groups
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_group), total=len(groups), desc=f"Analyzing {output_filename}"):
            res = future.result()
            if res is not None:
                group_name = future_to_group[future]
                
                if not isinstance(group_by_cols, list): 
                    res[group_by_cols] = group_name
                else: 
                    if not isinstance(group_name, (list, tuple)):
                        group_name = [group_name]
                        
                    for i, col in enumerate(group_by_cols):
                        res[col] = group_name[i]
                all_results.append(res)

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        final_cols = group_by_cols + ['count', 'alpha', 'ci_lower', 'ci_upper']
        results_df = results_df[final_cols]
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        logger.info(f"Saved results to {output_filename}")
    else:
        logger.warning(f"No results generated for {output_filename}")

def prepare_data_for_analysis(papers_csv, journal_info_csv, cache_file, max_workers):
    if os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}...")
        return joblib.load(cache_file)

    logger.info("Loading and processing new data...")
    
    papers_df = pd.read_csv(papers_csv, low_memory=False)
    journal_policy_df = pd.read_csv(journal_info_csv, low_memory=False)

    journal_policy_df['% of Citable OA'] = (
        journal_policy_df['% of Citable OA']
        .astype(str)
        .str.replace('%', '')
        .str.replace(' ', '')
        .apply(pd.to_numeric, errors='coerce')
    )
    
    oa_values = journal_policy_df['% of Citable OA']
    logger.info(f"OA distribution diagnosis:")
    logger.info(f"  Total records: {len(oa_values)}")
    logger.info(f"  Non-null count: {oa_values.notna().sum()}")
    logger.info(f"  Null count: {oa_values.isna().sum()}")
    if oa_values.notna().sum() > 0:
        logger.info(f"  OA value range: {oa_values.min():.2f}% - {oa_values.max():.2f}%")
        logger.info(f"  Count >=50%: {(oa_values >= 50).sum()}")
        logger.info(f"  Count <50%: {(oa_values < 50).sum()}")
        
    journal_policy_df['oa_bin_50'] = np.where(
        journal_policy_df['% of Citable OA'] >= 50,
        'OA>=50%',
        np.where(
            journal_policy_df['% of Citable OA'] < 50,
            'OA<50%',
            'OA_unknown'
        )
    )

    papers_df = papers_df.merge(
        journal_policy_df,
        how='left',
        left_on='journal',
        right_on='journal_name',
        validate='m:1'
    )
    
    logger.info(f"OA distribution after merge:")
    oa_dist_after_merge = papers_df['oa_bin_50'].value_counts()
    logger.info(f"{oa_dist_after_merge}")
    
    papers_df['publication_date'] = pd.to_datetime(papers_df['publication_date'], errors='coerce')
    papers_df.dropna(subset=['publication_date', 'abstract', 'journal'], inplace=True)
    
    papers_df['year'] = papers_df['publication_date'].dt.year
    papers_df['month'] = papers_df['publication_date'].dt.month
    papers_df['half_year'] = papers_df['publication_date'].dt.month.apply(lambda m: 'H1' if m <= 6 else 'H2')
    
    papers_df = parallelize_dataframe_processing(papers_df, process_chunk, max_workers)
    
    papers_df['country_list'] = papers_df['country'].fillna('').str.split(';')
    
    cols_to_keep = [
        'abstract', 'adjectives', 'year', 'month', 'half_year',
        'has_ai_policy', 'policy_category', 'is_oa', 'oa_bin_50',
        'country_list', 'domain', 'field','journal'
    ]
    papers_df = papers_df[cols_to_keep]

    joblib.dump(papers_df, cache_file)
    logger.info(f"Cached preprocessed data to {cache_file}")
    
    return papers_df

def main():
    start_time = time.time()
    
    cpu_count = os.cpu_count() or 8
    max_workers = int(cpu_count * 0.8)
    output_dir = './mle_results_final'
    os.makedirs(output_dir, exist_ok=True)
    
    PAPAERS_CSV = './get_paperInfo/output.csv' 
    JOURNAL_INFO_CSV = './journal_info.csv'
    DIST_DATA_JOBILB = './train_model/human_ai_distributions.joblib'
    CACHE_FILE = './preprocessed_papers_merged.joblib'

    logger.info("Loading word distributions...")
    dist_data = joblib.load(DIST_DATA_JOBILB)
    human_distribution = dist_data['human_distribution']
    ai_distribution = dist_data['ai_distribution']
    token_indices = dist_data['token_indices']
    FORCE_REPROCESS = True 
    
    if os.path.exists(CACHE_FILE) and not FORCE_REPROCESS:
        logger.info(f"Loading cached data directly from {CACHE_FILE}...")
        df = joblib.load(CACHE_FILE)
        logger.info(f"Loaded {len(df)} records from cache")
    else:
        if FORCE_REPROCESS and os.path.exists(CACHE_FILE):
            logger.info("Force reprocess: deleting old cache file...")
            os.remove(CACHE_FILE) 
        logger.info("Running data preparation (cache not found or forced reprocess)...")
        df = prepare_data_for_analysis(PAPAERS_CSV, JOURNAL_INFO_CSV, CACHE_FILE, max_workers)

    df_country = df.explode('country_list').rename(columns={'country_list': 'country'})
    df_country.dropna(subset=['country'], inplace=True)
    df_country = df_country[df_country['country'] != '']
    
    scenarios = [
        # (df, ['year', 'month'], 'by_year_month.csv'),
        # (df, ['year', 'half_year'], 'by_half_year.csv'),
        
        # (df, ['year', 'month', 'has_ai_policy'], 'by_year_month_policy_h.csv'),
        # (df, ['year', 'half_year', 'has_ai_policy'], 'by_half_year_policy_h.csv'),
        # (df, ['year', 'month', 'policy_category'], 'by_year_month_policy_c.csv'),
        # (df, ['year', 'half_year', 'policy_category'], 'by_half_year_policy_c.csv'),

        # (df_country, ['year', 'month', 'country'], 'by_year_month_country.csv'),
        # (df_country, ['year', 'half_year', 'country'], 'by_half_year_country.csv'),
        # (df, ['year', 'month', 'domain'], 'by_year_month_domain.csv'),
        # (df, ['year', 'half_year', 'domain'], 'by_half_year_domain.csv'),
        # (df, ['year', 'month', 'field'], 'by_year_month_field.csv'),
        # (df, ['year', 'half_year', 'field'], 'by_half_year_field.csv'),

        # (df, ['year', 'month', 'is_oa'], 'by_year_month_is_oa.csv'),
        # (df, ['year', 'half_year', 'is_oa'], 'by_half_year_is_oa.csv'),
        # (df, ['year', 'month', 'oa_bin_50'], 'by_year_month_oa_bin.csv'),
        # (df, ['year', 'half_year', 'oa_bin_50'], 'by_half_year_oa_bin.csv'),
        
        # (df_country, ['year', 'month', 'country', 'has_ai_policy'], 'by_year_month_country_h.csv'),
        # (df_country, ['year', 'half_year', 'country', 'has_ai_policy'], 'by_half_year_country_h.csv'),
        
        # (df, ['year', 'month', 'domain', 'has_ai_policy'], 'by_year_month_domain_h.csv'),
        # (df, ['year', 'half_year', 'domain', 'has_ai_policy'], 'by_half_year_domain_h.csv'),
        
        # (df, ['year', 'month', 'field', 'has_ai_policy'], 'by_year_month_field_h.csv'),
        # (df, ['year', 'half_year', 'field', 'has_ai_policy'], 'by_half_year_field_h.csv'),
        
        # (df, ['year', 'month', 'is_oa', 'has_ai_policy'], 'by_year_month_is_oa_h.csv'),
        # (df, ['year', 'half_year', 'is_oa', 'has_ai_policy'], 'by_half_year_is_oa_h.csv'),
        
        # (df, ['year', 'month', 'oa_bin_50', 'has_ai_policy'], 'by_year_month_oa_bin_h.csv'),
        (df, ['year', 'half_year', 'journal'], 'by_half_year_journal.csv'),
        (df, ['year', 'month', 'journal'], 'by_month_journal.csv'),
    ]

    for data, group_cols, filename in scenarios:
        output_path = os.path.join(output_dir, filename)
        run_analysis(data, group_cols, human_distribution, ai_distribution, token_indices, output_path, max_workers)

    logger.info(f"All {len(scenarios)} analyses complete. Total time: {(time.time() - start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    main()