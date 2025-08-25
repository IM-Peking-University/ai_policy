import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import json
from datetime import datetime
from scipy.optimize import minimize
import nltk
from collections import Counter
import os
import logging
import concurrent.futures
import time
from scipy import stats
import gc

# NLTK data path
nltk.data.path.append('/home/2300016615_kd/AIPaper/new_country/nltk_data')

# Logger configuration
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

def document_log_probability(adjectives, dist, indices):
    word_counts = {word: adjectives[word] for word in adjectives if word in indices}
    if not word_counts:
        return np.log(1e-10)
    
    idx = [indices[word] for word in word_counts]
    counts = np.array(list(word_counts.values()))
    probs = dist[idx]
    return np.sum(counts * np.log(probs + 1e-10))

def compute_log_likelihood(alpha, adjectives_list, human_dist, ai_dist, token_indices):
    log_likelihood = 0.0
    for adjectives in adjectives_list:
        log_p_human = document_log_probability(adjectives, human_dist, token_indices)
        log_p_ai = document_log_probability(adjectives, ai_dist, token_indices)
        
        term1 = np.log(1 - alpha + 1e-10) + log_p_human
        term2 = np.log(alpha + 1e-10) + log_p_ai
        
        max_log = np.maximum(term1, term2)
        log_likelihood += max_log + np.log(np.exp(term1 - max_log) + np.exp(term2 - max_log))
        
    return -log_likelihood

def compute_fisher_information(alpha, adjectives_list, human_dist, ai_dist, token_indices):
    fisher_info = 0.0
    
    for adjectives in adjectives_list:
        log_p_human = document_log_probability(adjectives, human_dist, token_indices)
        log_p_ai = document_log_probability(adjectives, ai_dist, token_indices)
        
        if log_p_human > -500 and log_p_ai > -500:
            p_human = np.exp(log_p_human)
            p_ai = np.exp(log_p_ai)
            
            mixture_prob = (1 - alpha) * p_human + alpha * p_ai
            
            if mixture_prob > 1e-100:  
                # Fisher information: (p_ai - p_human)² / mixture_prob²
                diff_squared = (p_ai - p_human) ** 2
                fisher_contrib = diff_squared / (mixture_prob ** 2)
                fisher_info += fisher_contrib
        else:
            # For very small probabilities, use stable calculation in log space
            # Calculate log mixture probability
            log_1_minus_alpha = np.log(1 - alpha + 1e-15)
            log_alpha = np.log(alpha + 1e-15)
            
            log_term1 = log_1_minus_alpha + log_p_human
            log_term2 = log_alpha + log_p_ai
            
            # logsumexp trick
            max_log = np.maximum(log_term1, log_term2)
            log_mixture = max_log + np.log(np.exp(log_term1 - max_log) + np.exp(log_term2 - max_log))
            
            # If the difference between two probabilities is small, skip
            if abs(log_p_ai - log_p_human) < 1e-10:
                continue
                
            # Calculate log(|p_ai - p_human|²)
            # |p_ai - p_human|² = |exp(log_p_ai) - exp(log_p_human)|²
            if log_p_ai > log_p_human + 2:  # p_ai >> p_human
                log_diff_squared = 2 * log_p_ai
            elif log_p_human > log_p_ai + 2:  # p_human >> p_ai
                log_diff_squared = 2 * log_p_human
            else:
                # Both are similar, need to calculate carefully
                max_log_p = np.maximum(log_p_ai, log_p_human)
                min_log_p = np.minimum(log_p_ai, log_p_human)
                ratio = np.exp(min_log_p - max_log_p)
                if ratio > 1e-10:
                    log_diff_squared = 2 * max_log_p + 2 * np.log(abs(1 - ratio))
                else:
                    log_diff_squared = 2 * max_log_p
            
            # Fisher information: exp(log_diff_squared - 2*log_mixture)
            log_fisher_contrib = log_diff_squared - 2 * log_mixture
            
            if log_fisher_contrib > -100:  # Avoid very small values
                fisher_info += np.exp(log_fisher_contrib)
    
    return fisher_info

def estimate_alpha(adjectives, human_dist, ai_dist, token_indices):
    result = minimize(
        compute_log_likelihood,
        x0=np.array([0.5]),
        args=(adjectives, human_dist, ai_dist, token_indices),
        bounds=[(0, 1)],
        method='L-BFGS-B'
    )
    return result.x[0]

def compute_confidence_interval_fisher(alpha, fisher_info, confidence=0.95):
    """
    Standard MLE confidence interval based on Fisher information (improved version)

    Parameters:
    - alpha: MLE estimated α value
    - fisher_info: Observed Fisher information
    - confidence: Confidence level
    
    Returns:
    - (lower_bound, upper_bound): Confidence interval lower and upper bounds
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    if fisher_info > 1e-10:  # Avoid division by zero
        standard_error = 1.0 / np.sqrt(fisher_info)
        
        # For bounded parameters, use logit transformation for better confidence interval
        if 0.001 < alpha < 0.999:  # Avoid extreme values
            # Logit transformation: logit(p) = log(p/(1-p))
            logit_alpha = np.log(alpha / (1 - alpha))
            
            # Delta method to calculate standard error in logit space
            # Var(logit(α)) ≈ Var(α) / (α(1-α))²
            logit_se = standard_error / (alpha * (1 - alpha))
            
            # Build confidence interval in logit space
            logit_lower = logit_alpha - z * logit_se
            logit_upper = logit_alpha + z * logit_se
            
            # Back-transform to original space
            lower = np.exp(logit_lower) / (1 + np.exp(logit_lower))
            upper = np.exp(logit_upper) / (1 + np.exp(logit_upper))
        else:
            # For extreme values, use simple Wald interval but enforce boundaries
            margin = z * standard_error
            lower = max(0.0, alpha - margin)
            upper = min(1.0, alpha + margin)
    else:
        # When Fisher information is too small, fall back to wide confidence interval
        lower, upper = 0.0, 1.0
        logger.warning(f"Fisher information too small: {fisher_info}, using wide confidence interval")
    
    return lower, upper

def compute_confidence_interval_profile_likelihood(adjectives_list, human_dist, ai_dist, token_indices, alpha_mle, confidence=0.95):
    """
    Confidence interval based on Profile Likelihood (theoretically the most accurate method)
    
    Use likelihood ratio test to build confidence interval:
    2 * (L(α_mle) - L(α)) ~ χ²(1)
    
    Parameters:
    - adjectives_list: Data
    - human_dist, ai_dist, token_indices: Distribution parameters
    - alpha_mle: MLE estimated value
    - confidence: Confidence level
    
    Returns:
    - (lower_bound, upper_bound): Confidence interval
    """
    # Critical value
    critical_value = stats.chi2.ppf(confidence, df=1)
    
    # Calculate log likelihood at MLE
    log_likelihood_mle = -compute_log_likelihood(
        np.array([alpha_mle]), adjectives_list, human_dist, ai_dist, token_indices
    )
    
    def likelihood_ratio(alpha):
        """Calculate -2 times log likelihood ratio"""
        if alpha <= 0 or alpha >= 1:
            return np.inf
        
        log_likelihood_alpha = -compute_log_likelihood(
            np.array([alpha]), adjectives_list, human_dist, ai_dist, token_indices
        )
        return 2 * (log_likelihood_mle - log_likelihood_alpha)
    
    # Find confidence interval boundaries
    def find_boundary(direction, start_alpha):
        """Find boundary points that make the likelihood ratio equal to the critical value"""
        if direction == 'lower':
            search_range = (1e-6, start_alpha)
        else:
            search_range = (start_alpha, 1 - 1e-6)
        
        try:
            from scipy.optimize import brentq
            
            def objective(alpha):
                return likelihood_ratio(alpha) - critical_value
            
            # Check boundary conditions
            if objective(search_range[0]) * objective(search_range[1]) > 0:
                # No root, return boundary values
                if direction == 'lower':
                    return search_range[0]
                else:
                    return search_range[1]
            
            boundary = brentq(objective, search_range[0], search_range[1], xtol=1e-6)
            return boundary
            
        except Exception as e:
            logger.warning(f"Profile likelihood boundary search failed: {e}")
            # Fall back to Fisher information method
            return None
    
    # Find lower and upper bounds
    lower = find_boundary('lower', alpha_mle)
    upper = find_boundary('upper', alpha_mle)
    
    # If profile likelihood fails, fall back to Fisher information method
    if lower is None or upper is None:
        fisher_info = compute_fisher_information(alpha_mle, adjectives_list, human_dist, ai_dist, token_indices)
        return compute_confidence_interval_fisher(alpha_mle, fisher_info, confidence)
    
    return max(0.0, lower), min(1.0, upper)

def compute_confidence_interval(alpha, n, confidence=0.95, base_model_uncertainty=0.01, min_model_uncertainty=0.001):
    """Keep old method as backup"""
    # use z-score to calculate the confidence interval
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2/n
    center = (alpha + z**2/(2*n))/denominator
    stat_halfwidth = z * np.sqrt((alpha*(1-alpha) + z**2/(4*n))/n)/denominator
    
    adaptive_model_uncertainty = max(min_model_uncertainty, 
                                   base_model_uncertainty / np.sqrt(n/1000))
    
    total_halfwidth = np.sqrt(stat_halfwidth**2 + (z * adaptive_model_uncertainty)**2)

    lower = max(0.0, center - total_halfwidth)
    upper = min(1.0, center + total_halfwidth)
    
    return lower, upper

def analyze_group(group_data, group_columns, human_dist, ai_dist, token_indices, ci_method='fisher_logit'):
    """
    Analyze group data
    
    Parameters:
    - ci_method: Confidence interval method ('fisher_logit', 'profile_likelihood', 'fisher_wald')
    """
    if len(group_data) < 10:
        return None
        
    adjectives = group_data['adjectives'].tolist()
    alpha = estimate_alpha(adjectives, human_dist, ai_dist, token_indices)
    
    # Calculate Fisher information
    fisher_info = compute_fisher_information(alpha, adjectives, human_dist, ai_dist, token_indices)
    
    # Calculate confidence interval based on specified method
    if ci_method == 'profile_likelihood' and len(group_data) >= 50:
        # Profile likelihood method (use when sample size is large)
        try:
            ci_lower, ci_upper = compute_confidence_interval_profile_likelihood(
                adjectives, human_dist, ai_dist, token_indices, alpha
            )
            method_used = 'profile_likelihood'
        except Exception as e:
            logger.warning(f"Profile likelihood failed, falling back to Fisher: {e}")
            ci_lower, ci_upper = compute_confidence_interval_fisher(alpha, fisher_info)
            method_used = 'fisher_logit_fallback'
    elif ci_method == 'fisher_logit':
        # Improved Fisher information method (use logit transformation)
        ci_lower, ci_upper = compute_confidence_interval_fisher(alpha, fisher_info)
        method_used = 'fisher_logit'
    elif ci_method == 'fisher_wald':
        # Traditional Wald confidence interval (only for comparison)
        z = stats.norm.ppf(0.975)  # 95% CI
        if fisher_info > 1e-10:
            se = 1.0 / np.sqrt(fisher_info)
            margin = z * se
            ci_lower = max(0.0, alpha - margin)
            ci_upper = min(1.0, alpha + margin)
        else:
            ci_lower, ci_upper = 0.0, 1.0
        method_used = 'fisher_wald'
    else:
        # Default to improved Fisher method
        ci_lower, ci_upper = compute_confidence_interval_fisher(alpha, fisher_info)
        method_used = 'fisher_logit'
    
    result = {
        'count': len(group_data),
        'mean': alpha,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'fisher_info': fisher_info,
        'se': 1.0/np.sqrt(fisher_info) if fisher_info > 1e-10 else np.nan,
        'ci_method': method_used
    }
    
    # Add diagnostic information
    if fisher_info < 1e-6:
        result['warning'] = 'very_low_fisher_info'
    elif alpha < 0.01 or alpha > 0.99:
        result['warning'] = 'extreme_estimate'
    else:
        result['warning'] = 'none'
            
    return pd.DataFrame([result])

def run_analysis(df, group_by_cols, human_dist, ai_dist, token_indices, output_filename, num_processes, ci_method='fisher_logit'):
    logger.info(f"Running analysis for: {output_filename} using {ci_method} confidence intervals")
    
    # Ensure boolean columns are treated correctly
    for col in ['has_ai_policy', 'is_oa']:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    groups = df.groupby(group_by_cols)
    
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_group = {
            executor.submit(analyze_group, group_df, group_by_cols, human_dist, ai_dist, token_indices, ci_method): name
            for name, group_df in groups
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_group), total=len(groups), desc=f"Analyzing {output_filename}"):
            res = future.result()
            if res is not None:
                group_name = future_to_group[future]
                if isinstance(group_by_cols, list):
                    for i, col in enumerate(group_by_cols):
                        res[col] = group_name[i]
                else:
                     res[group_by_cols] = group_name
                all_results.append(res)

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        logger.info(f"Saved results to {output_filename}")
    else:
        logger.warning(f"No results generated for {output_filename}")


def main():
    start_time = time.time()
    
    # System setup
    cpu_count = os.cpu_count() or 8
    max_workers = int(cpu_count * 0.8)
    output_dir = 'mle_res'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load distributions
    logger.info("Loading word distributions...")
    dist_data = joblib.load('human_ai_distributions.joblib')
    human_distribution = dist_data['human_distribution']
    ai_distribution = dist_data['ai_distribution']
    token_indices = dist_data['token_indices']

    # Load and preprocess data
    cache_file = 'preprocessed_papers_with_policy.joblib'
    if os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}...")
        df = joblib.load(cache_file)
    else:
        logger.info("Loading and preprocessing new data...")
        input_file = 'ai_policy\data\paper_2021-2025.csv'
        df = pd.read_csv(input_file)
        
        # Date processing
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['half_year'] = df['date'].dt.month.apply(lambda m: 'H1' if m <= 6 else 'H2')
        
        # Adjective extraction
        df = parallelize_dataframe_processing(df, process_chunk, max_workers)
        
        # Cache preprocessed data
        joblib.dump(df, cache_file)
        logger.info(f"Cached preprocessed data to {cache_file}")

    # Explode lists
    df['country_list'] = df['country_list'].fillna('').str.split('|')
    df['domain_list'] = df['domain_list'].fillna('').str.split('|')
    
    df_country = df.explode('country_list')
    df_domain = df.explode('domain_list')

    # Analysis scenarios
    scenarios = [
        (df, ['year', 'half_year'], 'by_half_year.csv'),
        (df, ['year', 'half_year', 'has_ai_policy'], 'by_half_year_policy.csv'),
        (df, ['year', 'half_year', 'policy_category'], 'by_half_year_policy_category.csv'),
        (df, ['year', 'half_year', 'is_oa'], 'by_half_year_oa.csv'),
        (df_country, ['year', 'half_year', 'country_list'], 'by_half_year_country.csv'),
        (df_domain, ['year', 'half_year', 'domain_list'], 'by_half_year_domain.csv'),
        (df_country, ['year', 'half_year', 'has_ai_policy', 'country_list'], 'by_half_year_policy_country.csv'),
        (df_domain, ['year', 'half_year', 'has_ai_policy', 'domain_list'], 'by_half_year_policy_domain.csv'),
        # (df_country, ['year', 'half_year', 'is_oa', 'country_list'], 'by_half_year_oa_country.csv'),
        # (df_domain, ['year', 'half_year', 'is_oa', 'domain_list'], 'by_half_year_oa_domain.csv'),
        (df, ['year', 'month'], 'by_month.csv'),
        (df,['year','half_year','is_oa','has_ai_policy'],'by_half_year_oa_policy.csv')
    ]
    
    for data, group_cols, filename in scenarios:
        output_path = os.path.join(output_dir, filename)
        run_analysis(data, group_cols, human_distribution, ai_distribution, token_indices, output_path, max_workers, ci_method='fisher_logit')

    logger.info(f"All analyses complete. Total time: {(time.time() - start_time)/60:.2f} minutes.")

def compare_confidence_interval_methods(df_sample, human_dist, ai_dist, token_indices, output_file='ci_method_comparison.csv'):
    logger.info("Comparing confidence interval methods...")
    
    methods = ['fisher_wald', 'fisher_logit', 'profile_likelihood']
    results = []
    
    groups = df_sample.groupby(['year', 'half_year'])
    
    for (year, half_year), group_data in groups:
        if len(group_data) < 50:  # 只测试样本量较大的组
            continue
            
        adjectives = group_data['adjectives'].tolist()
        alpha_mle = estimate_alpha(adjectives, human_dist, ai_dist, token_indices)
        fisher_info = compute_fisher_information(alpha_mle, adjectives, human_dist, ai_dist, token_indices)
        
        group_result = {
            'year': year,
            'half_year': half_year,
            'sample_size': len(group_data),
            'alpha_mle': alpha_mle,
            'fisher_info': fisher_info
        }
        
        for method in methods:
            try:
                if method == 'fisher_wald':
                    z = stats.norm.ppf(0.975)
                    if fisher_info > 1e-10:
                        se = 1.0 / np.sqrt(fisher_info)
                        margin = z * se
                        ci_lower = max(0.0, alpha_mle - margin)
                        ci_upper = min(1.0, alpha_mle + margin)
                    else:
                        ci_lower, ci_upper = 0.0, 1.0
                        
                elif method == 'fisher_logit':
                    ci_lower, ci_upper = compute_confidence_interval_fisher(alpha_mle, fisher_info)
                    
                elif method == 'profile_likelihood':
                    ci_lower, ci_upper = compute_confidence_interval_profile_likelihood(
                        adjectives, human_dist, ai_dist, token_indices, alpha_mle
                    )
                
                group_result[f'{method}_lower'] = ci_lower
                group_result[f'{method}_upper'] = ci_upper  
                group_result[f'{method}_width'] = ci_upper - ci_lower
                group_result[f'{method}_success'] = True
                
            except Exception as e:
                logger.warning(f"Method {method} failed for group {year}-{half_year}: {e}")
                group_result[f'{method}_lower'] = np.nan
                group_result[f'{method}_upper'] = np.nan
                group_result[f'{method}_width'] = np.nan
                group_result[f'{method}_success'] = False
        
        results.append(group_result)
    
    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    logger.info("Confidence Interval Method Comparison Summary:")
    for method in methods:
        width_col = f'{method}_width'
        success_col = f'{method}_success'
        
        if width_col in comparison_df.columns:
            success_rate = comparison_df[success_col].mean() * 100
            mean_width = comparison_df[width_col].mean()
            median_width = comparison_df[width_col].median()
            
            logger.info(f"{method:20s}: Success rate = {success_rate:5.1f}%, "
                       f"Mean width = {mean_width:.4f}, Median width = {median_width:.4f}")
    
    logger.info(f"Detailed comparison results saved to {output_file}")
    return comparison_df

if __name__ == "__main__":
    main() 