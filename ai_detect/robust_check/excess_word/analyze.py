import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import json
import os
import logging
import concurrent.futures
import time
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from scipy import stats
from collections import defaultdict
import re

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define external file paths
PAPAERS_CSV = './output.csv'
JOURNAL_INFO_CSV = './journal_info.csv'
ANNOTAION_CSV = './my_excess_words.csv'
RESULTS_FOLDER = Path("./llm_excess_vocab_results")
CACHE_FILE = './llm_excess_vocab/preprocessed_papers_llm_merged.joblib'
RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)


# --- Core utility functions (cleanup, probability, extrapolation) ---

# Abstract cleanup function
def cleanup_abstracts_inplace(df):
    to_replace = {
        "&ldquo;": {"&ldquo;": '"', "&rdquo;": '"'},
        "&lsquo;": {"&lsquo;": "'", "&rsquo;": "'"},
        "&nbsp;": {"&nbsp;": " "},
        "&shy;": {"&shy;": ""},
        "&mdash;": {"&mdash;": "---"},
        "&ndash;": {"&ndash;": "--"},
        "u2002": {"\\u2002": " "},
        "<p>": {"<p>": "", "</p>": ""},
        "<em>": {"<em>": "", "</em>": ""},
        "This article": {"^This article has been.*": ""},
        "This manuscript": {"^This manuscript has been.*": ""},
        "The above article": {
            "^The above article.*": "",
            ".*The above article, published online.*": "",
        },
        "http": {"^http.*": ""},
        "For complete details": {
            "\s*For complete details on the use and execution of this protocol.*": "",
        },
        "For further information": {
            "For further information please consult linked data\.*": "",
        },
        "Communicated by": {"\s*\(?Communicated by.{0,100}$": ""},
        "Graphical abstract": {"\.\s*Graphical abstract.*": "."},
        "GRAPHICAL ABSTRACT": {"\.\s*GRAPHICAL ABSTRACT.*": "."},
        "VIDEO ABSTRACT": {"\.\s*VIDEO ABSTRACT.*": "."},
        "Video Abstract": {"\s*Video Abstract Available\.*": ""},
        "MINI ABSTRACT": {"\.\s*MINI ABSTRACT.*": "."},
        "ABSTRACT": {
            "^ABSTRACT[:.]?\s*": "",
            "^Abstract ABSTRACT[:.]?\s*": "",
            "^.{0,200} ABSTRACT: ": "",
        },
        "Abstract": {"^Abstract:?\s*": ""},
        "CONSPECTUS": {"CONSPECTUS: ": ""},
        "THIS ARTICLE": {
            "\s*THIS ARTICLE HAD BEEN MADE AVAILABLE FREE OF CHARGE.*": ""
        },
        "Copyright ©": {"\s*Copyright ©.*": ""},
        " © ": {
            "\.\s*[^.]*[0-9]\. © [12].*": ".",
            "\. [a-zA-Z]+\.$": ".",
            "\. [ab-zA-Z]+\.$": ".",
            "\. [abc-zA-Z]+\.$": ".",
            "\. [abcd-zA-Z]+\.$": ".",
            "\. Pediatr Pulmonol.$": ".",
            "\. Lasers Surg.$": ".",
            "\s+© .*": "",
        },
        ".© ": {
            "\.\s*[^.]*[0-9]\.© [12].*": ".",
            "\. [a-zA-Z]+\.$": ".",
            "\. [ab-zA-Z]+\.$": ".",
            "\. [abc-zA-Z]+\.$": ".",
            "\. [abcd-zA-Z]+\.$": ".",
            "\. Pediatr Pulmonol.$": ".",
            "\. Lasers Surg.$": ".",
            "\s+\.© .*": "",
        },
        " ©20": {
            "\.\s*[^.]*[0-9]\. ©20.{0,20}$": ".",
            "\s+©20.{0,20}$": "",
            "\.\s*[^.]*[0-9]\. ©20[0-2][0-9] AACRSee.*": ".",
            "\s+©20[0-2][0-9] AACRSee.*": "",
        },
        "Wiley Periodicals Inc": {
            "\.\s*[^.]*[0-9]\.\s*Published [12][890][0-9][0-9] Wiley Periodicals Inc.*": ".",
            "\. [a-zA-Z]+\.$": ".",
            "\. [ab-zA-Z]+\.$": ".",
            "\. [abc-zA-Z]+\.$": ".",
            "\. [abcd-zA-Z]+\.$": ".",
            "\s*Published [12][890][0-9][0-9] Wiley Periodicals Inc.*": ".",
        },
        "doi": {
            "\s*doi:\s*10\.[0-9a-zA-Z\.\/\-]*\s*$": "",
            "\s*doi:\s*10\.[0-9a-zA-Z\.\/\-]*\s*\(.*\)\.?\s*$": "",
            "\s*http://dx\.doi\.org/10\.[0-9a-zA-Z\.\/\-]*\s*$": "",
        },
        "DOI": {"\s*DOI: http://dx.doi.org/[0-9a-zA-Z\.\/\-]*\s*$": ""},
        "PMID": {".*PubMed PMID: [0-9]*\.\s*": ""},
        "Epub": {"\sEpub.{0,10}[12][890][0-9][0-9]\.?\s*$": ""},
        "Level of Evidence": {"\s*Level of Evidence:?\s*[0-9IV].*": ""},
        "LEVEL OF EVIDENCE": {"\s*LEVEL OF EVIDENCE:?\s*[0-9IV].*": ""},
        "Technical Efficacy": {"\s*[0-9] Technical Efficacy: Stage [0-9].*": ""},
        "Geriatr": {
            "\s*Geriatr Gerontol Int [\s0-9,;:\-\.\(\)]*$": "",
            "\s*J Am Geriatr Soc [\s0-9,;:\-\.\(\)]*$": "",
        },
        "Genet Med": {"\s*Genet Med [\s0-9,;:\-\.\(\)]*$": ""},
        "Ann Neurol": {
            "\s*Ann Neurol [\s0-9,;:\-\.\(\)]*$": "",
            "\s*Ann Neurol [\s01-9,;:\-\.\(\)]*$": "",
        },
        "ANN NEUROL": {"\s*ANN NEUROL [\s0-9,;:\-\.\(\)]*$": ""},
        "J Drugs Dermatol": {"\s*J Drugs Dermatol\. [\s0-9,;:\-\.\(\)]*$": ""},
        "Infect Control Hosp": {
            "\s*Infect Control Hosp Epidemiol [\s0-9,;:\-\.\(\)]*$": ""
        },
        "Magn. Reson.": {
            "\.\s*[0-9]\s*J\. Magn\. Reson\. Imaging [\s0-9,;:\-\.\(\)]*$": ""
        },
        "MAGN. RESON.": {
            "\.\s*[0-9]\s*J\. MAGN\. RESON\. IMAGING [\s0-9,;:\-\.\(\)]*$": ""
        },
        "Magnetic Resonance": {
            "\s*Magnetic Resonance in Medicine published by Wiley Periodicals\.*": "",
        },
        "(Pediatr Dent": {"\s*\(Pediatr Dent 20.*": ""},
        "Environ Toxicol Chem": {"\s*Environ Toxicol Chem [\s0-9,;:\-\.\(\)]*$": ""},
        "Environ Health Perspect": {
            "\s*Environ Health Perspect [\s0-9,;:\-\.\(\)]*$": ""
        },
        "Antioxid. Redox Signal.": {
            "\s*Antioxid\. Redox Signal\. [\s0-9,;:\-\.\(\)]*$": ""
        },
        "J Orthop Sports Phys Ther": {
            "\s*J Orthop Sports Phys Ther\.? [\sA0-9,;:\-\.\(\)]*$": ""
        },
        "J Strength Cond Res": {
            ".*J Strength Cond Res.{0,20}20[012][0-9]-([A-Z])": "\\1"
        },
        "Turk J Pediatr": {".*Turk J Pediatr [\s0-9,;:\-\.\(\)]*([A-Z])": "\\1"},
        "Laryngoscope": {
            "\s*[1-9][a-zA-Z]?\. Laryngoscope[^\.]*\.\s*$": "",
            "[\.][^\.]*Laryngoscope[^\.]*\.\s*$": ".",
            "\s*N/A\.$": "",
        },
        "Indian J Crit Care Med": {
            "\s*[^\.]*\.[^\.]*[\.?!] Indian J Crit Care Med [\s0-9,;:\-\.\(\)]*$": "",
        },
        "Int J Clin Pediatr Dent": {
            "\s*[^\.]*\.[^\.]*[\.?!] Int J Clin Pediatr Dent [\s0-9,;:\-\.\(\)]*$": "",
        },
        "J Clin Sleep Med": {
            "\s*[^\.]*\.[^\.]*[\.?!] J Clin Sleep Med\. [\s0-9,;:\-\.\(\)]*$": "",
        },
        "Hepatology": {
            "\(Hepatology [\s0-9,;:\-\.\(\)]*\)\.?\s*$": "",
            "\(Hepatology Communications [\s0-9,;:\-\.\(\)]*\)\.?\s*$": "",
        },
        "AASLD": {"[^\.]*AASLD\.\s*$": ""},
        "Database Record": {
            "\s*\(PsycINFO Database Record.*": "",
            "\s*\(PsycInfo Database Record.*": "",
        },
        "advance online publication": {
            "[^\.]* advance online publication,?\s*[0-9][0-9] [A-Za-z]* [0-9]{4}[\.;,]?\s*$": ""
        },
        "This article is protected": {"\sThis article is protected by copyright.*": ""},
        "This article is part": {
            "\s*This article is part of the themed issue.*": "",
            "\s*This article is part of a themed issue.*": "",
            "\s*This article is part of a themed section.*": "",
            "\s*This article is part of a Special Issue entitled.*": "",
        },
        "Elsevier Ltd": {"\s*20[012][0-9] Elsevier Ltd.*": ""},
        "How to cite this article:": {"\s*How to cite this article:.*": ""},
        "Cite this article:": {"\s*Cite this article:.*": ""},
        "Citation": {"\s+Citation: .*": ""},
        "ClinicalTrials.gov:": {"\s*\(?ClinicalTrials.gov: .{0,100}$": ""},
        ".].": {"\[[^\[]*[0-9]\.\]\.?$": ""},
        "https://youtu.be": {
            "\.\s*[^.]*: https://youtu.be.{0,100}$": ".",
            "\. https://youtu.be.{0,50}.$": ".",
        },
        "The virtual slide(s) for this article": {
            "\s*The virtual slide\(s\) for this article.*": ""
        },
        "IMPACT STATEMENT": {"\.\s*IMPACT STATEMENT[A-Z].*": "."},
        "Impact statement": {"\.\s*Impact Statement[A-Z].*": "."},
        ("RESULTS", "CONCLUSIONS"): {
            "\.\s*PURPOSE[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*BACKGROUND[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*INTRODUCTION[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*OBJECTIVE[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*MATERIALS AND METHODS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*MATERIALS & METHODS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*METHODS?[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*METHODOLOGY[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*DESIGN[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*STUDY DESIGN[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*KEY RESULTS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*RESULTS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*CONCLUSIONS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*CONCLUSIONS AND INFERENCES[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*CONCLUSIONS & INFERENCES[.: ][.:]?\s*([A-Z])": ". \\1",
        },
        ("Results", "Conclusions"): {
            "\.\s*Purpose[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Background[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Introduction[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Objective[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Materials and methods[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Materials and Methods[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Materials & Methods[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Methods?[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Methodology[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Design[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Study Design[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Key Results[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Results[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions and inferences[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions and Inferences[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions & inferences[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions & Inferences[.: ][.:]?\s*([A-Z])": ". \\1",
        },
        "Expert commentary": {
            "\.\s*Expert commentary:\s*": ". ",
            "\.\s*Areas covered:\s*": ". ",
        },
        "Details of funding": {"\s*Details of funding are provided.*": ""},
        "This journal requires": {"\s*This journal requires.*": ""},
        "Proprietary or commercial disclosure": {
            "\s*Proprietary or commercial disclosure.*": ""
        },
        "See acknowledgments": {".\s*See acknowledgments.\s*$": "."},
        "This article is one of ten": {"\s*This article is one of ten reviews.*": ""},
        "In an effort to expedite the publication of articles": {
            "^In an effort to expedite the publication of articles.*": ""
        },
        "For complete coverage": {
            "\s*For complete coverage of all related areas of Endocrinology.*": ""
        },
        "Abbreviations": {
            "\.\s*Abbreviations:.*": ".",
            "\.\s*Abbreviations [Uu]sed:.*": ".",
        },
        "ABBREVIATIONS": {
            "\.\s*ABBREVIATIONS:.*": ".",
            "\.\s*ABBREVIATIONS USED:.*": ".",
        },
        "Registration number": {"\s*Registration number of the clinical trial:.*": ""},
        " de ": {
            "\.\s*:?\s*[Aa]nalisar .*": ".",
            "\.\s*:?\s*[Dd]escrever .*": ".",
            "\.\s*:?\s*[Ii]mplementar .*": ".",
            "\.\s*:?\s*[Cc]ompreender .*": ".",
            "\.\s*:?\s*[Aa]valiar .*": ".",
            "\.\s*:?\s*[Ee]stimar .*": ".",
            "\.\s*:?\s*[Dd]eterminar .*": ".",
            "\.\s*:?\s*[Rr]ealizar .*": ".",
            "\.\s*:?\s*[Cc]aracterizar .*": ".",
            "\.\s*:?\s*[Ii]dentificar .*": ".",
            "\.\s*:?\s*[Dd]iscutir .*": ".",
            "\.\s*:?\s*[Cc]onhecer .*": ".",
            "\.\s*:?\s*[Cc]onocer .*": ".",
            "\.\s*:?\s*Resumo .*": ".",
        },
    }
    
    # Actual cleanup logic
    for search_string, replacements in tqdm(to_replace.items(), desc="Applying Clean-up Rules"):
        # Simplified matching logic
        if isinstance(search_string, str):
            ind = df['AbstractText'].str.contains(search_string, na=False, regex=False)
        else:
            ind = np.ones(len(df), dtype=bool)
            for s in search_string:
                ind &= df['AbstractText'].str.contains(s, na=False, regex=False)
                
        for replace_pattern, replacement in replacements.items():
            df.loc[ind, "AbstractText"] = df[ind]['AbstractText'].str.replace(
                replace_pattern, replacement, regex=True
            )
            
    # Final cleanup
    df['AbstractText'] = df['AbstractText'].str.strip()


def group_prob(df_group, X_matrix, word_list, word2idx):
    """Proportion of papers with at least one word from the group (Laplace smoothing (k+1)/(n+1))"""
    idxs = [word2idx[w] for w in word_list if w in word2idx]
    
    n = len(df_group)
    if n == 0 or len(idxs) == 0:
        return np.nan, n
        
    # Get submatrix (use original DataFrame index to select rows in X_matrix)
    sub = X_matrix[df_group.index.values, :][:, idxs]
    
    # Number of papers with at least one word
    k = int((sub.sum(axis=1) > 0).A1.sum()) 
    
    return (k + 1) / (n + 1), n


# LLM Delta calculation function (based on 2021-2022 baseline)
def calculate_llm_delta(df_base, X_matrix, common_words, rare_words, word2idx):
    """
    Calculate LLM Delta_avg trend for given time series data (df_base).
    df_base must contain 2021-2022 data for establishing baseline.
    
    Returns a DataFrame containing Year, Month, Delta_avg, Delta_common, Delta_rare.
    """
    
    # Ensure baseline year exists
    if df_base['Year'].isin([2021, 2022]).sum() < 10:
        logger.warning("Base data lacks sufficient 2021/2022 observations for projection.")
        return pd.DataFrame()
        
    results = []
    
    # P_base should be cached by (year, month)
    P_base = {}  # Key is (year, month)
    
    # Calculate baseline probability P_g(2021, m) and P_g(2022, m)
    for y in [2021, 2022]:
        for m in range(1, 13):
            mask = (df_base['Year'] == y) & (df_base['Month'] == m)
            df_ym = df_base[mask]
            
            # Cache using (y, m) as key
            P_base[(y, m)] = { 
                'common': group_prob(df_ym, X_matrix, common_words, word2idx),
                'rare': group_prob(df_ym, X_matrix, rare_words, word2idx)
            }

    # Iterate target years (2023-2025H1)
    target_months = []
    for y in range(2023, 2026):
        for m in range(1, 13):
            if y == 2025 and m > 6: break
            target_months.append((y, m))

    for y, m in target_months:
        
        # Read 2022 and 2021 probabilities separately from P_base
        # Use .get() to return (np.nan, 0) if month data is missing
        P22_c, N22_c = P_base.get((2022, m), {'common': (np.nan, 0)})['common']
        P21_c, N21_c = P_base.get((2021, m), {'common': (np.nan, 0)})['common']
        
        P22_r, N22_r = P_base.get((2022, m), {'rare': (np.nan, 0)})['rare']
        P21_r, N21_r = P_base.get((2021, m), {'rare': (np.nan, 0)})['rare']
        
        # Calculate baseline prediction Q_g(y,m)
        def projected_group_prob(P22, P21, year):
            if np.isnan(P22) or np.isnan(P21): return np.nan
            step = max(P22 - P21, 0.0)
            return P22 + (year - 2022) * step

        Q_c = projected_group_prob(P22_c, P21_c, y)
        Q_r = projected_group_prob(P22_r, P21_r, y)

        # Calculate observed frequency P_g(y,m)
        mask_obs = (df_base['Year'] == y) & (df_base['Month'] == m)
        df_obs = df_base[mask_obs]
        
        # If df_obs is empty, P_c/P_r will be np.nan
        P_c, N_c = group_prob(df_obs, X_matrix, common_words, word2idx)
        P_r, N_r = group_prob(df_obs, X_matrix, rare_words, word2idx)
        
        N_docs = len(df_obs)
        
        # Calculate Delta and Delta_avg
        # RuntimeWarning: Mean of empty slice is triggered when both d_c and d_r are NaN
        d_c = P_c - Q_c if not (np.isnan(P_c) or np.isnan(Q_c)) else np.nan
        d_r = P_r - Q_r if not (np.isnan(P_r) or np.isnan(Q_r)) else np.nan
        
        d_avg = np.nanmean([d_c, d_r]) 
        
        results.append({
            "Year": y, "Month": m, "N_docs": N_docs,
            "Delta_avg": d_avg, "Delta_common": d_c, "Delta_rare": d_r,
        })
        
    return pd.DataFrame(results)


# --- Data preparation and vectorization (adapted for parallel processing) ---

def prepare_data_for_analysis(papers_csv, journal_info_csv, cache_file, max_workers, ann_path):
    """
    Load data, merge, clean, vectorize, and build common/rare vocabulary groups.
    Returns: df (with policy features), X_matrix (vectorized matrix), word2idx, common/rare_words
    """
    
    # Load data
    papers_df = pd.read_csv(papers_csv, low_memory=False)
    journal_policy_df = pd.read_csv(journal_info_csv, low_memory=False)

    papers_df = papers_df.rename(columns={"abstract": "AbstractText"})
    
    # Ensure date column exists and convert
    papers_df['publication_date'] = pd.to_datetime(papers_df['publication_date'], errors='coerce')
    papers_df.dropna(subset=['publication_date', 'AbstractText', 'journal'], inplace=True)
    
    papers_df['Year'] = papers_df['publication_date'].dt.year
    papers_df['Month'] = papers_df['publication_date'].dt.month
    
    # Original LLM paper filtering and cleaning
    papers_df = papers_df[papers_df['Year'].between(2021, 2025)].copy()
    papers_df['abstract_len'] = papers_df['AbstractText'].str.len()
    # Ensure abstract_len is numeric for comparison
    papers_df.dropna(subset=['abstract_len'], inplace=True)
    papers_df['abstract_len'] = papers_df['abstract_len'].astype(int) 
    papers_df = papers_df[(papers_df['abstract_len'] >= 250) & (papers_df['abstract_len'] <= 4000)].copy()
    
    logger.info(f"Initial paper count after filtering (2021-2025, length): {len(papers_df)}")
    
    # Abstract cleaning and feature merging
    
    # 1. Clean abstracts
    cleanup_abstracts_inplace(papers_df)
    
    # 2. Merge journal policy features
    
    
    # Ensure % of Citable OA is numeric, set to NaN if conversion fails
    journal_policy_df['% of Citable OA'] = pd.to_numeric(
        journal_policy_df['% of Citable OA'], errors='coerce' 
    )
    
    # Calculate oa_bin_50 and handle NaN values
    is_numeric = journal_policy_df['% of Citable OA'].notna()
    
    journal_policy_df['oa_bin_50'] = np.where(
        is_numeric & (journal_policy_df['% of Citable OA'] >= 50), 
        'OA>=50%', 
        'OA<50%'
    )
    # Explicitly set unconvertible (NaN) values to NaN
    journal_policy_df.loc[~is_numeric, 'oa_bin_50'] = np.nan
    
    # Merge DataFrame
    papers_df = papers_df.merge(
        journal_policy_df,
        how='left',
        left_on='journal',
        right_on='journal_name',
        validate='m:1'
    )
    
    # 3. Additional features
    papers_df['half_year'] = papers_df['publication_date'].dt.month.apply(lambda m: 'H1' if m <= 6 else 'H2')
    papers_df['country_list'] = papers_df['country'].fillna('').str.split(';')
    
    # 4. Vectorization (core step)
    vectorizer = CountVectorizer(binary=True, min_df=1e-6)
    # Convert AbstractText to sparse matrix, maintaining same row index as papers_df
    X_matrix = vectorizer.fit_transform(papers_df['AbstractText'].values)
    X_matrix = X_matrix.tocsr()  # Convert to CSR format for optimized row slicing
    X_matrix.eliminate_zeros()
    
    words = vectorizer.get_feature_names_out()
    word2idx = {w:i for i, w in enumerate(words)}
    
    # Build Common/Rare vocabulary groups (using 2024 data)
    
    ann = pd.read_csv(ann_path)
    alphabet = set('abcdefghijklmnopqrstuvwxyz')
    def is_allowed(w): return len(w) >= 4 and all(ch in alphabet for ch in w)
    
    style_words = [w for w,t in zip(ann['word'], ann['type'])
                   if t == 'style' and w in word2idx and is_allowed(w)]

    # Calculate 2024 annual occurrence ratio p_2024(w)
    mask_2024 = (papers_df['Year'] == 2024)
    n_2024 = int(mask_2024.sum())
    
    if n_2024 == 0:
        logger.warning("No 2024 data found for P_2024 base calculation. Falling back to simple filtering.")
        # If no 2024 data, unable to build Rare group
        counts_2024 = None
    else:
        # Extract 2024 data submatrix, maintaining index corresponding to papers_df
        counts_2024 = (X_matrix[papers_df[mask_2024].index.values, :].astype(bool).sum(axis=0)).A1
        p_2024 = (counts_2024 + 1) / (n_2024 + 1)  # (k+1)/(n+1) smoothing
        word2p2024 = {w: p for w, p in zip(words, p_2024)}

    RARE_THRESH = 0.02 
    
    COMMON10 = ['across', 'additionally', 'comprehensive', 'crucial', 'enhancing',
                'exhibited', 'insights', 'notably', 'particularly', 'within']
    
    common_words = [w for w in COMMON10 if w in word2idx]
    
    # Rare group: style words with p_2024 < 0.02 (long tail)
    if counts_2024 is not None:
        rare_words = [w for w in style_words if word2p2024.get(w, 1.0) < RARE_THRESH]
    else:
        rare_words = []
        
    logger.info(f"Common words (n={len(common_words)}): {common_words}")
    logger.info(f"Rare words (n={len(rare_words)}): {rare_words[:5]}...")

    # Keep only necessary feature columns and index
    cols_to_keep = [
        'Year', 'Month', 'half_year',
        'has_ai_policy', 'policy_category', 'is_oa', 'oa_bin_50',
        'country_list', 'domain', 'field'
    ]
    # Use reset_index to ensure index corresponds to sparse matrix row number
    df_for_grouping = papers_df.reset_index(drop=False)[['index'] + cols_to_keep].set_index('index')
    
    # Cache results: DF, sparse matrix, vocabulary tools
    joblib.dump({
        'df': df_for_grouping, 
        'X_matrix': X_matrix, 
        'word2idx': word2idx,
        'common_words': common_words,
        'rare_words': rare_words
    }, cache_file)
    logger.info(f"Cached processed data, matrix, and word lists to {cache_file}")
    
    return df_for_grouping, X_matrix, word2idx, common_words, rare_words

# --- Grouped analysis and save function (adapted for parallel processing) ---

def process_group_for_llm_delta(group_name, group_df, df_full, X_matrix, common_words, rare_words, word2idx, group_by_cols):
    
    # 1. Calculate monthly Delta
    delta_results = calculate_llm_delta(
        group_df, X_matrix, common_words, rare_words, word2idx
    )
    
    if delta_results.empty:
        return None

    # 2. Add grouping columns (force length matching)
    if group_by_cols: 
        N_rows = len(delta_results)
        
        if not isinstance(group_name, tuple) or len(group_by_cols) == 1:
            # Single column grouping
            # Use np.full() to avoid broadcasting errors
            delta_results[group_by_cols[0]] = np.full(N_rows, group_name)
        else:
            # Multi-column grouping
            for i, col in enumerate(group_by_cols):
                # Use np.full()
                delta_results[col] = np.full(N_rows, group_name[i])
        
        final_cols = group_by_cols + ['Year', 'Month', 'count', 'Delta_avg', 'Delta_common', 'Delta_rare']
        
    else:
        # Special case for by_year_month.csv
        final_cols = ['Year', 'Month', 'count', 'Delta_avg', 'Delta_common', 'Delta_rare']

    # 3. Return results
    delta_results = delta_results.rename(columns={'N_docs': 'count'})
    
    return delta_results[final_cols]


def run_llm_delta_analysis(df_full, X_matrix, word_tools, scenarios, num_processes):
    """
    Run grouped LLM Delta analysis and save results.
    """
    
    word2idx, common_words, rare_words = word_tools
    
    for df_data, group_by_cols, filename in scenarios:
        
        current_df = df_data
        
        # Skip special case (no grouping) as it is handled separately
        if not group_by_cols:
            continue
            
        logger.info(f"Running LLM Delta analysis for: {filename} on columns {group_by_cols}")

        groups = current_df.groupby(group_by_cols)
        all_results = []
        
        # Use parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_group = {
                executor.submit(
                    process_group_for_llm_delta,
                    name, 
                    group_df, 
                    df_full, 
                    X_matrix, 
                    common_words, 
                    rare_words, 
                    word2idx, 
                    group_by_cols
                ): name
                for name, group_df in groups
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_group), 
                               total=len(groups), desc=f"Analyzing {filename}"):
                res = future.result()
                if res is not None:
                    all_results.append(res)

        output_path = RESULTS_FOLDER / filename
        if all_results:
            results_df = pd.concat(all_results, ignore_index=True)
            # Ensure output is 2023-2025H1 (required for Delta mechanism)
            results_df = results_df[results_df['Year'] >= 2023]
            
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"Saved LLM Delta results to {output_path}")
        else:
            logger.warning(f"No results generated for {filename}")


# --- 4. Main function ---

def main():
    start_time = time.time()
    
    cpu_count = os.cpu_count() or 8
    max_workers = int(cpu_count * 0.8)

    # --- A. Data Preparation ---
    try:
        if os.path.exists(CACHE_FILE):
             cached_data = joblib.load(CACHE_FILE)
             df_full = cached_data['df']
             X_matrix = cached_data['X_matrix']
             word_tools = (cached_data['word2idx'], cached_data['common_words'], cached_data['rare_words'])
             logger.info(f"Loaded cached data from {CACHE_FILE}")
        else:
            df_full, X_matrix, word2idx, common_words, rare_words = prepare_data_for_analysis(
                PAPAERS_CSV, JOURNAL_INFO_CSV, CACHE_FILE, max_workers, ANNOTAION_CSV
            )
            word_tools = (word2idx, common_words, rare_words)
            
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        return

    # --- B. Prepare Exploded DataFrames ---
    # Explode country list
    df_country = df_full.explode('country_list').rename(columns={'country_list': 'country'}).copy()
    df_country.dropna(subset=['country'], inplace=True)
    df_country = df_country[df_country['country'] != '']
    
    # ----------------------------------------------------------------------------------
    # C. 1. Special processing: Global monthly trend
    # ----------------------------------------------------------------------------------
    logger.info("Running LLM Delta analysis for: by_year_month.csv (Global Trend)")
    
    df_by_year_month = calculate_llm_delta(
        df_full, X_matrix, word_tools[1], word_tools[2], word_tools[0]
    )
    
    df_by_year_month = df_by_year_month[df_by_year_month['Year'] >= 2023]
    df_by_year_month = df_by_year_month.rename(columns={'N_docs': 'count'})
    
    output_path = RESULTS_FOLDER / 'by_year_month.csv'
    df_by_year_month.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved global LLM Delta results to {output_path}")
    
    
    # --------------------------------------------------------------------------
    # C. 2. Analysis scenarios
    # --------------------------------------------------------------------------
    scenarios = [
        # Policy grouping
        (df_full, ['has_ai_policy'], 'by_policy_h.csv'), 
        (df_full, ['policy_category'], 'by_policy_c.csv'),

        # Region/Domain/Field
        (df_country, ['country'], 'by_country.csv'),
        (df_full, ['domain'], 'by_domain.csv'),
        (df_full, ['field'], 'by_field.csv'),

        # OA grouping
        (df_full, ['is_oa'], 'by_is_oa.csv'),
        (df_full, ['oa_bin_50'], 'by_oa_bin.csv'),
        
        # --- Combined grouping ---
        
        # Country + Policy
        (df_country, ['country', 'has_ai_policy'], 'by_country_h.csv'),
        
        # Domain + Policy
        (df_full, ['domain', 'has_ai_policy'], 'by_domain_h.csv'),
        
        # Field + Policy
        (df_full, ['field', 'has_ai_policy'], 'by_field_h.csv'),
        
        # is_oa + Policy
        (df_full, ['is_oa', 'has_ai_policy'], 'by_is_oa_h.csv'),
        
        # oa_bin_50 + Policy
        (df_full, ['oa_bin_50', 'has_ai_policy'], 'by_oa_bin_h.csv'),
    ]

    # --- D. Run analysis ---
    run_llm_delta_analysis(df_full, X_matrix, word_tools, scenarios, max_workers)

    logger.info(f"All LLM Delta analyses complete. Total time: {(time.time() - start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    main()