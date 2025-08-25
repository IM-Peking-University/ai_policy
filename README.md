# Academic Journals' AI Policies Fail to Curb Surge in AI-Assisted Academic Writing

This repository contains the data and code for the research paper "Academic journals' AI policies fail to curb surge in AI-assisted academic writing" by Yongyuan He and Yi Bu from the Department of Information Management, Peking University.

## Overview

This study investigates the effectiveness of AI usage policies implemented by academic journals in regulating AI-assisted academic writing. We analyzed 5,114 JCR Q1 journals and their 5,235,012 papers published between January 2021 and June 2025. Our findings reveal that journal AI policies have limited impact on curbing the surge in AI-assisted academic writing, with significant disparities across domains, countries, and open access status.


## Project Structure

```
ai_policy/
├── ai_keywords/          # Keyword-based AI detection analysis
├── data/                 # Research datasets and journal information
│   ├── Journal.csv       # Journal metadata
│   ├── journalInfo.json  # Detailed journal information
│   ├── policy_texts.txt  # Journal policy texts
|   ├── paper_2021-2025.csv # Information of papers
│   └── openalex_field_hierarchy.json
├── drawing/              # Visualization notebooks and figures
│   ├── Fig 1.ipynb       # Figure 1: Journal distribution by categories
│   ├── Fig 2.ipynb       # Figure 2: Temporal trends in AI content
│   └── SI.ipynb          # Supplementary information figures
├── fulltext/             # Full-text analysis of AI usage disclosure
├── mle/                  # Maximum Likelihood Estimation for AI detection
├── policy_analysis/      # Journal policy classification using LLM
├── results/              # Analysis results organized by method
│   ├── keyword_res/      # Keyword-based detection results
│   ├── mle_res/          # MLE detection results
│   └── policy_res/       # Policy analysis results
└── zerogpt/             # External AI detection tool analysis
```

## Methodology

### AI Detection Methods
1. **Maximum Likelihood Estimation (MLE)**: Primary method for measuring AI content proportion
2. **Keyword Analysis**: Pattern-based detection using AI-related terminology
3. **Full-text Analysis**: Manual review of 164,579 papers for explicit AI disclosure
4. **Correlation Analysis**: Validation of consistency between detection methods

### Policy Classification
- Used GPT-4o-mini to classify journal policies
- Four categories: strict prohibition, open policy, disclosure required, not mentioned
- Manual verification and validation of LLM classifications

### Data Sources
- **Journals**: 5,114 JCR Q1 journals
- **Papers**: 5,235,012 publications (January 2021 - June 2025)
- **Metadata**: OpenAlex for author countries and paper domains
- **Policy Documents**: Journal submission guidelines and editorial policies
- **Full text**: Due to copyright restrictions, the data from PDFs downloaded via a web link obtained from OpenAlex cannot be made public.


## Usage

### 1. Keyword-based Detection
```bash
cd ai_keywords/
python analyze_keywords.py
```

### 2. MLE-based Detection
```bash
cd mle/
python analyze_papers.py
```

### 3. Policy Analysis
```bash
cd policy_analysis/
python classification_byLLM.py
```

### 4. Generate Figures
Open and run the Jupyter notebooks in the `drawing/` directory:
- `Fig 1.ipynb`: Journal distribution analysis
- `Fig 2.ipynb`: Temporal trend analysis
- `SI.ipynb`: Supplementary figures

## Results

All analysis results are stored in the `results/` directory:
- **keyword_res/**: Time series analysis by domain, country, and OA status
- **mle_res/**: MLE-based detection results with confidence intervals
- **policy_res/**: Policy classification and effectiveness analysis
- **full_text/**: AI disclosure analysis from full-text examination

