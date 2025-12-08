# Since WOS scraped data is in multiple xlsx files, we've already merged all excel files together
# and deduplicated/cleaned them, so we only need to read this wos_paper.csv file

import csv

# Read supplementary data and build mapping dictionary
doi_map = {}
title_map = {}
with open('wos_paper.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Normalize title (lowercase + remove symbols)
        clean_title = ''.join(c for c in row['Article Title'].lower() if c.isalnum() or c.isspace())
        if row['DOI']:
            doi_map[row['DOI'].lower()] = row['Abstract']
        if clean_title:
            title_map[clean_title] = row['Abstract']

# Process main file
with open('Q1_papers_v1.csv', 'r', encoding='utf-8') as fin, \
     open('Q1_papers_v2.csv', 'w', encoding='utf-8', newline='') as fout:
    
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    
    for row in reader:
        if not row['abstract']:  # Only process rows with missing abstract
            # Prioritize DOI matching
            if row['doi']:
                clean_doi = row['doi'].lower()
                if clean_doi in doi_map:
                    row['abstract'] = doi_map[clean_doi]
            
            # Then use title matching
            if not row['abstract'] and row['title']:
                clean_title = ''.join(c for c in row['title'].lower() if c.isalnum() or c.isspace())
                if clean_title in title_map:
                    row['abstract'] = title_map[clean_title]
        
        writer.writerow(row)