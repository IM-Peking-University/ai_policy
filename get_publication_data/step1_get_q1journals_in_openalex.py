import json
import pandas as pd
import re

# Load data

df = pd.read_csv('../openalex_data/sources.tsv', sep='\t') 

with open('2023_JCR.json') as f:
    data = json.load(f)

# Extract Q1 journals
q1_journals = [
    {'name': j['journalName'], 'issn': j['issn']}
    for j in data['data'] if j.get('quartile') == 'Q1'
]

# Normalization function
norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower()) if pd.notna(s) else ''
df['norm_names'] = df['display_name'].apply(lambda x: [norm(p.strip()) for part in str(x).split('(') for p in [part.split(')')[0]] + part.split(')')[1:] if p.strip()])
df['norm_issn_list'] = df['issn'].apply(lambda x: [i.strip().strip('"') for i in str(x).strip('[]').replace("'", '"').split(', ')] if pd.notna(x) and x != 'nan' else [])

# Exact match (ISSN + name)
matches = []
for q in q1_journals:
    name, issn = q['name'], q['issn']
    found = False

    # ISSN matching
    if issn and issn != 'N/A':
        hits = df[df['norm_issn_list'].apply(lambda lst: issn in lst)]
        if not hits.empty:
            for _, r in hits.iterrows():
                matches.append({'q1_name': name, 'df_id': r['id'], 'df_name': r['display_name'], 'is_oa': r.get('is_oa', None)})
            found = True

    # Name exact matching (unmatched and no ISSN)
    if not found:
        norm_q = norm(name)
        hits = df[df['norm_names'].apply(lambda lst: any(norm_q == n for n in lst))]
        if not hits.empty:
            r = hits.iloc[0]
            matches.append({'q1_name': name, 'df_id': r['id'], 'df_name': r['display_name'], 'is_oa': r.get('is_oa', None)})
            found = True

# Manually supplement 47 unmatched journals
manual_matches = {
    "ANNALS ACADEMY OF MEDICINE SINGAPORE": ("https://openalex.org/S134891020", "Annals of the Academy of Medicine Singapore"),
    "Agriculture-Basel": ("https://openalex.org/S2737720053", "Agriculture"),
    "Agronomy-Basel": ("https://openalex.org/S2738977497", "Agronomy"),
    "Annals of Rehabilitation Medicine-ARM": ("https://openalex.org/S2754925757", "Annals of Rehabilitation Medicine"),
    "Appeal": ("https://openalex.org/S4306502613", "Appeal: Review of Current Law and Law Reform"),
    "Applied Sciences-Basel": ("https://openalex.org/S4306502639", "Applied Sciences (Web)"),
    "Australian Journal of Administrative Law": ("https://openalex.org/S128357121", "Australian Journal of Public Administration"),
    "BMJ-British Medical Journal": ("https://openalex.org/S4363604940", "British medical journal (Clinical research ed.)"),
    "Bereavement-Journal of Grief and Responses to Death": ("https://openalex.org/S178923243", "Bereavement Care"),
    "Biology-Basel": ("https://openalex.org/S153489431", "Biology"),
    "Biosafety and Health": ("https://openalex.org/S4210211253", "Biosafety and Health"),
    "Biosensors-Basel": ("https://openalex.org/S150925516", "Biosensors"),
    "Business Strategy and Development": ("https://openalex.org/S4210179146", "Business Strategy & Development"),
    "CABI Agriculture & Bioscience": ("https://openalex.org/S4210206712", "CABI Agriculture and Bioscience"),
    "Cell and Bioscience": ("https://openalex.org/S14303005", "Cell & Bioscience"),
    "Depositional Record": ("https://openalex.org/S4210194606", "The Depositional Record"),
    "Energy & Environmental Materials": ("https://openalex.org/S4210219471", "Energy & environment materials"),
    "Environmental Sciences Europe": ("https://openalex.org/S100566859", "Environmental Sciences Europe"),
    "European Journal of Psychotraumatology": ("https://openalex.org/S2737785665", "European journal of psychotraumatology"),
    "Experimental Hematology & Oncology": ("https://openalex.org/S2765007564", "Experimental Hematology and Oncology"),
    "Eye and Vision": ("https://openalex.org/S4210190797", "Eye and Vision"),
    "Folia Geographica": ("https://openalex.org/S4210177822", "Folia Geographica"),
    "Humanities & Social Sciences Communications": ("https://openalex.org/S4210206302", "Humanities and Social Sciences Communications"),
    "IEEE Communications Surveys and Tutorials": ("https://openalex.org/S23688054", "IEEE Communications Surveys & Tutorials"),
    "Internet Interventions-The Application of Information Technology in Mental and Behavioural Health": ("https://openalex.org/S4210232579", "Internet Interventions"),
    "JFR-Journal of Family Research": ("https://openalex.org/S4210205219", "Journal of Family Research"),
    "Journal of Bioresources and Bioproducts": ("https://openalex.org/S4210174242", "Journal of Bioresources and Bioproducts"),
    "Journal of Pathology Clinical Research": ("https://openalex.org/S4210226208", "The Journal of Pathology Clinical Research"),
    "Knee Surgery & Related Research": ("https://openalex.org/S2765024871", "Knee Surgery and Related Research"),
    "Lancet Digital Health": ("https://openalex.org/S4210237014", "The Lancet Digital Health"),
    "Lancet Gastroenterology & Hepatology": ("https://openalex.org/S2530914053", "The Lancet. Gastroenterology & hepatology"),
    "Lancet Microbe": ("https://openalex.org/S4210190705", "The Lancet Microbe"),
    "Lancet Planetary Health": ("https://openalex.org/S2898182138", "The Lancet Planetary Health"),
    "Lancet Regional Health-Western Pacific": ("https://openalex.org/S4210225048", "The Lancet Regional Health - Western Pacific"),
    "Life-Basel": ("https://openalex.org/S4210200765", "Life"),
    "Living Reviews in Relativity": ("https://openalex.org/S110783047", "Living Reviews in Relativity"),
    "NEW CARBON MATERIALS": ("https://openalex.org/S131385101", "New Carbon Materials"),
    "NPJ Schizophrenia": ("https://openalex.org/S4210229438", "Schizophrenia"),
    "Physics & Imaging in Radiation Oncology": ("https://openalex.org/S2898405426", "Physics and Imaging in Radiation Oncology"),
    "Plant Genome": ("https://openalex.org/S84695101", "The Plant Genome"),
    "Plants-Basel": ("https://openalex.org/S4210230202", "Plants"),
    "REVISTA ESPANOLA DE CARDIOLOGIA": ("https://openalex.org/S4210221393", "Revista Española de Cardiología (English Edition)"),
    "Research": ("https://openalex.org/S4210199742", "Research"),
    "SPACE WEATHER-THE INTERNATIONAL JOURNAL OF RESEARCH AND APPLICATIONS": ("https://openalex.org/S166492742", "Space Weather"),
    "TWMS Journal of Pure and Applied Mathematics": ("https://openalex.org/S4306533592", "TWMS Journal of Applied and Engineering Mathematics [SISTER]"),
    "Transplantation and Cellular Therapy": ("https://openalex.org/S4386621754", "Transplantation and Cellular Therapy"),
    "Ultrasound Journal": ("https://openalex.org/S3035383220", "The Ultrasound Journal")
}

for name, (oid, dname) in manual_matches.items():
    row = df[df['id'] == oid].iloc[0] if oid in df['id'].values else None
    is_oa = row['is_oa'] if row is not None else None
    matches.append({
        'q1_name': name,
        'df_id': oid,
        'df_name': dname,
        'is_oa': is_oa
    })

# Output CSV
result_df = pd.DataFrame(matches)[['q1_name', 'df_id', 'df_name', 'is_oa']]
result_df.to_csv('Q1_matched_openalex.csv', index=False)

print(f"Final match: {len(result_df)} / 5114, exported to Q1_matched_final.csv")