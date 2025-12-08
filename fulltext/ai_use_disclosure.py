
import json
import os
import logging
import concurrent.futures
import time
import re
import gc
from tqdm import tqdm
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_disclosure_detection.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting AI disclosure detection")

# Define detection sections and regex patterns
SECTION_PATTERNS = {
    "Methods": [
        r"method(s|ology)?[\s\.:]*",
        r"experimental[\s\w]*section",
        r"materials[\s\w]*(and|&)[\s\w]*methods",
        r"procedure(s)?[\s\.:]*",
    ],
    "Acknowledgements": [
        r"acknowledg[a-z]*[\s\.:]*",
        r"funding[\s\.:]*",
        r"grant(s)?[\s\.:]*",
    ],
    "AI Declaration Section": [
        r"ai[\s\w]*(declaration|statement|disclosure)",
        r"(declaration|statement|disclosure)[\s\w]*ai",
        r"language[\s\w]*model[\s\w]*(declaration|statement|disclosure)",
        r"llm[\s\w]*(declaration|statement|disclosure)",
    ],
    "Cover Letter": [
        r"cover[\s\w]*letter",
        r"submission[\s\w]*letter",
    ],
    "Before References": [
        r"(?=references|bibliography|works cited)",
    ],
    "Title Page": [
        r"title[\s\w]*page",
        r"front[\s\w]*page",
        r"first[\s\w]*page",
    ],
    "Contributor Section": [
        r"author(\s+)contribution(s)?[\s\.:]*",
        r"credit[\s\w]*authorship[\s\w]*contribution[\s\w]*statement",
        r"contributor(s)?[\s\.:]*",
    ],
    "Footnote": [
        r"footnote(s)?[\s\.:]*",
    ],
    "Figure Captions": [
        r"figure[\s\w]*caption(s)?",
        r"figure[\s\w]*legend(s)?",
    ],
    "Disclosure Section": [
        r"disclosure(s)?[\s\.:]*",
        r"declaration(s)?[\s\.:]*",
    ],
    "Declaration of Interests": [
        r"declaration[\s\w]*of[\s\w]*interest(s)?",
        r"conflict[\s\w]*of[\s\w]*interest(s)?",
        r"competing[\s\w]*interest(s)?",
    ],
    "Experimental Section": [
        r"experiment(s|al)?[\s\.:]*",
    ],
    "Materials": [
        r"material(s)?[\s\.:]*",
    ],
    "Ethics Statement": [
        r"ethic(s|al)?[\s\w]*statement",
        r"ethical[\s\w]*consideration(s)?",
    ],
    "Introduction/Preface": [
        r"introduction[\s\.:]*",
        r"preface[\s\.:]*",
    ],
    "End of Manuscript": [
        r"conclusion(s)?[\s\.:]*",
        r"summary[\s\.:]*",
    ],
}

# Define AI tool disclosure detection function
def detect_ai_disclosure(text):
    """
    Detect AI tool disclosure in papers
    Returns: (bool, str, str) - has disclosure, details, location
    """
    if not text or not isinstance(text, str):
        return False, "Invalid or empty text", "Not Specified"
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Common LLMs and AI tools list
    ai_tools = [
        # Large language models
        "gpt", "chatgpt", "gpt-3", "gpt-4", "gpt-3.5", "gpt3", "gpt4",
        "claude", "claude-2", "claude-3", "anthropic",
        "bard", "gemini", "palm", "palm-2",
        "llama", "llama-2", "llama-3", "meta ai",
        "mistral", "mixtral", "falcon",
        "openai", "microsoft copilot", "bing chat", "bing ai",
        # General terms
        "large language model", "llm", "language model", "generative ai",
        "foundation model", "ai assistant", "artificial intelligence assistant",
        "ai writing assistant", "ai tool", "generative model",
        
        # Specific AI assistant tools
        "grammarly", "quillbot", "writesonic", "jasper", "copy.ai",
        "wordtune", "hemingway", "paperpal", "scite", "trinka",
        "writefull", "scholarcy", "scispace", "elicit"
    ]
    
    # Disclosure statement patterns
    disclosure_patterns = [
        # Active voice patterns
        r"(used|utilized|employed|leveraged|applied)[\s\w]{0,30}(ai|artificial intelligence|language model|llm|generative|assistant)",
        r"(assisted|supported|aided|helped|generated|created|drafted|written|edited|revised|proofread)[\s\w]{0,30}(by|with|using)[\s\w]{0,30}(ai|artificial intelligence|language model|llm|generative|assistant)",
        
        # Passive voice patterns
        r"(ai|artificial intelligence|language model|llm|generative|assistant)[\s\w]{0,30}(was|were|has been|have been)[\s\w]{0,30}(used|utilized|employed|leveraged|applied)",
        r"(manuscript|text|writing|draft|paper|article|content|language|grammar)[\s\w]{0,30}(was|were|has been|have been)[\s\w]{0,30}(assisted|supported|generated|checked|improved|enhanced|refined)[\s\w]{0,30}(by|with|using)[\s\w]{0,30}(ai|artificial intelligence|language model|llm|generative|assistant)",
        
        # Acknowledgement patterns
        r"(acknowledg|thank|gratitude|grateful)[\s\w]{0,50}(ai|artificial intelligence|language model|llm|generative|assistant)",
        r"(acknowledg|thank|gratitude|grateful)[\s\w]{0,50}(support|assistance|help|contribution)[\s\w]{0,30}(from|by|of)[\s\w]{0,30}(ai|artificial intelligence|language model|llm|generative|assistant)",
        
        # Explicit declaration patterns
        r"this[\s\w]{0,20}(paper|manuscript|article|research|study|work)[\s\w]{0,30}(uses|used|utilizes|utilized)[\s\w]{0,30}(ai|artificial intelligence|language model|llm|generative|assistant)",
        r"this[\s\w]{0,20}(paper|manuscript|article|research|study|work)[\s\w]{0,30}(is|was|has been)[\s\w]{0,30}(ai-assisted|ai-generated|ai-written|ai-enhanced)",
        
        # Negation patterns
        r"(no|not)[\s\w]{0,20}(ai|artificial intelligence|language model|llm|generative|assistant)[\s\w]{0,30}(was|were)[\s\w]{0,30}(used|utilized|employed)",
        r"(did not|didn't|have not|haven't)[\s\w]{0,20}(use|employ|utilize)[\s\w]{0,30}(ai|artificial intelligence|language model|llm|generative|assistant)",
        
        # Partial use patterns
        r"(only|just|specifically)[\s\w]{0,30}(used|utilized|employed)[\s\w]{0,30}(ai|artificial intelligence|language model|llm|generative|assistant)[\s\w]{0,30}(for|to)",
        r"(ai|artificial intelligence|language model|llm|generative|assistant)[\s\w]{0,30}(was|were)[\s\w]{0,30}(only|just|specifically)[\s\w]{0,30}(used|utilized|employed)[\s\w]{0,30}(for|to)",
    ]
    
    # Extract potential sections containing disclosure
    section_matches = []
    
    # Identify different regions by section titles
    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text_lower):
                section_matches.append((match.start(), section_name))
    
    # Sort matches by position in text
    section_matches.sort(key=lambda x: x[0])
    
    # Extract text for each section
    sections = {}
    for i, (pos, section_name) in enumerate(section_matches):
        end_pos = section_matches[i+1][0] if i < len(section_matches)-1 else min(pos + 2000, len(text))
        section_text = text[pos:end_pos]
        if section_name not in sections:
            sections[section_name] = []
        sections[section_name].append(section_text)
    
    # If no sections found via headers, fallback to paragraph extraction
    if not sections:
        # Split by double newline to get paragraphs
        paragraphs = text.split('\n\n')
        # If too many paragraphs, check only first 10 and last 10 (usually contain methods and acknowledgements)
        if len(paragraphs) > 20:
            sections["Unspecified"] = paragraphs[:10] + paragraphs[-10:]
        else:
            sections["Unspecified"] = paragraphs
    
    # Check for AI tool names in each section
    for section_name, section_texts in sections.items():
        for section_text in section_texts:
            section_lower = section_text.lower()
            
            # Check for AI tool name and context
            for tool in ai_tools:
                tool_pattern = r'\b' + re.escape(tool) + r'\b'
                for match in re.finditer(tool_pattern, section_lower):
                    # Extract context around match
                    context_start = max(0, match.start() - 150)
                    context_end = min(len(section_text), match.end() + 150)
                    context = section_text[context_start:context_end]
                    
                    # Check if context indicates AI tool usage
                    usage_indicators = [
                        "used", "using", "utilize", "with", "through", "via", "by", 
                        "assisted", "generated", "written", "produced", "created",
                        "supported", "aided", "helped", "draft", "edit", "revise",
                        "check", "improve", "enhance", "refine", "polish", "correct"
                    ]
                    
                    if any(indicator in context.lower() for indicator in usage_indicators):
                        # Clean context for display
                        clean_context = re.sub(r'\s+', ' ', context).strip()
                        return True, f"AI tool usage statement found: '{clean_context}'", section_name
            
            # Check for disclosure patterns
            for pattern in disclosure_patterns:
                match = re.search(pattern, section_lower)
                if match:
                    # Extract context around match
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(section_text), match.end() + 100)
                    context = section_text[context_start:context_end]
                    
                    # Clean context for display
                    clean_context = re.sub(r'\s+', ' ', context).strip()
                    return True, f"AI disclosure pattern found: '{clean_context}'", section_name
    
    # Check full text for explicit AI tool names
    for tool in ai_tools:
        tool_pattern = r'\b' + re.escape(tool) + r'\b'
        for match in re.finditer(tool_pattern, text_lower):
            # Extract context around match
            context_start = max(0, match.start() - 150)
            context_end = min(len(text), match.end() + 150)
            context = text[context_start:context_end]
            
            # Check if context indicates AI tool usage
            usage_indicators = [
                "used", "using", "utilize", "with", "through", "via", "by", 
                "assisted", "generated", "written", "produced", "created",
                "supported", "aided", "helped", "draft", "edit", "revise",
                "check", "improve", "enhance", "refine", "polish", "correct"
            ]
            
            if any(indicator in context.lower() for indicator in usage_indicators):
                # Clean context for display
                clean_context = re.sub(r'\s+', ' ', context).strip()
                return True, f"AI tool usage statement found: '{clean_context}'", "Within Manuscript (Unspecified)"
    
    # Check full text for disclosure patterns
    for pattern in disclosure_patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Extract context around match
            context_start = max(0, match.start() - 100)
            context_end = min(len(text), match.end() + 100)
            context = text[context_start:context_end]
            
            # Clean context for display
            clean_context = re.sub(r'\s+', ' ', context).strip()
            return True, f"AI disclosure pattern found: '{clean_context}'", "Within Manuscript (Unspecified)"
    
    return False, "No AI tool usage statement detected", "Not Specified"

def process_chunk(chunk_data):
    """Process a chunk of papers to detect AI tool disclosure"""
    results = {}
    for paper_id, paper_data in chunk_data.items():
        try:
            # Only process papers with journal name and date
            journal_name = paper_data.get('journal_name', '')
            publication_date = paper_data.get('publication_date', '')
            
            if not journal_name or not publication_date:
                continue
                
            if 'text' in paper_data and paper_data['text']:
                has_disclosure, details, location = detect_ai_disclosure(paper_data['text'])
                results[paper_id] = {
                    'has_ai_disclosure': has_disclosure,
                    'details': details,
                    'disclosure_location': location,
                    'journal_name': journal_name,
                    'publication_date': publication_date
                }
            else:
                results[paper_id] = {
                    'has_ai_disclosure': False,
                    'details': 'No text content',
                    'disclosure_location': 'Not Specified',
                    'journal_name': journal_name,
                    'publication_date': publication_date
                }
        except Exception as e:
            if journal_name and publication_date:
                results[paper_id] = {
                    'has_ai_disclosure': False,
                    'details': f'Error processing paper: {str(e)}',
                    'disclosure_location': 'Error',
                    'journal_name': journal_name,
                    'publication_date': publication_date
                }
    return results

def process_papers_in_parallel(papers_data, num_workers=None):
    """Process papers in parallel using multiple CPU cores"""
    if not num_workers:
        cpu_count = os.cpu_count()
        num_workers = max(1, int(cpu_count * 0.8)) if cpu_count else 8
        
    logger.info(f"Using {num_workers} worker processes for parallel processing")
    
    # Split papers into chunks for parallel processing
    paper_items = [(pid, data) for pid, data in papers_data.items() 
                   if data.get('journal_name') and data.get('publication_date')]
    
    logger.info(f"Found {len(paper_items)} valid papers (with journal name and date)")
    
    chunk_size = max(1, len(paper_items) // (num_workers * 2))  # Create more chunks than workers
    chunks = []
    
    for i in range(0, len(paper_items), chunk_size):
        chunk = dict(paper_items[i:i+chunk_size])
        chunks.append(chunk)
    
    logger.info(f"Split {len(paper_items)} papers into {len(chunks)} chunks for processing")
    
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(chunks), desc="Processing paper chunks"):
            try:
                chunk_results = future.result()
                results.update(chunk_results)
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
    
    return results

def analyze_results(results):
    """Generate statistics and insights from detection results"""
    total_papers = len(results)
    papers_with_disclosure = sum(1 for result in results.values() if result['has_ai_disclosure'])
    disclosure_rate = (papers_with_disclosure / total_papers * 100) if total_papers > 0 else 0
    
    # Analyze by publication date
    date_analysis = {}
    for paper_id, result in results.items():
        pub_date = result.get('publication_date', '')
        if pub_date and len(pub_date) >= 4:
            year = pub_date[:4]
            if year not in date_analysis:
                date_analysis[year] = {'total': 0, 'disclosed': 0}
            date_analysis[year]['total'] += 1
            if result['has_ai_disclosure']:
                date_analysis[year]['disclosed'] += 1
    
    # Calculate disclosure rate per year
    for year in date_analysis:
        total = date_analysis[year]['total']
        disclosed = date_analysis[year]['disclosed']
        date_analysis[year]['rate'] = (disclosed / total * 100) if total > 0 else 0
    
    # Analyze by journal
    journal_analysis = {}
    for paper_id, result in results.items():
        journal = result.get('journal_name', '')
        if journal:
            if journal not in journal_analysis:
                journal_analysis[journal] = {'total': 0, 'disclosed': 0}
            journal_analysis[journal]['total'] += 1
            if result['has_ai_disclosure']:
                journal_analysis[journal]['disclosed'] += 1
    
    # Calculate disclosure rate per journal and filter out journals with insufficient samples
    min_papers = 5  # Minimum papers required for journal to be included
    journal_rates = {}
    for journal, stats in journal_analysis.items():
        if stats['total'] >= min_papers:
            journal_rates[journal] = {
                'total': stats['total'],
                'disclosed': stats['disclosed'],
                'rate': (stats['disclosed'] / stats['total'] * 100)
            }
    
    # Sort journals by disclosure rate
    sorted_journals = sorted(journal_rates.items(), 
                           key=lambda x: x[1]['rate'], 
                           reverse=True)
    
    # Analyze by disclosure location
    location_analysis = defaultdict(int)
    for result in results.values():
        if result['has_ai_disclosure']:
            location = result.get('disclosure_location', 'Not Specified')
            location_analysis[location] += 1
    
    # Sort locations by frequency
    sorted_locations = sorted(location_analysis.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
    
    return {
        'summary': {
            'total_papers': total_papers,
            'papers_with_disclosure': papers_with_disclosure,
            'disclosure_rate': disclosure_rate
        },
        'by_date': date_analysis,
        'by_journal': dict(sorted_journals[:20]),  # Top 20 journals by disclosure rate
        'by_location': dict(sorted_locations)      # Locations sorted by frequency
    }

if __name__ == "__main__":
    start_time = time.time()
    
    # File paths
    input_file = '../enriched_pdf_texts.json'
    output_file = 'ai_disclosure_results.json'
    analysis_file = 'ai_disclosure_analysis.json'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load paper data
    logger.info(f"Loading paper data from {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        logger.info(f"Loaded {len(paper_data)} papers")
    except Exception as e:
        logger.error(f"Error loading paper data: {str(e)}")
        raise
    
    # Process papers to detect AI tool disclosure
    logger.info("Starting AI tool disclosure detection")
    results = process_papers_in_parallel(paper_data)
    logger.info(f"Completed detection for {len(results)} papers")
    
    # Generate analysis
    logger.info("Analyzing results")
    analysis = analyze_results(results)
    
    # Log summary statistics
    disclosure_count = analysis['summary']['papers_with_disclosure']
    total_count = analysis['summary']['total_papers']
    disclosure_rate = analysis['summary']['disclosure_rate']
    logger.info(f"Found {disclosure_count} papers with AI disclosure out of {total_count} ({disclosure_rate:.2f}%)")
    
    # Log disclosure location statistics
    logger.info("AI Disclosure Location Distribution:")
    for location, count in analysis['by_location'].items():
        logger.info(f"  - {location}: {count} papers ({count/disclosure_count*100:.2f}%)")
    
    # Save results
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save analysis
    logger.info(f"Saving analysis to {analysis_file}")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    # Calculate and log execution time
    execution_time = time.time() - start_time
    logger.info(f"Processing completed in {execution_time:.2f} seconds")