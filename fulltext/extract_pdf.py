import os
import re
import json
import logging
import time
import traceback
import argparse
import subprocess
import tempfile
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# ======== Configuration Parameters ========
DEFAULT_CONFIG = {
    'PDF_DIRS': [''],  # List of PDF directories
    'OUTPUT_DIR': 'results',  # Output directory
    'MAX_WORKERS': 100,  # Number of parallel threads
    'BATCH_SIZE': 500,  # Batch size; smaller size reduces memory pressure
    'MIN_TEXT_LENGTH': 50,  # Minimum text length
    'DEBUG': False,  # Debug mode
    'MAX_PDF_SIZE_MB': 50,  # Max PDF size (MB); files exceeding this are skipped
    'MAX_TEXT_LENGTH': 10000000,  # Max text length limit to prevent memory overuse
    'SUSPECT_PREFIXES': ['https___', 'http___'],  # Suspicious filename prefixes processed with safer methods
}

# ======== Logging Setup ========
def setup_logging(output_dir):
    """Configure logging"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'pdf_processing.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ======== Simplified PDF Processor Class ========
class SimplePDFProcessor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.output_json = os.path.join(config['OUTPUT_DIR'], 'pdf_texts.json')
        
        # Ensure output directory exists
        os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    
    def clean_text(self, text):
        """Clean extracted text"""
        if not text:
            return ""
            
        # Limit text length to prevent memory overflow
        if len(text) > self.config['MAX_TEXT_LENGTH']:
            text = text[:self.config['MAX_TEXT_LENGTH']]
            
        # Remove reference sections
        references_patterns = [
            r'(?i)References\s*\n.*?($|\Z)',
            r'(?i)Bibliography\s*\n.*?($|\Z)',
            r'(?i)引用文献\s*\n.*?($|\Z)',
            r'(?i)参考文献\s*\n.*?($|\Z)',
            r'(?i)REFERENCES\s*\n.*?($|\Z)'
        ]
        
        for pattern in references_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # Remove headers and footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Remove page numbers
        text = re.sub(r'\n.{0,50}Vol\.\s*\d+.{0,30}\n', '\n', text)  # Remove volume numbers
        text = re.sub(r'\n.{0,50}20\d{2}.{0,30}\n', '\n', text)  # Remove years
        
        # Clean clutter characters
        text = re.sub(r' {2,}', ' ', text)  # Excess spaces
        text = re.sub(r'\n{3,}', '\n\n', text)  # Excess newlines
        
        # Remove DOI/URL
        text = re.sub(r'(?i)doi:?\s*10\.\d+/\S+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        
        return text.strip()
    
    def is_suspect_file(self, pdf_file):
        """Check if filename is suspicious (files that might cause crashes)"""
        for prefix in self.config['SUSPECT_PREFIXES']:
            if pdf_file.startswith(prefix):
                return True
        return False
    
    def is_valid_pdf(self, pdf_path):
        """Check if file is a valid PDF"""
        try:
            # Check if file header complies with PDF spec
            with open(pdf_path, 'rb') as f:
                header = f.read(10)
                return header.startswith(b'%PDF')
        except:
            return False
    
    def extract_text_with_pymupdf_safe(self, pdf_path):
        """Extract text safely using PyMuPDF (for normal PDFs)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    text += page.get_text() + "\n"
                except Exception:
                    pass
            doc.close()
            return text
        except Exception:
            return ""
    
    def extract_text_with_pdftotext(self, pdf_path):
        """Extract text using pdftotext command line tool"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.txt') as temp:
                # Run pdftotext command with timeout to prevent hanging
                try:
                    subprocess.run(['pdftotext', pdf_path, temp.name], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=30)  # 30 second timeout
                    
                    # Read output
                    with open(temp.name, 'r', errors='ignore') as f:
                        text = f.read()
                        
                    return text
                except:
                    return ""
        except:
            return ""
    
    def process_pdf_file(self, pdf_info):
        """Process a single PDF file and return text content"""
        pdf_dir, pdf_file = pdf_info
        pdf_path = os.path.join(pdf_dir, pdf_file)
        paper_id = os.path.splitext(pdf_file)[0]
        
        try:
            # Check file size
            try:
                if os.path.getsize(pdf_path) / (1024 * 1024) > self.config['MAX_PDF_SIZE_MB']:
                    return paper_id, None
            except:
                return paper_id, None
                
            # Extract text
            text = ""
            
            # Skip PyMuPDF and use pdftotext directly for suspicious files
            if self.is_suspect_file(pdf_file) or not self.is_valid_pdf(pdf_path):
                text = self.extract_text_with_pdftotext(pdf_path)
            else:
                # Try PyMuPDF for normal files
                if HAS_PYMUPDF:
                    text = self.extract_text_with_pymupdf_safe(pdf_path)
                
                # If PyMuPDF fails or is unavailable, try pdftotext
                if not text:
                    text = self.extract_text_with_pdftotext(pdf_path)
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Check if text is empty or too short
            if not cleaned_text or len(cleaned_text) < self.config['MIN_TEXT_LENGTH']:
                return paper_id, None
                
            return paper_id, cleaned_text
            
        except Exception:
            return paper_id, None
        finally:
            # Force garbage collection
            gc.collect()
    
    def process_pdfs(self):
        """Process all PDF files"""
        # Get all PDF files
        all_pdf_files = []
        for pdf_dir in self.config['PDF_DIRS']:
            if os.path.exists(pdf_dir):
                pdf_files = [(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
                all_pdf_files.extend(pdf_files)
                self.logger.info(f"Found {len(pdf_files)} PDF files in directory {pdf_dir}")
            else:
                self.logger.warning(f"Directory {pdf_dir} does not exist")
        
        if not all_pdf_files:
            self.logger.error("No PDF files found in any specified directories")
            return
        
        self.logger.info(f"Total {len(all_pdf_files)} PDF files to process")
        
        # Load existing results (for resuming)
        existing_results = {}
        if os.path.exists(self.output_json):
            try:
                with open(self.output_json, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                self.logger.info(f"Loaded {len(existing_results)} existing results")
            except Exception as e:
                self.logger.error(f"Failed to load existing results: {str(e)}")
        
        # Filter out already processed files
        all_pdf_files = [(pdf_dir, pdf_file) for pdf_dir, pdf_file in all_pdf_files 
                      if os.path.splitext(pdf_file)[0] not in existing_results]
        
        if not all_pdf_files:
            self.logger.info("All PDF files have been processed")
            return
        
        self.logger.info(f"Number of PDFs remaining to process: {len(all_pdf_files)}")
        
        # Batch processing
        total_batches = (len(all_pdf_files) + self.config['BATCH_SIZE'] - 1) // self.config['BATCH_SIZE']
        processed_count = 0
        success_count = 0
        failed_count = 0
        start_time = time.time()
        
        # Count suspicious files
        suspect_files = [pdf_file for _, pdf_file in all_pdf_files if self.is_suspect_file(pdf_file)]
        if suspect_files:
            self.logger.info(f"Detected {len(suspect_files)} suspicious filenames, will process with safer method")
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.config['BATCH_SIZE']
            end_idx = min((batch_num + 1) * self.config['BATCH_SIZE'], len(all_pdf_files))
            batch_files = all_pdf_files[start_idx:end_idx]
            batch_success = 0
            
            # Process PDFs using multithreading
            with ThreadPoolExecutor(max_workers=self.config['MAX_WORKERS']) as executor:
                # Create tasks
                futures = {executor.submit(self.process_pdf_file, pdf_info): pdf_info 
                           for pdf_info in batch_files}
                
                # Process results
                for future in as_completed(futures):
                    processed_count += 1
                    try:
                        paper_id, text = future.result()
                        if text:
                            existing_results[paper_id] = text
                            success_count += 1
                            batch_success += 1
                        else:
                            failed_count += 1
                            
                        # Print progress
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        eta_minutes = (len(all_pdf_files) - processed_count) / rate / 60 if rate > 0 else 0
                            
                        print(f"\rProgress: {processed_count}/{len(all_pdf_files)} "
                              f"({processed_count/len(all_pdf_files)*100:.1f}%) - "
                              f"Success: {success_count}, Failed: {failed_count} - "
                              f"Speed: {rate:.2f} items/sec, ETA: {eta_minutes:.1f} min", end="")
                            
                    except Exception:
                        failed_count += 1
            
            print()  # Newline
            
            # Save results per batch
            try:
                with open(self.output_json, 'w', encoding='utf-8') as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Batch {batch_num+1} complete, added {batch_success} results, total {len(existing_results)}")
            except Exception as e:
                self.logger.error(f"Failed to save results: {str(e)}")
                
            # Clean memory between batches
            gc.collect()
        
        # Final stats
        self.logger.info(f"Processing complete: Success {success_count}, Failed {failed_count}, Total {len(existing_results)}")

# ======== Main Function ========
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Simplified PDF Text Extraction Tool')
    parser.add_argument('--pdf_dirs', type=str, nargs='+', help='List of PDF file directories')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--workers', type=int, help='Number of parallel threads')
    parser.add_argument('--batch_size', type=int, help='Batch processing size')
    parser.add_argument('--min_length', type=int, help='Minimum text length')
    parser.add_argument('--max_size', type=int, help='Max PDF size (MB)')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    # Merge command line args with default config
    config = DEFAULT_CONFIG.copy()
    if args.pdf_dirs:
        config['PDF_DIRS'] = args.pdf_dirs
    if args.output_dir:
        config['OUTPUT_DIR'] = args.output_dir
    if args.workers:
        config['MAX_WORKERS'] = args.workers
    if args.batch_size:
        config['BATCH_SIZE'] = args.batch_size
    if args.min_length:
        config['MIN_TEXT_LENGTH'] = args.min_length
    if args.max_size:
        config['MAX_PDF_SIZE_MB'] = args.max_size
    if args.debug:
        config['DEBUG'] = True
    
    # Create output directory
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config['OUTPUT_DIR'])
    logger.info("=== Simplified PDF Text Extraction Started ===")
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Check if PDF directories exist
    valid_dirs = [d for d in config['PDF_DIRS'] if os.path.exists(d)]
    if not valid_dirs:
        logger.error(f"None of the specified PDF directories exist")
        return
    
    # Check extraction method
    if HAS_PYMUPDF:
        logger.info("Will use PyMuPDF for text extraction")
    else:
        logger.info("PyMuPDF not installed, will attempt to use pdftotext command line tool")
        try:
            subprocess.run(['pdftotext', '-v'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            logger.info("pdftotext command available")
        except Exception:
            logger.error("pdftotext command issue, some PDFs may not be extracted")
    
    # Process PDFs
    processor = SimplePDFProcessor(config, logger)
    processor.process_pdfs()
    
    logger.info("=== PDF Text Extraction Complete ===")

if __name__ == "__main__":
    main()