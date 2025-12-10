import csv
import os
import time
from math import ceil
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
import json
from multiprocessing import Process, Lock
import re

# ================== Configuration ==================
# Note: initial_url uses a temporary session ID. 
# If the session expires, manually navigate to a summary page to get a new URL.
INPUT_CSV = './unprocessed_s_data.csv'
PROCESSED_IDS_FILE = 'processed_work_ids_new.txt'
MAX_PROCESSES = 10
BATCH_SIZE_FOR_QUERY = 1000
# ===================================================

# WebDriver initialization

def get_driver(process_id):
    download_dir = os.path.join(os.getcwd(), f'new_downloads_proc_{process_id}')
    os.makedirs(download_dir, exist_ok=True)
    
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    # options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})

    try:
        driver = webdriver.Chrome(options=options)
        print(f"Process {process_id}: WebDriver started successfully, download dir: {download_dir}")
        
        # Temporary session URL
        initial_url = "https://webofscience.clarivate.cn/wos/alldb/summary/422da215-fd95-4401-b0fd-4917e62509f6-01879f9fef/relevance/1"
        driver.get(initial_url)
        return driver
        
    except Exception as e:
        print(f"Process {process_id}: WebDriver failed to start: {e}")
        return None

# Shared file safe operation functions (protected by Lock)

def load_processed_ids_safe(lock):
    processed_ids = set()
    try:
        lock.acquire()
        if os.path.exists(PROCESSED_IDS_FILE):
            with open(PROCESSED_IDS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    processed_ids.add(line.strip())
    finally:
        lock.release()
    return processed_ids

def append_processed_ids_safe(lock, work_ids):
    if not work_ids:
        return
    try:
        lock.acquire()
        with open(PROCESSED_IDS_FILE, 'a', encoding='utf-8') as f:
            for work_id in work_ids:
                f.write(work_id + '\n')
    finally:
        lock.release()
    print(f"SUCCESS: Appended {len(work_ids)} work_ids to the shared record file.")

# Helper functions

def wait_for_edit_page(driver, timeout=15):
    NEW_QUERY_TEXTAREA_ID = "search-option-0"
    
    print("Waiting for new query input box (DOI mode) to load...")
    
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, NEW_QUERY_TEXTAREA_ID))
    )
    print("Query input box located (ID: search-option-0).")
    time.sleep(1)

def handle_popups(driver):
    print("Attempting to handle popups...")
    try:
        close_button_xpath = "//button[@data-ta='close-button']"
        close_button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, close_button_xpath))
        )
        driver.execute_script("arguments[0].click();", close_button)
        print(f"  > Successfully closed popup element: {close_button_xpath}")
        time.sleep(1)
    except TimeoutException:
        print("  > No common auto-closeable popups found.")
    except Exception as e:
        print(f"  > Error handling popup: {e}")

def safe_execute(func, retries=3, delay=5):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(delay)
    raise Exception(f"Failed after {retries} retries")

def safe_click_element(driver, xpath, retries=3):
    def click():
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        driver.execute_script("arguments[0].click();", element)
    safe_execute(click, retries)

def safe_send_keys(element, keys, retries=3):
    def send_keys():
        element.clear()
        element.send_keys(keys)
    safe_execute(send_keys, retries)

# Query modification function

def direct_edit_and_search(driver, new_query):
    THE_REAL_EDIT_BUTTON_XPATH = "//div[@data-ta='search-terms']" 
    NEW_QUERY_TEXTAREA_ID = "search-option-0" 
    SEARCH_BUTTON_XPATH = "/html/body/app-wos/main/div/div[1]/div/div/div[2]/app-input-route/app-base-summary-component/app-search-friendly-display/div[1]/app-general-search-friendly-display/app-query-modifier/div[4]/div[2]/app-search-form/form/div[4]/button[2]"
    TOTAL_RECORDS_XPATH = "/html/body/app-wos/main/div/div[1]/div/div/div[2]/app-input-route/app-base-summary-component/app-search-friendly-display/div[1]/app-general-search-friendly-display/div/div/h1/span"
    
    try:
        old_records_element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, TOTAL_RECORDS_XPATH))
        )
        old_records_text = old_records_element.text
        print(f"Records count before submission: {old_records_text}")
    except Exception:
        old_records_text = ""
        print("Could not get old record count before submission, waiting for any new text.")


    print(f"\n=== Modifying query to: {new_query[:80]}... ===")
    
    try:
        print("Step 1: Attempting JS deep call to force open edit window...")
        wait = WebDriverWait(driver, 15)
        edit_element = wait.until(EC.presence_of_element_located((By.XPATH, THE_REAL_EDIT_BUTTON_XPATH)))
        driver.execute_script(
            """
            var el = arguments[0];
            for (var prop in el) {
                if (prop.startsWith('__ngContext__')) { 
                    var context = el[prop];
                    if (context && context.componentInstance && context.componentInstance.open) {
                        context.componentInstance.open();
                        return true;
                    }
                }
            }
            var evt = new MouseEvent('click', { bubbles: true, cancelable: true, view: window });
            el.dispatchEvent(evt);
            return false;
            """, edit_element
        )
        
        wait_for_edit_page(driver, timeout=10)
        print("  SUCCESS: Edit window opened.")

        print("Step 2: Clearing and entering new query...")
        query_textarea = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, NEW_QUERY_TEXTAREA_ID))
        )
        
        driver.execute_script("arguments[0].value = arguments[1];", query_textarea, new_query)
        driver.execute_script("arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", query_textarea)
        print(f"Entered: {new_query[:80]}...")
        time.sleep(2)

        print("Step 3: JS force click 'Search' button (absolute path)...")
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, SEARCH_BUTTON_XPATH))
        )
        driver.execute_script("arguments[0].click();", search_button)
        print("Forced click on 'Search' button.")

        print("Step 4: Waiting for edit page to close and records to update...")
        
        WebDriverWait(driver, 10).until(
             EC.invisibility_of_element_located((By.XPATH, "//div[@class='editing']"))
        )
        
        def records_text_has_changed(driver):
            try:
                element = driver.find_element(By.XPATH, TOTAL_RECORDS_XPATH)
                text = element.text.replace(',', '').strip()
                
                match = re.search(r'^\d+', text)
                current_records = int(match.group(0)) if match else 0
                
                return text and text != old_records_text.replace(',', '').strip() and current_records >= 0
            except:
                return False

        WebDriverWait(driver, 30).until(records_text_has_changed, "Timeout waiting for new search results, record count did not update.")
        
        print("SUCCESS: New search results page loaded and data updated.")
        
        total_records_element = driver.find_element(By.XPATH, TOTAL_RECORDS_XPATH)
        total_records_text = total_records_element.text
        print(f"New search results total: {total_records_text}")
        return True

    except Exception as e:
        print(f"ERROR: Failed to modify query: {e}")
        try:
            print(f"Error occurred")
        except Exception as se:
            print(f"Failed to save screenshot: {se}")
        return False

# Data export function

def export_data(driver, start_num, end_num, span):
    current = start_num
    count = 0
    
    process_pid = os.getpid()

    while True:
        if current > end_num:
            break

        try:
            end = current + span - 1
            if end > end_num:
                end = end_num
            
            print(f"Process {process_pid}: Exporting range {current}-{end}...")

            safe_click_element(driver, "//button[@id='export-trigger-btn']")
            
            safe_click_element(driver, "//*[@id='exportToExcelButton']/span")
            
            safe_click_element(driver, "//label[@for='radio3-input']")
            

            start_input = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.NAME, "markFrom"))
            )
            end_input = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.NAME, "markTo"))
            )

            safe_send_keys(start_input, str(current))
            safe_send_keys(end_input, str(end))
            print(f"  > Export range set: {current}-{end}")


            safe_click_element(driver, "/html/body/app-wos/main/div/div[1]/div/div/div[2]/app-input-route[1]/app-export-overlay/div/div[3]/div[2]/app-export-out-details/div/div[2]/form/div/div[1]/wos-select/button")
            safe_click_element(driver, "/html/body/app-wos/main/div/div[1]/div/div/div[2]/app-input-route[1]/app-export-overlay/div/div[3]/div[2]/app-export-out-details/div/div[2]/form/div/div[1]/wos-select/div/div/div/div[2]")
            print("  > Selected 'Full Record' content")

            safe_click_element(driver, "/html/body/app-wos/main/div/div[1]/div/div/div[2]/app-input-route[1]/app-export-overlay/div/div[3]/div[2]/app-export-out-details/div/div[2]/form/div/div[2]/button[1]/span[4]")
            print(f"  > Export task submitted: {time.strftime('%c', time.localtime(time.time()))}")
            
            time.sleep(40)

            current += span
            count += 1

        except Exception as e:
            print(f"Process {process_pid} Export Error: {e}")
            state = {'current': current, 'count': count}
            
            print("Export interrupted, attempting to refresh page to recover...")
            
            try:
                driver.refresh()
                time.sleep(10)
                handle_popups(driver)
            except WebDriverException as refresh_e:
                print(f"Page refresh failed: {refresh_e}. Exiting current process.")
                return count

    print(f"Process {process_pid} Export Completed")
    return count

# Data processing function

def extract_dois_and_work_ids():
    dois_with_ids = []
    
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Input file {INPUT_CSV} does not exist.")
        return dois_with_ids

    with open(INPUT_CSV, 'r', encoding='utf-8', newline='') as f:
        try:
            reader = csv.DictReader(f)
            for row in reader:
                work_id = row.get('work_id', '').strip()
                doi = row.get('doi', '').strip()
                
                if not doi:
                    continue
                
                doi = re.sub(r'^(https?://(dx\.)?doi\.org/|doi:)', '', doi, flags=re.IGNORECASE).strip()
                
                if doi:
                    dois_with_ids.append({
                        'work_id': work_id,
                        'doi': doi
                    })
        except KeyError:
            print("ERROR: CSV file missing 'work_id' or 'doi' columns. Please check header.")
            return []
    
    print(f"Extracted {len(dois_with_ids)} valid DOIs")
    return dois_with_ids

def generate_doi_batches(dois_with_ids, batch_size=1000):
    batches = []
    
    for i in range(0, len(dois_with_ids), batch_size):
        batch = dois_with_ids[i:i + batch_size]
        dois_batch = [item['doi'] for item in batch]
        work_ids_batch = [item['work_id'] for item in batch]
        
        query = " OR ".join(f'"{doi}"' for doi in dois_batch) 
        
        batches.append({
            'dois': dois_batch,
            'work_ids': work_ids_batch,
            'query': query,
            'batch_index': len(batches) + 1
        })
    
    print(f"Generated {len(batches)} DOI batches, max {batch_size} DOIs per batch")
    return batches

# Core worker function (executed by each process)

import os
import time

def worker_process(process_id, batches, lock):
    print(f"==================================================")
    print(f"🚀 Process {process_id} started, {len(batches)} batches to process.")
    print(f"==================================================")
    
    driver = get_driver(process_id)
    if driver is None:
        return

    print(f"⚠️ Process {process_id}: Waiting 1 minute to ensure all browser windows are loaded...")
    time.sleep(60)
    print(f"Process {process_id}: Wait complete, starting crawl.")
    
    handle_popups(driver)

    total_processed_work_ids = []
    
    for batch_index, batch in enumerate(batches):
        current_batch_count = batch_index + 1
        work_ids = batch['work_ids']
        query = batch['query']
        
        print(f"--------------------------------------------------")
        print(f"Process {process_id} - Processing Batch {current_batch_count}/{len(batches)}")
        
        processed_ids_snapshot = load_processed_ids_safe(lock)
        
        is_fully_processed = all(wid in processed_ids_snapshot for wid in work_ids if wid)
        
        if is_fully_processed:
            print(f"Process {process_id}: Batch {current_batch_count} all work_ids already in record file, skipping.")
            continue
        
        work_ids_to_mark = [wid for wid in work_ids if wid]
        if not work_ids_to_mark:
            continue

        print(f"Process {process_id}: Batch contains {len(work_ids)} work_ids (some may be processed)")

        success = direct_edit_and_search(driver, query) 
        
        if success:
            try:
                total_records_xpath = "/html/body/app-wos/main/div/div[1]/div/div/div[2]/app-input-route/app-base-summary-component/app-search-friendly-display/div[1]/app-general-search-friendly-display/div/div/h1/span"
                total_records_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, total_records_xpath))
                )
                total_records_text = total_records_element.text.replace(',', '').split()[0]
                total_records = int(total_records_text)
                print(f"Process {process_id}: Search results count: {total_records}")

                if total_records > 0:
                    print(f"Process {process_id}: Starting data export...")
                    export_data(driver, 1, total_records, 1000)
                    
                append_processed_ids_safe(lock, work_ids_to_mark)
                total_processed_work_ids.extend(work_ids_to_mark)
                
                print(f"SUCCESS: Process {process_id} - Batch {current_batch_count} completed")
                
                time.sleep(5)

            except Exception as e:
                print(f"ERROR: Process {process_id} - Export/Sync failed: {e}")
        
        else:
            print(f"ERROR: Process {process_id} - Batch {current_batch_count} search failed, skipping.")
            
    print(f"==================================================")
    print(f"🚀 Process {process_id} All tasks completed, successfully processed {len(total_processed_work_ids)} work_ids.")
    driver.quit()



# Main program scheduler

def main_multiprocess():
    
    file_lock = Lock()
    
    processed_ids = load_processed_ids_safe(file_lock)
    
    all_data = extract_dois_and_work_ids()
    
    data_to_process = [row for row in all_data if row.get('work_id') and row['work_id'] not in processed_ids]

    print(f"Total loaded records: {len(all_data)}")
    print(f"Filtered {len(all_data) - len(data_to_process)} processed work_ids.")
    print(f"Remaining {len(data_to_process)} valid DOIs to process.")
    
    if not data_to_process:
        print("All DOIs processed or no valid data, exiting.")
        return

    batches = generate_doi_batches(data_to_process, BATCH_SIZE_FOR_QUERY)
    
    processes_batches = [[] for _ in range(MAX_PROCESSES)]
    active_processes_count = 0
    for i, batch in enumerate(batches):
        processes_batches[i % MAX_PROCESSES].append(batch)
        if len(processes_batches[i % MAX_PROCESSES]) == 1:
            active_processes_count += 1
    
    print(f"Using {min(MAX_PROCESSES, len(batches))} processes for crawling.")

    processes = []
    for i in range(MAX_PROCESSES):
        if processes_batches[i]:
            p = Process(target=worker_process, 
                        args=(i + 1, processes_batches[i], file_lock))
            processes.append(p)
            p.start()
            time.sleep(2) 
            
    print("Waiting 1 minute before starting crawl...")
    time.sleep(60)

    for p in processes:
        p.join()
        
    print("\nAll crawl processes completed.")

if __name__ == "__main__":
    main_multiprocess()