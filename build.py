import argparse
import configparser
import logging
import re
import sqlite3
import time
import warnings
from multiprocessing import Process, Queue, cpu_count, set_start_method
from pathlib import Path
from queue import Empty

import faiss
import numpy as np
import tiktoken
import torch
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from libzim import Archive
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
log = logging.getLogger('oracle-builder')

def article_worker(task_queue: Queue, result_queue: Queue, path_blocklist: list):
    """
    A worker process that fetches articles from ZIM files, cleans them,
    chunks them, and puts them on the result queue, obeying the blocklist.
    """
    zim_handles = {}
    enc = tiktoken.get_encoding('cl100k_base')
    CHUNK_TOKENS = 384

    def get_zim(path_str):
        if path_str not in zim_handles:
            zim_handles[path_str] = Archive(path_str)
        return zim_handles[path_str]

    def plain_text(html: bytes) -> str:
        soup = BeautifulSoup(html, 'lxml')
        for t in soup(['script', 'style', 'table', 'nav', 'footer', 'header', 'aside']):
            t.decompose()
        return re.split(r'\bSee also\b|\bReferences\b', soup.get_text(' ', strip=True), maxsplit=1)[0]

    def chunk_text(text: str):
        ids = enc.encode(text)
        for i in range(0, len(ids), CHUNK_TOKENS):
            sub_ids = ids[i:i + CHUNK_TOKENS]
            if len(sub_ids) < 64: continue
            yield enc.decode(sub_ids)

    while True:
        try:
            task = task_queue.get()
            if task is None:
                break
            
            zim_path_str, article_id = task
            article_unique_id = f"{Path(zim_path_str).name}:{article_id}"
            
            zim = get_zim(zim_path_str)
            entry = zim._get_entry_by_id(article_id)

            if any(blocked in entry.path for blocked in path_blocklist):
                result_queue.put(("ARTICLE_DONE", article_unique_id, "Skipped: Blocked Path", None))
                continue

            if entry.is_redirect or len(bytes(entry.get_item().content)) < 256:
                result_queue.put(("ARTICLE_DONE", article_unique_id, "Skipped: Redirect/Small", None))
                continue
            
            text = plain_text(bytes(entry.get_item().content))
            if len(text) < 120:
                result_queue.put(("ARTICLE_DONE", article_unique_id, "Skipped: No Text", None))
                continue

            full_path = f"{Path(zim_path_str).stem}/{entry.path}"
            for chunk in chunk_text(text):
                result_queue.put(("CHUNK", full_path, entry.title, chunk))
            
            result_queue.put(("ARTICLE_DONE", article_unique_id, "Success", None))

        except Exception as e:
            article_unique_id = f"{Path(task[0]).name}:{task[1]}" if task and len(task) == 2 else "unknown"
            result_queue.put(("ARTICLE_DONE", article_unique_id, f"Error: {e}", None))

def queue_feeder(task_queue: Queue, zim_tasks: list, num_workers: int, completed_articles: set):
    """A dedicated process to fill the task queue."""
    log.info(f"Queueing tasks...")
    for zim_task in zim_tasks:
        zim_path = zim_task['path']
        for i in range(zim_task['count']):
            article_unique_id = f"{Path(zim_path).name}:{i}"
            if article_unique_id not in completed_articles:
                task_queue.put((zim_path, i))
        
    for _ in range(num_workers):
        task_queue.put(None)

def run_build(config: configparser.ConfigParser):
    """Main function to orchestrate the knowledge base creation."""
    zim_files = [Path(p.strip()) for p in config.get('Paths', 'zim_files').split(',')]
    output_base = Path(config.get('Paths', 'output_path'))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    faiss_path = output_base.with_suffix('.faiss')
    meta_db_path = output_base.with_suffix('.sqlite')
    embed_model_name = config.get('Builder', 'embedding_model')
    embed_dim = config.getint('Builder', 'embedding_dimension')
    batch_size = config.getint('Builder', 'batch_size')
    
    cfg_workers = config.getint('Builder', 'num_workers')
    if cfg_workers == 0:
        core_count = cpu_count()
        num_workers = min(max(1, core_count - 2), 16)
        if core_count > 16:
            log.warning(f"High core count ({core_count}) detected. Capping workers to {num_workers} by default to conserve memory. You can override this by setting 'num_workers' in config.ini")
    else:
        num_workers = cfg_workers

    exclude_patterns_str = config.get('Builder', 'path_exclude_patterns', fallback='')
    path_blocklist = [pattern.strip() for pattern in exclude_patterns_str.split(',') if pattern.strip()]
    if path_blocklist:
        log.info(f"Will exclude articles with paths containing: {path_blocklist}")

    conn = sqlite3.connect(meta_db_path)
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS meta(id INTEGER PRIMARY KEY, path TEXT NOT NULL, title TEXT NOT NULL)')
    cur.execute('CREATE TABLE IF NOT EXISTS processed_articles(id TEXT PRIMARY KEY)')
    conn.commit()
    completed_articles = {row[0] for row in cur.execute("SELECT id FROM processed_articles").fetchall()}
    log.info(f"Found {len(completed_articles):,} previously processed articles.")

    tasks_to_process, total_articles = [], 0
    for zim_path in zim_files:
        if not zim_path.exists():
            log.error(f"ZIM file not found: {zim_path}. Skipping.")
            continue
        try:
            count = Archive(str(zim_path)).article_count
            tasks_to_process.append({'path': str(zim_path), 'count': count})
            total_articles += count
        except Exception as e:
            log.error(f"Could not open '{zim_path.name}': {e}")
            continue
    
    remaining_articles = total_articles - len(completed_articles)
    if not tasks_to_process or remaining_articles <= 0:
        log.info("All articles from specified ZIM files have been processed.")
        conn.close()
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Using device: {device}")
    if device == 'cpu': log.warning("Running on CPU. The embedding process will be very slow.")

    log.info(f"Loading embedding model '{embed_model_name}'...")
    model = SentenceTransformer(embed_model_name, device=device)
    
    idx = faiss.read_index(str(faiss_path)) if faiss_path.exists() else faiss.IndexHNSWFlat(embed_dim, 32)
    log.info(f"FAISS index contains {idx.ntotal:,} vectors.")

    task_queue = Queue(maxsize=num_workers * 20)
    result_queue = Queue(maxsize=num_workers * 512)
    
    log.info(f"Starting {num_workers} worker processes...")
    workers = [Process(target=article_worker, args=(task_queue, result_queue, path_blocklist)) for _ in range(num_workers)]
    for i, p in enumerate(workers):
        log.info(f"Starting worker {i+1}/{num_workers}...")
        p.start()
        time.sleep(0.5)

    feeder = Process(target=queue_feeder, args=(task_queue, tasks_to_process, num_workers, completed_articles))
    feeder.start()
    
    chunk_buffer, meta_buffer, article_done_buffer = [], [], []
    
    log.info(f"Processing {remaining_articles:,} remaining articles...")
    
    try:
        with tqdm(total=remaining_articles, desc="Processing Articles", unit="art") as pbar:
            processed_count = 0
            while processed_count < remaining_articles:
                try:
                    msg_type, data1, data2, data3 = result_queue.get(timeout=300)
                    
                    if msg_type == "CHUNK":
                        chunk_buffer.append(data3)
                        meta_buffer.append((data1, data2))
                    elif msg_type == "ARTICLE_DONE":
                        article_done_buffer.append((data1,))
                        pbar.update(1)
                        processed_count += 1
                        if "Error:" in str(data2):
                            log.warning(f"Worker failed on article {data1}: {data2}")

                        if len(article_done_buffer) >= 512:
                            cur.executemany("INSERT OR IGNORE INTO processed_articles (id) VALUES (?)", article_done_buffer)
                            conn.commit()
                            article_done_buffer.clear()
                    
                    if len(chunk_buffer) >= batch_size:
                        vecs = model.encode(chunk_buffer, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
                        idx.add(np.asarray(vecs, dtype='float32'))
                        cur.executemany('INSERT INTO meta(path, title) VALUES (?,?)', meta_buffer)
                        conn.commit()
                        chunk_buffer.clear()
                        meta_buffer.clear()
                        pbar.set_postfix(vectors=f'{idx.ntotal:,}')
                        
                except Empty:
                    log.warning("Result queue timed out. Checking worker status...")
                    if not any(p.is_alive() for p in workers) and feeder.is_alive():
                        raise RuntimeError("All worker processes terminated unexpectedly.")
                    if not feeder.is_alive() and result_queue.empty():
                        log.info("Feeder is done and result queue is empty. Finalizing...")
                        break
                    
    except (KeyboardInterrupt, Exception) as e:
        log.error(f"Build interrupted or failed: {e}", exc_info=not isinstance(e, KeyboardInterrupt))
    finally:
        log.info("Finalizing build...")
        if chunk_buffer:
            vecs = model.encode(chunk_buffer, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
            idx.add(np.asarray(vecs, dtype='float32'))
            cur.executemany('INSERT INTO meta(path, title) VALUES (?,?)', meta_buffer)
        
        if article_done_buffer:
            cur.executemany("INSERT OR IGNORE INTO processed_articles (id) VALUES (?)", article_done_buffer)
        
        conn.commit()
        faiss.write_index(idx, str(faiss_path))
        conn.close()
        
        log.info("Terminating processes...")
        if 'feeder' in locals() and feeder.is_alive(): feeder.terminate()
        for p in workers:
            if p.is_alive(): p.terminate()
        
        log.info(f"Build complete. Total vectors in index: {idx.ntotal:,}.")

if __name__ == "__main__":
    set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Builds an Offline Oracle knowledge base from ZIM files.")
    parser.add_argument('--config', default='config.ini', help='Path to the configuration file.')
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Configuration file not found. Please rename 'config.ini.template' to 'config.ini'.")
    else:
        config = configparser.ConfigParser()
        config.read(config_path)
        run_build(config)