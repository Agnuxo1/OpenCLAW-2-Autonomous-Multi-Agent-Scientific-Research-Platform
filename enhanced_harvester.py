"""
OpenCLAW Scientific Research Platform v2 - Enhanced Data Harvester
=================================================================
FIXES:
1. Exponential backoff for Semantic Scholar API (Error 429)
2. Diversified sources: ArXiv, Semantic Scholar, CORE, PubMed, bioRxiv
3. Rate-limiting protection across all providers
4. Fallback chain: if one source fails, try the next

Drop this file into your Scientific v2 repo and import the harvester functions.
"""

import os
import json
import time
import random
import logging
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket rate limiter with exponential backoff."""
    
    def __init__(self, calls_per_minute: int = 10, max_retries: int = 5):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0.0
        self.max_retries = max_retries
        self.consecutive_failures = 0
    
    def wait(self):
        """Wait appropriate time before making next call."""
        now = time.time()
        elapsed = now - self.last_call
        
        # Base wait
        wait_time = max(0, self.min_interval - elapsed)
        
        # Exponential backoff on failures
        if self.consecutive_failures > 0:
            backoff = min(300, (2 ** self.consecutive_failures) + random.uniform(0, 1))
            wait_time = max(wait_time, backoff)
            logger.info(f"  Backoff: {wait_time:.1f}s (failures: {self.consecutive_failures})")
        
        if wait_time > 0:
            time.sleep(wait_time)
        
        self.last_call = time.time()
    
    def success(self):
        self.consecutive_failures = 0
    
    def failure(self):
        self.consecutive_failures += 1
    
    @property
    def should_skip(self) -> bool:
        return self.consecutive_failures >= self.max_retries


# Global rate limiters per source
_rate_limiters = {
    'semantic_scholar': RateLimiter(calls_per_minute=5, max_retries=3),
    'arxiv': RateLimiter(calls_per_minute=15, max_retries=5),
    'core': RateLimiter(calls_per_minute=10, max_retries=3),
    'pubmed': RateLimiter(calls_per_minute=8, max_retries=3),
    'biorxiv': RateLimiter(calls_per_minute=10, max_retries=3),
}


def _http_get(url: str, headers: dict = None, timeout: int = 30) -> Optional[dict]:
    """Safe HTTP GET with JSON parsing."""
    try:
        req = urllib.request.Request(url)
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        logger.warning(f"  HTTP {e.code}: {url[:80]}...")
        return None
    except Exception as e:
        logger.warning(f"  Request failed: {e}")
        return None


# =============================================================================
# SOURCE 1: ArXiv (Primary - most reliable)
# =============================================================================
def harvest_arxiv(query: str, max_results: int = 20) -> List[Dict]:
    """Harvest papers from ArXiv API. Most reliable source."""
    limiter = _rate_limiters['arxiv']
    if limiter.should_skip:
        logger.warning("ArXiv: skipping (too many failures)")
        return []
    
    limiter.wait()
    
    encoded_query = urllib.parse.quote(query)
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=all:{encoded_query}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode('utf-8')
        
        # Parse Atom XML (simple extraction)
        entries = []
        import re
        
        entry_blocks = re.findall(r'<entry>(.*?)</entry>', data, re.DOTALL)
        for block in entry_blocks:
            title = re.search(r'<title>(.*?)</title>', block, re.DOTALL)
            summary = re.search(r'<summary>(.*?)</summary>', block, re.DOTALL)
            arxiv_id = re.search(r'<id>(.*?)</id>', block)
            published = re.search(r'<published>(.*?)</published>', block)
            authors = re.findall(r'<name>(.*?)</name>', block)
            
            if title and summary:
                entries.append({
                    'source': 'arxiv',
                    'title': title.group(1).strip().replace('\n', ' '),
                    'abstract': summary.group(1).strip().replace('\n', ' ')[:500],
                    'id': arxiv_id.group(1) if arxiv_id else '',
                    'published': published.group(1) if published else '',
                    'authors': authors[:5],
                    'harvested_at': datetime.utcnow().isoformat(),
                })
        
        limiter.success()
        logger.info(f"ArXiv: harvested {len(entries)} papers for '{query}'")
        return entries
    
    except Exception as e:
        limiter.failure()
        logger.error(f"ArXiv error: {e}")
        return []


# =============================================================================
# SOURCE 2: Semantic Scholar (with exponential backoff for 429)
# =============================================================================
def harvest_semantic_scholar(query: str, max_results: int = 20) -> List[Dict]:
    """Harvest from Semantic Scholar with robust rate-limit handling."""
    limiter = _rate_limiters['semantic_scholar']
    if limiter.should_skip:
        logger.warning("Semantic Scholar: skipping (rate limited)")
        return []
    
    limiter.wait()
    
    encoded_query = urllib.parse.quote(query)
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search?"
        f"query={encoded_query}"
        f"&limit={max_results}"
        f"&fields=title,abstract,authors,year,externalIds,publicationDate"
    )
    
    headers = {}
    ss_key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
    if ss_key:
        headers['x-api-key'] = ss_key
    
    data = _http_get(url, headers=headers)
    
    if data is None:
        limiter.failure()
        return []
    
    entries = []
    for paper in data.get('data', []):
        if paper.get('title') and paper.get('abstract'):
            entries.append({
                'source': 'semantic_scholar',
                'title': paper['title'],
                'abstract': paper['abstract'][:500],
                'id': paper.get('paperId', ''),
                'published': paper.get('publicationDate', ''),
                'authors': [a.get('name', '') for a in paper.get('authors', [])[:5]],
                'year': paper.get('year'),
                'harvested_at': datetime.utcnow().isoformat(),
            })
    
    limiter.success()
    logger.info(f"Semantic Scholar: harvested {len(entries)} papers for '{query}'")
    return entries


# =============================================================================
# SOURCE 3: CORE.ac.uk (Open Access aggregator - new source)
# =============================================================================
def harvest_core(query: str, max_results: int = 20) -> List[Dict]:
    """Harvest from CORE.ac.uk API (free tier, no key required)."""
    limiter = _rate_limiters['core']
    if limiter.should_skip:
        return []
    
    limiter.wait()
    
    encoded_query = urllib.parse.quote(query)
    url = (
        f"https://api.core.ac.uk/v3/search/works?"
        f"q={encoded_query}"
        f"&limit={max_results}"
    )
    
    headers = {}
    core_key = os.environ.get('CORE_API_KEY')
    if core_key:
        headers['Authorization'] = f'Bearer {core_key}'
    
    data = _http_get(url, headers=headers)
    
    if data is None:
        limiter.failure()
        return []
    
    entries = []
    for result in data.get('results', []):
        title = result.get('title', '')
        abstract = result.get('abstract', '')
        if title and abstract:
            entries.append({
                'source': 'core',
                'title': title,
                'abstract': abstract[:500],
                'id': str(result.get('id', '')),
                'published': result.get('publishedDate', ''),
                'authors': [a.get('name', '') for a in result.get('authors', [])[:5]],
                'harvested_at': datetime.utcnow().isoformat(),
            })
    
    limiter.success()
    logger.info(f"CORE: harvested {len(entries)} papers for '{query}'")
    return entries


# =============================================================================
# SOURCE 4: PubMed (Biomedical focus)
# =============================================================================
def harvest_pubmed(query: str, max_results: int = 10) -> List[Dict]:
    """Harvest from PubMed E-utilities (free, no key needed)."""
    limiter = _rate_limiters['pubmed']
    if limiter.should_skip:
        return []
    
    limiter.wait()
    
    # Step 1: Search for IDs
    encoded_query = urllib.parse.quote(query)
    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={encoded_query}&retmax={max_results}"
        f"&sort=date&retmode=json"
    )
    
    search_data = _http_get(search_url)
    if not search_data:
        limiter.failure()
        return []
    
    ids = search_data.get('esearchresult', {}).get('idlist', [])
    if not ids:
        limiter.success()
        return []
    
    # Step 2: Fetch summaries
    time.sleep(0.5)  # PubMed courtesy delay
    ids_str = ','.join(ids)
    fetch_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
        f"db=pubmed&id={ids_str}&retmode=json"
    )
    
    fetch_data = _http_get(fetch_url)
    if not fetch_data:
        limiter.failure()
        return []
    
    entries = []
    result = fetch_data.get('result', {})
    for pmid in ids:
        paper = result.get(pmid, {})
        title = paper.get('title', '')
        if title:
            authors = [a.get('name', '') for a in paper.get('authors', [])[:5]]
            entries.append({
                'source': 'pubmed',
                'title': title,
                'abstract': '',  # PubMed summaries don't include abstracts
                'id': f"PMID:{pmid}",
                'published': paper.get('pubdate', ''),
                'authors': authors,
                'harvested_at': datetime.utcnow().isoformat(),
            })
    
    limiter.success()
    logger.info(f"PubMed: harvested {len(entries)} papers for '{query}'")
    return entries


# =============================================================================
# SOURCE 5: bioRxiv (Preprints in biology)
# =============================================================================
def harvest_biorxiv(query: str, max_results: int = 15) -> List[Dict]:
    """Harvest recent preprints from bioRxiv."""
    limiter = _rate_limiters['biorxiv']
    if limiter.should_skip:
        return []
    
    limiter.wait()
    
    # bioRxiv API uses date ranges, not query strings for the main endpoint
    end_date = datetime.utcnow().strftime('%Y-%m-%d')
    start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    url = (
        f"https://api.biorxiv.org/details/biorxiv/"
        f"{start_date}/{end_date}/0/{max_results}"
    )
    
    data = _http_get(url)
    if not data:
        limiter.failure()
        return []
    
    entries = []
    for paper in data.get('collection', []):
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        if title:
            entries.append({
                'source': 'biorxiv',
                'title': title,
                'abstract': abstract[:500] if abstract else '',
                'id': paper.get('doi', ''),
                'published': paper.get('date', ''),
                'authors': paper.get('authors', '').split('; ')[:5],
                'harvested_at': datetime.utcnow().isoformat(),
            })
    
    limiter.success()
    logger.info(f"bioRxiv: harvested {len(entries)} preprints")
    return entries


# =============================================================================
# UNIFIED HARVESTER (Fallback chain)
# =============================================================================
def harvest_all(
    queries: List[str],
    max_per_source: int = 15,
    output_file: str = 'seed_data/training_dataset.jsonl'
) -> Tuple[List[Dict], Dict]:
    """
    Master harvester with fallback chain.
    Tries all sources, deduplicates, and saves to JSONL.
    
    Returns: (entries, stats)
    """
    all_entries = []
    stats = {
        'arxiv': 0,
        'semantic_scholar': 0,
        'core': 0,
        'pubmed': 0,
        'biorxiv': 0,
        'total': 0,
        'deduplicated': 0,
    }
    
    for query in queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Harvesting: '{query}'")
        logger.info(f"{'='*60}")
        
        # Source 1: ArXiv (most reliable)
        results = harvest_arxiv(query, max_per_source)
        all_entries.extend(results)
        stats['arxiv'] += len(results)
        
        # Source 2: Semantic Scholar (with backoff)
        results = harvest_semantic_scholar(query, max_per_source)
        all_entries.extend(results)
        stats['semantic_scholar'] += len(results)
        
        # Source 3: CORE.ac.uk
        results = harvest_core(query, max_per_source)
        all_entries.extend(results)
        stats['core'] += len(results)
        
        # Source 4: PubMed (for biomedical queries)
        if any(kw in query.lower() for kw in ['bio', 'medical', 'health', 'drug', 'protein', 'gene', 'neural', 'brain']):
            results = harvest_pubmed(query, 10)
            all_entries.extend(results)
            stats['pubmed'] += len(results)
        
        # Source 5: bioRxiv (for biology queries)
        if any(kw in query.lower() for kw in ['bio', 'cell', 'gene', 'protein', 'molecular', 'evolution']):
            results = harvest_biorxiv(query, 10)
            all_entries.extend(results)
            stats['biorxiv'] += len(results)
    
    # Deduplicate by title similarity
    seen_titles = set()
    unique_entries = []
    for entry in all_entries:
        title_key = entry['title'].lower().strip()[:80]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_entries.append(entry)
    
    stats['total'] = len(all_entries)
    stats['deduplicated'] = len(unique_entries)
    
    # Save to JSONL
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Append to existing file
    existing_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_count = sum(1 for _ in f)
    
    with open(output_file, 'a') as f:
        for entry in unique_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"HARVEST COMPLETE")
    logger.info(f"  ArXiv:            {stats['arxiv']}")
    logger.info(f"  Semantic Scholar:  {stats['semantic_scholar']}")
    logger.info(f"  CORE:             {stats['core']}")
    logger.info(f"  PubMed:           {stats['pubmed']}")
    logger.info(f"  bioRxiv:          {stats['biorxiv']}")
    logger.info(f"  Total raw:        {stats['total']}")
    logger.info(f"  After dedup:      {stats['deduplicated']}")
    logger.info(f"  Previous entries: {existing_count}")
    logger.info(f"  New total:        {existing_count + stats['deduplicated']}")
    logger.info(f"  Output:           {output_file}")
    logger.info(f"{'='*60}\n")
    
    return unique_entries, stats


# =============================================================================
# CLI entry point
# =============================================================================
if __name__ == '__main__':
    import sys
    
    # Default research topics for SEED system
    default_queries = [
        "neuromorphic computing",
        "reservoir computing hardware",
        "optical neural networks",
        "thermodynamic computing",
        "GPU-accelerated machine learning",
        "physics-based neural networks",
    ]
    
    queries = sys.argv[1:] if len(sys.argv) > 1 else default_queries
    
    entries, stats = harvest_all(queries)
    
    print(f"\n✅ Harvested {stats['deduplicated']} unique papers from {sum(1 for k,v in stats.items() if k not in ('total','deduplicated') and v > 0)} sources")
