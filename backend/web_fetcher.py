import requests
from bs4 import BeautifulSoup
import re
import hashlib
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class WebFetcher:
    """
    Web content fetcher with safety measures and content processing pipeline.
    Implements fetch → sanitize → vet → quarantine → human review → ingest workflow.
    """

    def __init__(self, memory_manager: MemoryManager, quarantine_dir: str = "quarantine/"):
        self.memory = memory_manager
        self.quarantine_dir = quarantine_dir
        os.makedirs(quarantine_dir, exist_ok=True)

        # Safety patterns for content filtering
        self.safety_patterns = {
            'malicious_scripts': re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            'suspicious_links': re.compile(r'(javascript:|data:|vbscript:)', re.IGNORECASE),
            'excessive_caps': re.compile(r'[A-Z]{20,}'),
            'spam_indicators': re.compile(r'(buy now|click here|free money|urgent)', re.IGNORECASE)
        }

        # Trusted domains (can be configured)
        self.trusted_domains = [
            'wikipedia.org', 'github.com', 'stackoverflow.com',
            'arxiv.org', 'medium.com', 'towardsdatascience.com'
        ]

        # Content type priorities
        self.content_priorities = {
            'article': 10,
            'documentation': 9,
            'tutorial': 8,
            'research': 7,
            'blog': 6,
            'news': 5,
            'forum': 4,
            'social': 3,
            'other': 1
        }

    def fetch_url(self, url: str, user_agent: str = "BreinAI-WebFetcher/1.0") -> Dict[str, Any]:
        """
        Fetch content from a URL with safety checks.

        Args:
            url: URL to fetch
            user_agent: User agent string for requests

        Returns:
            Dictionary with fetch results and metadata
        """
        try:
            headers = {
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }

            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()

            # Basic validation
            if not self._is_valid_response(response):
                return {
                    'success': False,
                    'error': 'Invalid response content',
                    'url': url
                }

            # Extract content
            soup = BeautifulSoup(response.content, 'html.parser')
            content_data = self._extract_content(soup, url)

            # Generate content hash for deduplication
            content_hash = hashlib.sha256(content_data['text'].encode('utf-8')).hexdigest()

            fetch_result = {
                'success': True,
                'url': url,
                'content_hash': content_hash,
                'title': content_data['title'],
                'text': content_data['text'],
                'metadata': content_data['metadata'],
                'fetch_timestamp': datetime.now().isoformat(),
                'response_headers': dict(response.headers),
                'status_code': response.status_code
            }

            return fetch_result

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    def _is_valid_response(self, response: requests.Response) -> bool:
        """
        Validate HTTP response for safety and content type.
        """
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not ('text/html' in content_type or 'application/xhtml' in content_type):
            return False

        # Check content length (avoid extremely large pages)
        if len(response.content) > 10 * 1024 * 1024:  # 10MB limit
            return False

        # Check for basic HTML structure
        try:
            soup = BeautifulSoup(response.content[:1024], 'html.parser')  # Check first 1KB
            if not soup.find('html') and not soup.find('body'):
                return False
        except:
            return False

        return True

    def _extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract clean text content from HTML.
        """
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            tag.decompose()

        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()

        # Extract main content (prioritize article, main, content areas)
        content_selectors = ['article', 'main', '[class*="content"]', '[id*="content"]', 'body']
        main_content = None

        for selector in content_selectors:
            if selector in ['article', 'main']:
                main_content = soup.find(selector)
            else:
                main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body or soup

        # Extract text with basic cleaning
        text = main_content.get_text(separator=' ', strip=True)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Extract metadata
        metadata = {
            'url': url,
            'domain': urlparse(url).netloc,
            'word_count': len(text.split()),
            'content_type': self._classify_content_type(soup, url),
            'language': self._detect_language(text),
            'has_code_blocks': bool(soup.find_all(['code', 'pre'])),
            'link_count': len(soup.find_all('a', href=True))
        }

        return {
            'title': title,
            'text': text,
            'metadata': metadata
        }

    def _classify_content_type(self, soup: BeautifulSoup, url: str) -> str:
        """
        Classify the type of web content.
        """
        url_lower = url.lower()
        domain = urlparse(url).netloc.lower()

        # Check URL patterns
        if 'wikipedia.org' in domain:
            return 'encyclopedia'
        elif 'github.com' in domain:
            return 'code_repository'
        elif 'stackoverflow.com' in domain:
            return 'qa_forum'
        elif 'arxiv.org' in domain:
            return 'research_paper'
        elif any(word in url_lower for word in ['blog', 'article', 'post']):
            return 'blog'
        elif any(word in url_lower for word in ['tutorial', 'guide', 'how-to']):
            return 'tutorial'
        elif any(word in url_lower for word in ['news', 'article']):
            return 'news'

        # Check content patterns
        if soup.find('article'):
            return 'article'
        elif soup.find_all('code'):
            return 'technical'
        elif soup.find_all(['h1', 'h2', 'h3']):
            return 'documentation'

        return 'webpage'

    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns.
        """
        # Very basic detection - in production, use a proper language detection library
        if re.search(r'[а-яё]', text, re.IGNORECASE):
            return 'ru'
        elif re.search(r'[äöüß]', text, re.IGNORECASE):
            return 'de'
        elif re.search(r'[àâäéèêëïîôùûüÿç]', text, re.IGNORECASE):
            return 'fr'
        else:
            return 'en'  # Default to English

    def sanitize_content(self, content: str) -> Tuple[str, List[str]]:
        """
        Sanitize content by removing potentially harmful elements.

        Returns:
            Tuple of (sanitized_content, safety_warnings)
        """
        warnings = []
        sanitized = content

        # Remove script tags and event handlers
        original_length = len(sanitized)
        sanitized = self.safety_patterns['malicious_scripts'].sub('', sanitized)
        if len(sanitized) != original_length:
            warnings.append('Removed script tags')

        # Check for suspicious patterns
        for pattern_name, pattern in self.safety_patterns.items():
            if pattern.search(sanitized):
                warnings.append(f'Suspicious pattern detected: {pattern_name}')

        # Clean up excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        return sanitized, warnings

    def vet_content(self, fetch_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vet content for quality and safety.

        Returns:
            Dictionary with vetting results
        """
        content = fetch_result['text']
        metadata = fetch_result['metadata']

        vetting_result = {
            'approved': True,
            'score': 0,
            'reasons': [],
            'warnings': []
        }

        # Content quality checks
        word_count = metadata['word_count']
        if word_count < 50:
            vetting_result['approved'] = False
            vetting_result['reasons'].append('Content too short')
        elif word_count > 10000:
            vetting_result['warnings'].append('Very long content')

        # Domain trust check
        domain = metadata['domain']
        if not any(trusted in domain for trusted in self.trusted_domains):
            vetting_result['warnings'].append('Untrusted domain')

        # Content type priority
        content_type = metadata['content_type']
        priority = self.content_priorities.get(content_type, 1)
        vetting_result['score'] = priority * 10

        # Language check (only English for now)
        if metadata['language'] != 'en':
            vetting_result['warnings'].append('Non-English content')

        # Spam detection
        if self.safety_patterns['spam_indicators'].search(content):
            vetting_result['approved'] = False
            vetting_result['reasons'].append('Potential spam content')

        return vetting_result

    def quarantine_content(self, fetch_result: Dict[str, Any], vetting_result: Dict[str, Any]) -> str:
        """
        Store content in quarantine for human review.

        Returns:
            Quarantine file path
        """
        quarantine_data = {
            'fetch_result': fetch_result,
            'vetting_result': vetting_result,
            'quarantine_timestamp': datetime.now().isoformat(),
            'status': 'pending_review'
        }

        # Generate filename based on content hash
        content_hash = fetch_result['content_hash']
        filename = f"quarantine_{content_hash}_{int(datetime.now().timestamp())}.json"
        filepath = os.path.join(self.quarantine_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(quarantine_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Content quarantined: {filepath}")
        return filepath

    def approve_quarantined_content(self, quarantine_path: str, approved: bool = True,
                                   reviewer_notes: str = "") -> bool:
        """
        Approve or reject quarantined content after human review.
        """
        try:
            with open(quarantine_path, 'r', encoding='utf-8') as f:
                quarantine_data = json.load(f)

            quarantine_data['status'] = 'approved' if approved else 'rejected'
            quarantine_data['review_timestamp'] = datetime.now().isoformat()
            quarantine_data['reviewer_notes'] = reviewer_notes

            # Save updated quarantine data
            with open(quarantine_path, 'w', encoding='utf-8') as f:
                json.dump(quarantine_data, f, indent=2, ensure_ascii=False)

            if approved:
                # Ingest approved content
                fetch_result = quarantine_data['fetch_result']
                success = self._ingest_approved_content(fetch_result)
                if success:
                    logger.info(f"Successfully ingested approved content: {quarantine_path}")
                return success
            else:
                logger.info(f"Content rejected: {quarantine_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to process quarantined content {quarantine_path}: {e}")
            return False

    def _ingest_approved_content(self, fetch_result: Dict[str, Any]) -> bool:
        """
        Ingest approved content into memory.
        """
        try:
            content = fetch_result['text']
            metadata = fetch_result['metadata']

            # Add provenance information
            enhanced_metadata = {
                **metadata,
                'source': 'web_fetch',
                'fetch_timestamp': fetch_result['fetch_timestamp'],
                'content_hash': fetch_result['content_hash'],
                'provenance': {
                    'url': fetch_result['url'],
                    'title': fetch_result['title'],
                    'ingest_timestamp': datetime.now().isoformat()
                }
            }

            # Chunk and ingest content
            chunks = self.memory.chunk_text(content)
            for chunk in chunks:
                self.memory.add_memory(
                    chunk,
                    memory_type="stable",
                    metadata=enhanced_metadata
                )

            logger.info(f"Ingested {len(chunks)} chunks from {fetch_result['url']}")
            return True

        except Exception as e:
            logger.error(f"Failed to ingest content: {e}")
            return False

    def get_quarantine_stats(self) -> Dict[str, Any]:
        """
        Get statistics about quarantined content.
        """
        try:
            quarantine_files = [f for f in os.listdir(self.quarantine_dir) if f.endswith('.json')]

            stats = {
                'total_quarantined': len(quarantine_files),
                'pending_review': 0,
                'approved': 0,
                'rejected': 0
            }

            for filename in quarantine_files:
                filepath = os.path.join(self.quarantine_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        status = data.get('status', 'pending_review')
                        if status in stats:
                            stats[status] += 1
                except:
                    continue

            return stats

        except Exception as e:
            logger.error(f"Failed to get quarantine stats: {e}")
            return {'error': str(e)}

    def fetch_and_process_pipeline(self, url: str) -> Dict[str, Any]:
        """
        Complete pipeline: fetch → sanitize → vet → quarantine.
        """
        # Step 1: Fetch
        fetch_result = self.fetch_url(url)
        if not fetch_result['success']:
            return {'success': False, 'error': fetch_result.get('error'), 'stage': 'fetch'}

        # Step 2: Sanitize
        sanitized_text, warnings = self.sanitize_content(fetch_result['text'])
        fetch_result['text'] = sanitized_text
        fetch_result['safety_warnings'] = warnings

        # Step 3: Vet
        vetting_result = self.vet_content(fetch_result)

        # Step 4: Quarantine
        quarantine_path = self.quarantine_content(fetch_result, vetting_result)

        return {
            'success': True,
            'url': url,
            'quarantine_path': quarantine_path,
            'vetting_result': vetting_result,
            'safety_warnings': warnings,
            'stage': 'quarantined'
        }