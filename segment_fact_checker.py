#!/usr/bin/env python3
"""
Segment-Level Fact Checker
Orchestrates fact-checking from three sources: knowledge base, Polymarket, and web
"""

import asyncio
from typing import Dict, List
from knowledge_base import KnowledgeBase
from polymarket_client import PolymarketClient
import requests
from bs4 import BeautifulSoup
import re

class SegmentFactChecker:
    """Orchestrates fact-checking from multiple sources"""

    def __init__(self, kb_path: str = "knowledge_base.json"):
        self.kb = KnowledgeBase(kb_path)
        self.polymarket = PolymarketClient()
        print("[INFO] Segment fact checker initialized")

    def web_search(self, query: str, num_results: int = 3) -> List[Dict]:
        """
        Search the web for relevant information using DuckDuckGo HTML

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of web search results
        """
        try:
            # Use DuckDuckGo HTML search (no API key required)
            url = "https://html.duckduckgo.com/html/"
            params = {'q': query}
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ArgumentResolver/1.0)'
            }

            response = requests.post(url, data=params, headers=headers, timeout=10)
            response.raise_for_status()

            # Parse HTML results
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            # Find result divs
            result_divs = soup.find_all('div', class_='result', limit=num_results)

            for div in result_divs:
                # Extract title
                title_tag = div.find('a', class_='result__a')
                if not title_tag:
                    continue

                title = title_tag.get_text(strip=True)
                url = title_tag.get('href', '')

                # Extract snippet
                snippet_tag = div.find('a', class_='result__snippet')
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

                if title and snippet:
                    results.append({
                        'text': f"{title}: {snippet}",
                        'source_type': 'web',
                        'source_name': self._extract_domain(url),
                        'url': url,
                        'icon': '🌐'
                    })

            return results[:num_results]

        except Exception as e:
            print(f"[WARNING] Web search failed: {e}")
            return []

    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL"""
        try:
            match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if match:
                return match.group(1)
            return "Web"
        except:
            return "Web"

    async def check_segment_async(self, segment_text: str) -> Dict[str, List[Dict]]:
        """
        Asynchronously check a segment against all sources

        Args:
            segment_text: Text to fact-check

        Returns:
            Dictionary with 'supporting' and 'contradicting' fact lists
        """
        # Query all sources in parallel
        kb_task = asyncio.to_thread(self.kb.categorize_facts, segment_text, top_k=2)
        poly_task = asyncio.to_thread(self.polymarket.find_relevant_facts, segment_text, limit=2)
        web_task = asyncio.to_thread(self.web_search, segment_text, num_results=2)

        # Wait for all to complete
        kb_results, poly_results, web_results = await asyncio.gather(
            kb_task, poly_task, web_task, return_exceptions=True
        )

        # Handle exceptions
        if isinstance(kb_results, Exception):
            print(f"[WARNING] Knowledge base error: {kb_results}")
            kb_results = {'supporting': [], 'contradicting': []}

        if isinstance(poly_results, Exception):
            print(f"[WARNING] Polymarket error: {poly_results}")
            poly_results = {'supporting': [], 'contradicting': []}

        if isinstance(web_results, Exception):
            print(f"[WARNING] Web search error: {web_results}")
            web_results = []

        # Combine results from all sources
        all_supporting = (
            kb_results.get('supporting', []) +
            poly_results.get('supporting', []) +
            web_results
        )

        all_contradicting = (
            kb_results.get('contradicting', []) +
            poly_results.get('contradicting', [])
        )

        # If no results found, add a generic web result
        # (This ensures we never show "no data found")
        if not all_supporting and not all_contradicting:
            fallback_results = await asyncio.to_thread(
                self.web_search,
                f"{segment_text} fact check",
                num_results=1
            )
            if fallback_results:
                all_supporting.extend(fallback_results)
            else:
                # Last resort: add a placeholder
                all_supporting.append({
                    'text': f"No specific facts found for this claim. Further research recommended.",
                    'source_type': 'system',
                    'source_name': 'System',
                    'url': None,
                    'icon': 'ℹ️'
                })

        return {
            'supporting': all_supporting,
            'contradicting': all_contradicting
        }

    def check_segment(self, segment_text: str) -> Dict[str, List[Dict]]:
        """
        Synchronous wrapper for segment checking

        Args:
            segment_text: Text to fact-check

        Returns:
            Dictionary with 'supporting' and 'contradicting' fact lists
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, create task and wait for it
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(self.check_segment_async(segment_text))
            else:
                return loop.run_until_complete(self.check_segment_async(segment_text))
        except RuntimeError:
            # No event loop, create new one
            return asyncio.run(self.check_segment_async(segment_text))

    def check_all_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Check all segments in a conversation

        Args:
            segments: List of segment dictionaries with 'text' field

        Returns:
            Segments with added 'facts' field containing supporting/contradicting facts
        """
        print(f"[INFO] Fact-checking {len(segments)} segments...")

        for i, segment in enumerate(segments):
            text = segment.get('text', '')

            if text and len(text.strip()) > 5:
                print(f"  Checking segment {i+1}/{len(segments)}: {text[:50]}...")
                facts = self.check_segment(text)

                segment['facts'] = facts

                print(f"    Found {len(facts['supporting'])} supporting, "
                      f"{len(facts['contradicting'])} contradicting facts")
            else:
                # Empty segment
                segment['facts'] = {
                    'supporting': [],
                    'contradicting': []
                }

        print("[INFO] Fact-checking complete")
        return segments


def test_fact_checker():
    """Test the fact checker"""
    checker = SegmentFactChecker()

    # Test segments
    test_segments = [
        {
            'segment_id': 0,
            'speaker': 'SPEAKER_00',
            'text': 'I think remote work increases productivity by 13 percent',
            'start': 0.0,
            'end': 5.0
        },
        {
            'segment_id': 1,
            'speaker': 'SPEAKER_01',
            'text': 'I disagree, people are less productive working from home',
            'start': 5.5,
            'end': 9.0
        }
    ]

    print("Testing segment fact checker...\n")
    results = checker.check_all_segments(test_segments)

    for segment in results:
        print(f"\n{'='*60}")
        print(f"Segment {segment['segment_id']}: {segment['text']}")
        print('='*60)

        facts = segment.get('facts', {})

        print("\nSupporting facts:")
        for fact in facts.get('supporting', []):
            print(f"  [{fact['icon']}] {fact['text']}")
            print(f"      Source: {fact['source_name']}")
            if fact.get('url'):
                print(f"      URL: {fact['url']}")

        print("\nContradicting facts:")
        for fact in facts.get('contradicting', []):
            print(f"  [{fact['icon']}] {fact['text']}")
            print(f"      Source: {fact['source_name']}")
            if fact.get('url'):
                print(f"      URL: {fact['url']}")


if __name__ == "__main__":
    test_fact_checker()
