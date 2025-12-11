#!/usr/bin/env python3
"""
Polymarket Client - Query prediction market data via Gamma API
"""

import requests
from typing import List, Dict, Optional
import time

class PolymarketClient:
    """Client for querying Polymarket prediction markets via Gamma API"""

    def __init__(self):
        self.base_url = "https://gamma-api.polymarket.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ArgumentResolver/1.0)'
        })

    def search_markets(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for relevant prediction markets

        Args:
            query: Search query
            limit: Maximum number of markets to return

        Returns:
            List of market data dictionaries
        """
        try:
            # Use the /markets endpoint to search
            url = f"{self.base_url}/markets"
            params = {
                'limit': limit,
                'active': True
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            markets = response.json()

            # Filter markets by query relevance
            relevant_markets = []
            query_lower = query.lower()

            for market in markets:
                # Check if query terms appear in market question or description
                question = market.get('question', '').lower()
                description = market.get('description', '').lower()

                # Simple keyword matching
                if any(term in question or term in description
                       for term in query_lower.split()):
                    relevant_markets.append(market)

            return relevant_markets[:limit]

        except Exception as e:
            print(f"[WARNING] Failed to search Polymarket markets: {e}")
            return []

    def get_market_details(self, market_id: str) -> Optional[Dict]:
        """Get detailed information about a specific market"""
        try:
            url = f"{self.base_url}/markets/{market_id}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[WARNING] Failed to get market details: {e}")
            return None

    def extract_facts_from_markets(self, markets: List[Dict]) -> List[Dict]:
        """
        Extract fact-checkable information from markets

        Args:
            markets: List of market data from Polymarket

        Returns:
            List of facts with market context
        """
        facts = []

        for market in markets:
            question = market.get('question', 'Unknown question')
            outcomes = market.get('outcomes', [])

            # Get current probabilities if available
            probabilities = []
            for outcome in outcomes:
                outcome_name = outcome.get('outcome', 'Unknown')
                price = outcome.get('price', None)

                if price is not None:
                    # Price represents probability (0-1 scale)
                    prob_pct = float(price) * 100
                    probabilities.append(f"{outcome_name}: {prob_pct:.1f}%")

            # Build fact text
            if probabilities:
                fact_text = f"{question} Current prediction: {', '.join(probabilities)}"
            else:
                fact_text = question

            # Get market URL
            condition_id = market.get('condition_id', '')
            market_url = f"https://polymarket.com/event/{condition_id}" if condition_id else None

            facts.append({
                'text': fact_text,
                'source_type': 'polymarket',
                'source_name': 'Polymarket',
                'url': market_url,
                'market_data': {
                    'question': question,
                    'outcomes': outcomes,
                    'end_date': market.get('end_date_iso'),
                    'volume': market.get('volume')
                },
                'icon': '📊'
            })

        return facts

    def find_relevant_facts(self, query: str, limit: int = 3) -> Dict[str, List[Dict]]:
        """
        Search for markets relevant to a query and categorize as supporting/contradicting

        Args:
            query: Text to analyze
            limit: Maximum number of markets to find

        Returns:
            Dictionary with 'supporting' and 'contradicting' lists
        """
        markets = self.search_markets(query, limit=limit)
        facts = self.extract_facts_from_markets(markets)

        # For now, all Polymarket facts are neutral/informative
        # They don't directly support or contradict, just provide market sentiment
        return {
            'supporting': facts,
            'contradicting': []
        }


def test_polymarket_client():
    """Test the Polymarket client"""
    client = PolymarketClient()

    # Test queries
    queries = [
        "Will Trump win the 2024 election?",
        "AI safety regulation",
        "climate change policy"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        results = client.find_relevant_facts(query, limit=3)

        supporting = results['supporting']
        if supporting:
            print(f"\nFound {len(supporting)} relevant markets:")
            for i, fact in enumerate(supporting, 1):
                print(f"\n{i}. {fact['text']}")
                print(f"   URL: {fact.get('url', 'N/A')}")
        else:
            print("\nNo relevant markets found")

        time.sleep(1)  # Rate limiting


if __name__ == "__main__":
    test_polymarket_client()
