#!/usr/bin/env python3
"""
Knowledge Base Module - Semantic search over curated facts
Uses RAG (Retrieval-Augmented Generation) approach with sentence embeddings
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import torch

class KnowledgeBase:
    """RAG-based knowledge base with semantic search"""

    def __init__(self, kb_path: str = "knowledge_base.json", max_polymarket_cache: int = 30):
        self.kb_path = kb_path
        self.facts = []
        self.polymarket_markets = []
        self.embeddings = None
        self.max_polymarket_cache = max_polymarket_cache
        self.market_access_times = {}  # Track last access time for LRU

        # Load sentence transformer model (same as emotion classifier)
        print("[INFO] Loading sentence embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_model.eval()

        # Load knowledge base
        self.load_knowledge_base()

        # Initialize Polymarket client for fallback
        try:
            from polymarket_client import PolymarketClient
            self.polymarket_client = PolymarketClient()
        except:
            self.polymarket_client = None
            print("[WARNING] Polymarket client not available")

    def load_knowledge_base(self):
        """Load facts from JSON file and compute embeddings"""
        if os.path.exists(self.kb_path):
            with open(self.kb_path, 'r') as f:
                data = json.load(f)
                self.facts = data.get('facts', [])
                self.polymarket_markets = data.get('polymarket_markets', [])

            print(f"[INFO] Loaded {len(self.facts)} facts and {len(self.polymarket_markets)} Polymarket markets from knowledge base")

            # Compute embeddings for all facts
            if self.facts:
                fact_texts = [fact['text'] for fact in self.facts]
                print("[INFO] Computing embeddings for facts...")
                self.embeddings = self.embedding_model.encode(
                    fact_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                print(f"[INFO] Computed embeddings with shape {self.embeddings.shape}")
        else:
            print(f"[WARNING] Knowledge base not found at {self.kb_path}")
            print("[INFO] Starting with empty knowledge base")
            self.facts = []
            self.polymarket_markets = []
            self.embeddings = None

    def search(self, query: str, top_k: int = 3, threshold: float = 0.3) -> List[Dict]:
        """
        Search for relevant facts using semantic similarity

        Args:
            query: Text to search for
            top_k: Number of top results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of relevant facts with similarity scores
        """
        if not self.facts or self.embeddings is None:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.embeddings
        )

        # Get top-k results above threshold
        scores, indices = torch.topk(similarities, min(top_k, len(self.facts)))

        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            if score >= threshold:
                fact = self.facts[idx].copy()
                fact['similarity_score'] = score
                results.append(fact)

        return results

    def search_polymarket(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        Search for relevant Polymarket markets using keyword matching with API fallback

        Args:
            query: Text to search for
            top_k: Number of markets to return

        Returns:
            List of relevant Polymarket markets
        """
        import time

        query_lower = query.lower()
        results = []

        # First, search cached markets in RAG
        for market in self.polymarket_markets:
            keywords = market.get('keywords', [])
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in query_lower)

            if keyword_matches > 0:
                results.append({
                    'market': market,
                    'match_count': keyword_matches
                })

        # Sort by number of keyword matches
        results.sort(key=lambda x: x['match_count'], reverse=True)

        # If we found enough results in cache, use them
        if len(results) >= top_k:
            formatted_results = []
            for item in results[:top_k]:
                market = item['market']
                market_id = market.get('market_id')

                # Update access time for LRU
                self.market_access_times[market_id] = time.time()

                formatted_results.append({
                    'text': market['text'],
                    'source_type': 'polymarket',
                    'source_name': 'Polymarket',
                    'url': market['url'],
                    'market_id': market_id,
                    'category': market.get('category'),
                    'icon': '📊'
                })
            return formatted_results

        # Fallback to Polymarket API if not enough cached results
        if self.polymarket_client and len(results) < top_k:
            print(f"[INFO] Searching Polymarket API for: {query}")
            try:
                api_results = self.polymarket_client.find_relevant_facts(query, limit=top_k - len(results))
                api_markets = api_results.get('supporting', [])

                # Add API results to cache
                for api_fact in api_markets:
                    # Create market entry
                    market_id = f"api_{hash(api_fact.get('url', query))}"
                    new_market = {
                        'text': api_fact['text'],
                        'source': 'Polymarket',
                        'url': api_fact.get('url'),
                        'market_id': market_id,
                        'category': 'api_discovered',
                        'keywords': query_lower.split()
                    }

                    # Add to cache with LRU eviction
                    self._add_market_to_cache(new_market)

                    # Add to results
                    results.append({
                        'market': new_market,
                        'match_count': 1
                    })
            except Exception as e:
                print(f"[WARNING] Polymarket API search failed: {e}")

        # Format final results
        formatted_results = []
        for item in results[:top_k]:
            market = item['market']
            market_id = market.get('market_id')

            # Update access time
            self.market_access_times[market_id] = time.time()

            formatted_results.append({
                'text': market['text'],
                'source_type': 'polymarket',
                'source_name': 'Polymarket',
                'url': market['url'],
                'market_id': market_id,
                'category': market.get('category'),
                'icon': '📊'
            })

        return formatted_results

    def _add_market_to_cache(self, market: Dict):
        """Add a market to cache with LRU eviction"""
        import time

        market_id = market.get('market_id')

        # Check if market already exists
        if any(m.get('market_id') == market_id for m in self.polymarket_markets):
            return

        # If cache is full, evict least recently used
        if len(self.polymarket_markets) >= self.max_polymarket_cache:
            # Find LRU market
            lru_market_id = min(self.market_access_times.keys(),
                               key=lambda k: self.market_access_times.get(k, 0))

            # Remove from cache
            self.polymarket_markets = [m for m in self.polymarket_markets
                                      if m.get('market_id') != lru_market_id]
            del self.market_access_times[lru_market_id]

            print(f"[INFO] Evicted market from cache (LRU): {lru_market_id}")

        # Add new market
        self.polymarket_markets.append(market)
        self.market_access_times[market_id] = time.time()

        # Save updated cache to disk
        self.save_knowledge_base()

        print(f"[INFO] Added market to cache: {market_id}")

    def categorize_facts(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """
        Search and categorize facts as supporting or contradicting
        Also includes Polymarket markets

        Args:
            query: Text to analyze
            top_k: Number of facts to retrieve

        Returns:
            Dictionary with 'supporting' and 'contradicting' lists
        """
        relevant_facts = self.search(query, top_k=top_k, threshold=0.3)
        polymarket_results = self.search_polymarket(query, top_k=2)

        supporting = []
        contradicting = []

        # Add regular facts
        for fact in relevant_facts:
            # Check if fact supports or contradicts based on stance field
            stance = fact.get('stance', 'neutral')

            fact_data = {
                'text': fact['text'],
                'source_type': 'knowledge_base',
                'source_name': fact.get('source', 'Knowledge Base'),
                'url': fact.get('url', None),
                'similarity': fact['similarity_score'],
                'icon': '📚'
            }

            if stance == 'supporting':
                supporting.append(fact_data)
            elif stance == 'contradicting':
                contradicting.append(fact_data)
            else:
                # Neutral facts go to supporting by default
                supporting.append(fact_data)

        # Add Polymarket markets (neutral/informative, go to supporting)
        supporting.extend(polymarket_results)

        return {
            'supporting': supporting,
            'contradicting': contradicting
        }

    def add_fact(self, text: str, source: str = None, url: str = None,
                 stance: str = 'neutral', category: str = None):
        """Add a new fact to the knowledge base"""
        fact = {
            'text': text,
            'source': source,
            'url': url,
            'stance': stance,
            'category': category
        }

        self.facts.append(fact)

        # Recompute embeddings
        if self.facts:
            fact_texts = [f['text'] for f in self.facts]
            self.embeddings = self.embedding_model.encode(
                fact_texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )

        # Save to file
        self.save_knowledge_base()

    def save_knowledge_base(self):
        """Save knowledge base to JSON file"""
        with open(self.kb_path, 'w') as f:
            json.dump({
                'facts': self.facts,
                'polymarket_markets': self.polymarket_markets
            }, f, indent=2)
        print(f"[INFO] Saved {len(self.facts)} facts and {len(self.polymarket_markets)} Polymarket markets to {self.kb_path}")


def test_knowledge_base():
    """Test the knowledge base with sample queries"""
    kb = KnowledgeBase()

    # Test queries
    queries = [
        "remote work is more productive",
        "climate change is caused by humans",
        "electric vehicles are better for the environment"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        results = kb.search(query, top_k=3)

        if results:
            for i, fact in enumerate(results, 1):
                print(f"\n{i}. {fact['text']}")
                print(f"   Source: {fact.get('source', 'Unknown')}")
                print(f"   Similarity: {fact['similarity_score']:.3f}")
        else:
            print("No relevant facts found")


if __name__ == "__main__":
    test_knowledge_base()
