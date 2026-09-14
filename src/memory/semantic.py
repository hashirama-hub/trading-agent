from typing import List, Dict, Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False


class SemanticMemory:
    def __init__(self, host: str = "chromadb", port: int = 8000):
        if CHROMA_AVAILABLE:
            try:
                self.client = chromadb.HttpClient(host=host, port=port)
                self.collection = self.client.get_or_create_collection("trading_strategies")
                self._available = True
            except Exception:
                self.client = chromadb.Client()
                self.collection = self.client.get_or_create_collection("trading_strategies")
                self._available = False
        else:
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection("trading_strategies")
            self._available = False

    def add_strategy(self, name: str, content: str, metadata: Dict = None):
        self.collection.add(
            documents=[content],
            metadatas=[metadata or {}],
            ids=[name],
        )

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return [
            {"id": rid, "document": doc, "metadata": meta}
            for rid, doc, meta in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0]
            )
        ]

    def add_regime_pattern(self, regime: str, pattern: str, performance: Dict):
        self.add_strategy(
            name=f"{regime}_{hash(pattern) % 10000}",
            content=pattern,
            metadata={"regime": regime, "performance": performance},
        )

    def get_recent_strategies(self, limit: int = 10) -> List[Dict]:
        results = self.collection.query(n_results=limit, where={})
        return [
            {"id": rid, "document": doc, "metadata": meta}
            for rid, doc, meta in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0]
            )
        ]