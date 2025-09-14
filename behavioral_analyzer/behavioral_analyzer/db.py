"""
MongoDB persistence utilities for Behavioral Analyzer.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except Exception:  # pragma: no cover - optional dependency during tests
    MongoClient = None  # type: ignore
    PyMongoError = Exception  # type: ignore


class MongoPersistence:
    """Thin wrapper over PyMongo for storing analyzer data."""

    def __init__(self, uri: str, database: str,
                 combined_collection: str = "combined_data",
                 sessions_collection: str = "sessions"):
        if MongoClient is None:
            raise RuntimeError("pymongo is not installed. Install with 'pip install pymongo[srv]'")
        self.client = MongoClient(uri, appname="behavioral-analyzer")
        self.db = self.client[database]
        self.combined = self.db[combined_collection]
        self.sessions = self.db[sessions_collection]

    def insert_combined(self, document: Dict[str, Any]) -> Optional[str]:
        try:
            doc = dict(document)
            if "timestamp" not in doc:
                doc["timestamp"] = time.time()
            result = self.combined.insert_one(doc)
            return str(result.inserted_id)
        except PyMongoError as e:  # pragma: no cover
            print(f"Mongo insert_combined error: {e}")
            return None

    def insert_session(self, document: Dict[str, Any]) -> Optional[str]:
        try:
            result = self.sessions.insert_one(dict(document))
            return str(result.inserted_id)
        except PyMongoError as e:  # pragma: no cover
            print(f"Mongo insert_session error: {e}")
            return None

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


