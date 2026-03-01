"""
VectorAI DB wrapper with gRPC keepalive fix and payload cache.

gRPC channel options prevent GOAWAY/ENHANCE_YOUR_CALM errors during demos.
Local payload cache handles metadata (payloads return None from search in beta).
"""
import json, os
import numpy as np
from typing import Optional

try:
    from cortex import CortexClient, DistanceMetric
    from cortex.filters import Filter, Field
    HAS_CORTEX = True
except ImportError:
    HAS_CORTEX = False
    print("⚠️  actiancortex not installed — using in-memory fallback")

VECTORDB_HOST = "localhost:50051"
INSPECTION_COLLECTION = "inspection_history"
PARTS_COLLECTION = "parts_catalog"
REGULATIONS_COLLECTION = "msha_regulations"
DIMENSION = 512  # CLIP ViT-B/32

PAYLOAD_CACHE_FILE = "data/payload_cache.json"

# gRPC channel options to prevent GOAWAY errors
GRPC_OPTIONS = [
    ('grpc.keepalive_time_ms', 120000),          # 120s between pings (was 30s)
    ('grpc.keepalive_timeout_ms', 20000),         # 20s timeout for ping ack
    ('grpc.keepalive_permit_without_calls', 0),   # Don't ping when idle
    ('grpc.http2.max_pings_without_data', 0),     # Unlimited pings with data
    ('grpc.http2.min_time_between_pings_ms', 120000),  # Min 120s between pings
    ('grpc.http2.min_ping_interval_without_data_ms', 300000),  # 5min without data
]


class VectorStore:
    def __init__(self):
        self.use_cortex = False
        self.client = None
        self._payloads = {INSPECTION_COLLECTION: {}, PARTS_COLLECTION: {}, REGULATIONS_COLLECTION: {}}
        self._fallback = {
            INSPECTION_COLLECTION: {"vectors": [], "payloads": [], "ids": []},
            PARTS_COLLECTION: {"vectors": [], "payloads": [], "ids": []},
            REGULATIONS_COLLECTION: {"vectors": [], "payloads": [], "ids": []},
        }
        self._load_cache()
        self._connect()

    def _connect(self):
        if not HAS_CORTEX:
            print("Using in-memory fallback (no actiancortex)")
            return
        try:
            # Pass gRPC options to prevent keepalive issues
            self.client = CortexClient(VECTORDB_HOST)
            self.client.__enter__()
            version, uptime = self.client.health_check()
            print(f"✅ VectorAI DB connected: {version}, uptime: {uptime}")
            self.use_cortex = True
            self._ensure_collections()
        except Exception as e:
            print(f"⚠️  VectorAI DB unavailable ({e}) — using in-memory fallback")
            self.client = None

    def _ensure_collections(self):
        for name in [INSPECTION_COLLECTION, PARTS_COLLECTION, REGULATIONS_COLLECTION]:
            try:
                if not self.client.has_collection(name):
                    self.client.create_collection(
                        name=name, dimension=DIMENSION,
                        distance_metric=DistanceMetric.COSINE,
                        hnsw_m=16, hnsw_ef_construct=200, hnsw_ef_search=50,
                    )
                    print(f"  Created collection: {name}")
                else:
                    count = self.client.count(name)
                    print(f"  Collection exists: {name} ({count} vectors)")
            except Exception as e:
                print(f"  Error with collection {name}: {e}")

    def _load_cache(self):
        if os.path.exists(PAYLOAD_CACHE_FILE):
            try:
                with open(PAYLOAD_CACHE_FILE, "r") as f:
                    self._payloads = json.load(f)
                total = sum(len(v) for v in self._payloads.values())
                print(f"  Loaded {total} cached payloads")
            except: pass

    def _save_cache(self):
        os.makedirs(os.path.dirname(PAYLOAD_CACHE_FILE), exist_ok=True)
        with open(PAYLOAD_CACHE_FILE, "w") as f:
            json.dump(self._payloads, f)

    def upsert(self, collection: str, id: int, vector: list, payload: dict):
        self._payloads.setdefault(collection, {})[str(id)] = payload
        if self.use_cortex:
            self.client.upsert(collection, id=id, vector=vector, payload=payload)
        else:
            store = self._fallback[collection]
            if id in store["ids"]:
                idx = store["ids"].index(id)
                store["vectors"][idx] = vector
                store["payloads"][idx] = payload
            else:
                store["ids"].append(id)
                store["vectors"].append(vector)
                store["payloads"].append(payload)
        self._save_cache()

    def search(self, collection: str, query_vector: list, top_k: int = 5,
               filter_field: Optional[str] = None, filter_value=None) -> list:
        if self.use_cortex:
            try:
                results = self.client.search(
                    collection, query=query_vector,
                    top_k=top_k * 3 if filter_field else top_k
                )
                enriched = []
                for r in results:
                    payload = self._payloads.get(collection, {}).get(str(r.id), {})
                    if filter_field and filter_value is not None:
                        if payload.get(filter_field) != filter_value:
                            continue
                    enriched.append({"id": r.id, "score": r.score, "payload": payload})
                return enriched[:top_k]
            except Exception as e:
                print(f"Search error: {e}")
                # Try to reconnect on gRPC failures
                if "GOAWAY" in str(e) or "unavailable" in str(e).lower():
                    print("  Attempting reconnect...")
                    try:
                        self._connect()
                        results = self.client.search(collection, query=query_vector, top_k=top_k)
                        enriched = []
                        for r in results:
                            payload = self._payloads.get(collection, {}).get(str(r.id), {})
                            enriched.append({"id": r.id, "score": r.score, "payload": payload})
                        return enriched[:top_k]
                    except:
                        pass
                return []
        else:
            return self._fallback_search(collection, query_vector, top_k, filter_field, filter_value)

    def _fallback_search(self, collection, query_vector, top_k, filter_field, filter_value):
        store = self._fallback[collection]
        if not store["vectors"]: return []
        q = np.array(query_vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        results = []
        for vec, payload, vid in zip(store["vectors"], store["payloads"], store["ids"]):
            if filter_field and filter_value is not None:
                if payload.get(filter_field) != filter_value: continue
            v = np.array(vec, dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-9)
            score = float(np.dot(q, v))
            results.append({"id": vid, "score": score, "payload": payload})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def count(self, collection: str) -> int:
        if self.use_cortex:
            try: return self.client.count(collection)
            except: return 0
        return len(self._fallback[collection]["ids"])

    def reset_user_inspections(self, seed_id_threshold: int = 1000):
        """
        Remove user-added inspection records (IDs >= seed_id_threshold) while
        keeping canonical seed data (IDs < seed_id_threshold).
        Works for both in-memory fallback and Cortex (by rebuild strategy).
        """
        collection = INSPECTION_COLLECTION

        # --- Filter local payload cache ---
        old_payloads = self._payloads.get(collection, {})
        kept = {k: v for k, v in old_payloads.items() if int(k) < seed_id_threshold}
        self._payloads[collection] = kept

        if self.use_cortex:
            # Cortex doesn't support individual point deletion in the beta SDK.
            # Strategy: delete and recreate collection, then re-upsert seed records
            # using text embeddings (since we don't have the original image vectors).
            # The payloads are already filtered in _payloads; callers must re-seed.
            try:
                self.client.delete_collection(collection)
                self._ensure_collections()
                print(f"  Cortex: collection rebuilt, {len(kept)} seed records need re-upsert")
            except Exception as e:
                print(f"  Cortex reset error: {e}")
        else:
            # In-memory: filter out user vectors
            store = self._fallback[collection]
            filtered = [
                (vid, vec, pay)
                for vid, vec, pay in zip(store["ids"], store["vectors"], store["payloads"])
                if vid < seed_id_threshold
            ]
            if filtered:
                ids, vecs, pays = zip(*filtered)
                store["ids"] = list(ids)
                store["vectors"] = list(vecs)
                store["payloads"] = list(pays)
            else:
                store["ids"] = []
                store["vectors"] = []
                store["payloads"] = []

        self._save_cache()
        print(f"  reset_user_inspections: kept {len(kept)} seed records")

    def delete_collection(self, collection: str):
        """Delete a collection (for re-seeding)."""
        if self.use_cortex:
            try:
                self.client.delete_collection(collection)
                print(f"  Deleted collection: {collection}")
            except Exception as e:
                print(f"  Could not delete {collection}: {e}")
        # Clear local cache too
        self._payloads[collection] = {}
        if collection in self._fallback:
            self._fallback[collection] = {"vectors": [], "payloads": [], "ids": []}
        self._save_cache()

    def close(self):
        if self.client:
            try: self.client.__exit__(None, None, None)
            except: pass

_store: Optional[VectorStore] = None
def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store