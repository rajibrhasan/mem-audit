"""Native memory adapters and an explicitly labelled SQLite test fixture.

Snapshots preserve native records AND indexes, rather than regenerating an
index from records and accidentally repairing the deletion bug under study.
Only this evaluator module can enumerate storage. Attack code cannot.
"""
import copy
import json
import logging
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch
import uuid


class SQLiteFixture:
    name = "fixture"

    def __init__(self, path, config=None, client=None):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(Path(path) / "fixture.sqlite"))
        self.db.execute("CREATE TABLE notes (id TEXT PRIMARY KEY, text TEXT)")

    def add(self, text, raw=False):
        key = str(uuid.uuid4())
        self.db.execute("INSERT INTO notes VALUES (?,?)", (key, text))
        self.db.commit()
        return [key]

    def records(self):
        return [{"id": x, "text": y} for x, y in self.db.execute("SELECT id,text FROM notes ORDER BY id")]

    def search(self, query, k):
        return sorted(self.records(), key=lambda r: len(set(query.lower().split()) &
                      set(r["text"].lower().split())), reverse=True)[:k]

    def delete(self, ids):
        for key in ids:
            self.db.execute("DELETE FROM notes WHERE id=?", (key,))
        self.db.commit()

    def snapshot(self):
        return {"records": self.records()}

    def restore(self, snapshot):
        if self.records():
            raise ValueError("Restore requires an empty destination")
        self.db.executemany("INSERT INTO notes VALUES (:id,:text)", snapshot["records"])
        self.db.commit()

    def inspect(self):
        return {"records": self.records(), "indexes": self.records(), "history": [],
                "inspection_scope": "fixture logical rows"}

    def close(self):
        self.db.close()


class Mem0Backend:
    name = "mem0"

    def __init__(self, path, config, client):
        os.environ["MEM0_TELEMETRY"] = "false"
        os.environ.setdefault("MEM0_DIR", str(Path(path).resolve().parent / "mem0-config"))
        from mem0 import Memory
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        self.user = "audit-subject"
        self.history_path = path / "history.sqlite"
        llm = config["llm"]
        native_config = {
            "llm": {"provider": "openai", "config": {
                "model": llm["model"], "openai_base_url": llm["base_url"],
                "api_key": os.environ.get(llm.get("api_key_env", "LOCAL_LLM_API_KEY"), "local"),
                "temperature": llm.get("temperature", 0), "max_tokens": llm.get("memory_max_tokens", 2048)}},
            "embedder": {"provider": "huggingface", "config": {
                "model": config["embedding_model"], "model_kwargs": {"device": "cpu"}}},
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": "audit", "path": str(path / "qdrant"),
                "embedding_model_dims": config["embedding_dims"], "on_disk": True}},
            "history_db_path": str(self.history_path),
        }
        from mem0.embeddings import huggingface
        from .embeddings import encoder
        with patch.object(huggingface, "SentenceTransformer", lambda model, **kwargs: encoder(model)):
            self.memory = Memory.from_config(native_config)
        class LocalLLM:
            def generate_response(self, messages, response_format=None, **kwargs):
                if kwargs.get("tools"):
                    raise ValueError("This audit transport does not execute model tools")
                return client.complete(messages, json_mode=bool(response_format),
                                       max_tokens=llm.get("memory_max_tokens", llm.get("max_tokens", 2048)))
        # Keep native extraction/evolution logic, but use the same local
        # transport and generation controls for both memory systems.
        self.memory.llm = LocalLLM()

    def add(self, text, raw=False):
        result = self.memory.add([{"role": "user", "content": text}], user_id=self.user, infer=not raw)
        return [r["id"] for r in result["results"] if r.get("event") in ("ADD", "UPDATE")]

    def records(self):
        # Directly paginate the underlying index: get_all has a bounded top_k.
        rows = []
        for point in self._points("audit"):
            payload = point["payload"] or {}
            if payload.get("user_id") == self.user:
                rows.append({"id": str(point["id"]), "text": payload.get("data", ""), "metadata": payload})
        return sorted(rows, key=lambda x: x["id"])

    def search(self, query, k):
        result = self.memory.search(query, filters={"user_id": self.user}, top_k=k)
        return [{"id": r["id"], "text": r["memory"]} for r in result["results"]]

    def delete(self, ids):
        for key in ids:
            self.memory.delete(key)

    def _points(self, collection):
        client, offset, rows = self.memory.vector_store.client, None, []
        while True:
            points, offset = client.scroll(collection, limit=256, offset=offset,
                                           with_payload=True, with_vectors=True)
            rows.extend(p.model_dump(mode="json") for p in points)
            if offset is None:
                return rows

    def snapshot(self):
        client = self.memory.vector_store.client
        collections = {}
        for col in client.get_collections().collections:
            collections[col.name] = {"config": client.get_collection(col.name).config.params.model_dump(mode="json"),
                                     "points": self._points(col.name)}
        # sqlite backup/restore through SQL includes histories, not only live rows.
        history = "\n".join(self.memory.db.connection.iterdump())
        return {"collections": collections, "history": history}

    def restore(self, snapshot):
        from qdrant_client import models
        client = self.memory.vector_store.client
        for name, data in snapshot["collections"].items():
            if client.collection_exists(name):
                if client.count(name).count:
                    raise ValueError("Restore requires an empty destination")
            else:
                params = data["config"]
                vectors = params.get("vectors")
                if vectors and "size" in vectors:
                    vectors = models.VectorParams(**vectors)
                elif vectors:
                    vectors = {k: models.VectorParams(**v) for k, v in vectors.items()}
                sparse = params.get("sparse_vectors")
                client.create_collection(name, vectors_config=vectors or {},
                                         sparse_vectors_config={k: models.SparseVectorParams(**v)
                                                                for k, v in (sparse or {}).items()} or None)
            if data["points"]:
                client.upsert(name, [models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                                     for p in data["points"]])
        db = self.memory.db.connection
        for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            if not name.startswith("sqlite_"):
                db.execute('DROP TABLE "' + name.replace('"', '""') + '"')
        db.commit()
        db.executescript(snapshot["history"])

    def inspect(self):
        snapshot = self.snapshot()
        # Vectors omitted from inspection text, retained in reproducibility snapshot.
        indexes = [{"collection": name, "id": str(p["id"]), "payload": p["payload"]}
                   for name, data in snapshot["collections"].items() for p in data["points"]]
        return {"records": self.records(), "indexes": indexes, "history": snapshot["history"],
                "inspection_scope": "all active local Qdrant collections and SQLite history; not erased disk blocks/backups"}

    def close(self):
        self.memory.close()
        self.memory.vector_store.client.close()


class AMemBackend:
    name = "amem"

    def __init__(self, path, config, client):
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        from agentic_memory import memory_system as module
        from agentic_memory.retrievers import ChromaRetriever
        self.module = module
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        embedding_model = config["embedding_model"]

        class ScopedRetriever(ChromaRetriever):
            def __init__(self, collection_name="memories", model_name=embedding_model):
                # The upstream constructor resets its client; scope it to this
                # newly created experiment directory, never a shared client.
                self.client = chromadb.PersistentClient(path=str(path / "chroma"),
                                                       settings=Settings(allow_reset=True, anonymized_telemetry=False))
                self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name, device="cpu")
                self.collection = self.client.get_or_create_collection(collection_name,
                                                                       embedding_function=self.embedding_function)

        class LocalLLM:
            def get_completion(self, prompt, response_format=None, **kwargs):
                # Use the exact same local backbone for native evolution.
                return client.complete([{"role": "user", "content": prompt}], json_mode=bool(response_format),
                                       response_format=response_format,
                                       max_tokens=config["llm"].get("memory_max_tokens", config["llm"].get("max_tokens", 2048)))

        self.retriever_class = ScopedRetriever
        with patch.object(module, "ChromaRetriever", ScopedRetriever), patch.object(
                module, "LLMController", lambda *a, **k: SimpleNamespace(llm=LocalLLM())):
            self.memory = module.AgenticMemorySystem(model_name=embedding_model,
                                                     llm_model=config["llm"]["model"])

    def add(self, text, raw=False):
        # Preserve native add_note and evolution; only redirect retriever storage.
        # Upstream catches evolution exceptions and returns an unchanged note.
        # Detect that fallback so a failed evolution cannot become valid data.
        errors = []
        class CaptureErrors(logging.Handler):
            def emit(self, record):
                errors.append(record.getMessage())
        handler = CaptureErrors(level=logging.ERROR)
        self.module.logger.addHandler(handler)
        try:
            with patch.object(self.module, "ChromaRetriever", self.retriever_class):
                ids = [self.memory.add_note(text)]
            if errors:
                raise RuntimeError("A-MEM ingestion/evolution failed; case invalid, inspect native error log")
            return ids
        finally:
            self.module.logger.removeHandler(handler)

    def records(self):
        return [{"id": key, "text": note.content, "metadata": copy.deepcopy(vars(note))}
                for key, note in sorted(self.memory.memories.items())]

    def search(self, query, k):
        if not self.memory.memories:
            return []
        return [{"id": r["id"], "text": r["content"] + "\nContext: " + r.get("context", "")}
                for r in self.memory.search(query, k=min(k, len(self.memory.memories))) ]

    def delete(self, ids):
        for key in ids:
            if not self.memory.delete(key):
                raise ValueError(f"A-MEM could not delete selected record {key}")

    def snapshot(self):
        index = self.memory.retriever.collection.get(include=["documents", "metadatas", "embeddings"])
        if hasattr(index.get("embeddings"), "tolist"):
            index["embeddings"] = index["embeddings"].tolist()
        return {"notes": {k: copy.deepcopy(vars(v)) for k, v in self.memory.memories.items()},
                "index": {k: index[k] for k in ("ids", "documents", "metadatas", "embeddings")},
                "evo_cnt": self.memory.evo_cnt}

    def restore(self, snapshot):
        if self.memory.memories or self.memory.retriever.collection.count():
            raise ValueError("Restore requires an empty destination")
        self.memory.memories = {k: self.module.MemoryNote(**v) for k, v in snapshot["notes"].items()}
        data = snapshot["index"]
        if data["ids"]:
            self.memory.retriever.collection.upsert(**data)
        self.memory.evo_cnt = snapshot["evo_cnt"]

    def inspect(self):
        snapshot = self.snapshot()
        index = snapshot["index"]
        return {"records": self.records(), "indexes": [
            {"id": key, "text": text, "metadata": meta} for key, text, meta in
            zip(index["ids"], index["documents"], index["metadatas"])], "history": [],
            "inspection_scope": "native notes, links, tags, contexts, evolution history and live Chroma index; not erased disk blocks/backups"}

    def close(self):
        # Chroma does not expose a public close on PersistentClient. Retain
        # artifacts; the runner isolates each fact in a worker process.
        pass


def make_backend(name, path, config, client):
    return {"fixture": SQLiteFixture, "mem0": Mem0Backend, "amem": AMemBackend}[name](path, config, client)
