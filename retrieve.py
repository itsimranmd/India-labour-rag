"""Retrieval: BGE embeddings + BM25 hybrid + two-reranker ensemble.

Chosen by measurement over 20 configurations. See DECISIONS.md.
"""
import json, re
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANKERS = ["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"]
POOL_SIZE = 40
RRF_RETRIEVAL = 60   # damping for fusing keyword + semantic lists
RRF_RERANK = 10      # lower: both rerankers already filtered, so weight the top


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self, chunks_path="data/chunks.jsonl", device="cpu"):
        self.chunks = [json.loads(line) for line in open(chunks_path)]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.ids = [c["id"] for c in self.chunks]

        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)
        self.rerankers = [CrossEncoder(m, device=device) for m in RERANKERS]

        # BM25 is the keyword layer. It fails where embeddings succeed and vice
        # versa, which is precisely when combining two methods is worth doing.
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

        vectors = self.embedder.encode(
            [c["text"] for c in self.chunks],
            batch_size=32, normalize_embeddings=True, show_progress_bar=True,
        )
        client = chromadb.EphemeralClient()
        self.index = client.create_collection("labour", metadata={"hnsw:space": "cosine"})
        for i in range(0, len(self.chunks), 500):
            batch = self.chunks[i:i + 500]
            self.index.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                embeddings=vectors[i:i + 500].tolist(),
            )

    def semantic(self, question, k):
        vector = self.embedder.encode(question, normalize_embeddings=True)
        result = self.index.query(query_embeddings=[vector.tolist()], n_results=k)
        return result["ids"][0]

    def hybrid(self, question, k):
        """Fuse keyword and semantic rankings.

        The two scores live on incomparable scales, so RRF discards them and uses
        only position: each list contributes 1/(constant + rank).
        """
        scores = {}
        keyword_scores = self.bm25.get_scores(tokenize(question))
        for rank, idx in enumerate(np.argsort(keyword_scores)[::-1][:k]):
            cid = self.ids[idx]
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_RETRIEVAL + rank + 1)
        for rank, cid in enumerate(self.semantic(question, k)):
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_RETRIEVAL + rank + 1)
        return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]

    def search(self, question, k=4, pool=POOL_SIZE):
        """Retrieve a candidate pool, then reorder it with two cross-encoders.

        The embedder is a bi-encoder: it encodes question and passage separately,
        so it is fast but never sees them together. A cross-encoder reads the pair
        and scores relevance properly, but is far too slow to run over every chunk.
        Hence: cheap filter, expensive ranker.

        Two rerankers are used because they fail differently - one ranks the top
        well, the other finds more overall. Fusing their ranks was worth +6 points
        of Recall@4 over either alone.
        """
        candidates = self.hybrid(question, pool)
        texts = [self.by_id[cid]["text"] for cid in candidates]
        pairs = [(question, t) for t in texts]

        fused = {}
        for model in self.rerankers:
            scores = np.array(model.predict(pairs))
            for rank, idx in enumerate(np.argsort(scores)[::-1]):
                cid = candidates[idx]
                fused[cid] = fused.get(cid, 0) + 1 / (RRF_RERANK + rank)

        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [{**self.by_id[cid], "score": score} for cid, score in top]


if __name__ == "__main__":
    r = Retriever()
    for hit in r.search("How much notice before retrenching a worker?"):
        print(f"[{hit['score']:.3f}] {hit['code']} p.{hit['page']}  {hit['text'][:80]}...")
