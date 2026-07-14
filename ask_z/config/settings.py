"""
ask_z/config/settings.py
─────────────────────────
Central configuration for the Ask-Z service.

All values are read from environment variables so the same codebase runs
locally (against a dev Elasticsearch), in CI, and in production on OpenShift
without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ElasticsearchSettings:
    host: str = field(
        default_factory=lambda: os.environ.get("ELASTIC_HOST", "https://localhost:9200")
    )
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("ELASTIC_API_KEY")
    )
    user: str | None = field(default_factory=lambda: os.environ.get("ELASTIC_USER"))
    password: str | None = field(
        default_factory=lambda: os.environ.get("ELASTIC_PASSWORD")
    )
    ca_cert: str | None = field(
        default_factory=lambda: os.environ.get("ELASTIC_CA_CERT")
    )
    verify_certs: bool = field(
        default_factory=lambda: os.environ.get("ELASTIC_VERIFY_CERTS", "false").lower()
        == "true"
    )
    index_name: str = field(
        default_factory=lambda: os.environ.get("ELASTIC_INDEX", "ask-z-knowledge")
    )


@dataclass(frozen=True)
class EmbeddingSettings:
    # Primary: ibm/granite-embedding-278m-multilingual via watsonx.ai
    #   - 768 dims, actively maintained (no deprecation date)
    #   - Supports: embedding, autoai_rag
    # Fallback (local / on-prem): BAAI/bge-large-en-v1.5 via sentence-transformers
    #   - Also 1024 dims locally but we match IBM's 768 for consistency
    # Model IDs confirmed: GET https://us-south.ml.cloud.ibm.com/ml/v1/foundation_model_specs
    model_name: str = field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_MODEL", "ibm/granite-embedding-278m-multilingual"
        )
    )
    # Reranker: ibm/slate-125m-english-rtrvr-v2 supports rerank natively on watsonx.ai.
    # No separate BGE reranker needed if using watsonx.ai.
    reranker_model: str = field(
        default_factory=lambda: os.environ.get(
            "RERANKER_MODEL", "ibm/slate-125m-english-rtrvr-v2"
        )
    )
    # 768 — confirmed embedding_dimension for both granite-embedding-278m and slate-125m.
    dims: int = 768
    # Max sequence length for Granite/Slate is 512 tokens (from model spec).
    chunk_size: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_SIZE", "512"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "64"))
    )


@dataclass(frozen=True)
class RetrievalSettings:
    # Top-k candidates from hybrid BM25+vector search before reranking.
    top_k_candidates: int = field(
        default_factory=lambda: int(os.environ.get("RETRIEVAL_TOP_K", "20"))
    )
    # Final chunks passed to Bob after cross-encoder reranking.
    top_k_final: int = field(
        default_factory=lambda: int(os.environ.get("RETRIEVAL_TOP_K_FINAL", "5"))
    )
    # RRF weighting: dense vector vs BM25 (must sum to 1.0).
    rrf_dense_weight: float = field(
        default_factory=lambda: float(os.environ.get("RRF_DENSE_WEIGHT", "0.6"))
    )
    rrf_bm25_weight: float = field(
        default_factory=lambda: float(os.environ.get("RRF_BM25_WEIGHT", "0.4"))
    )
    # Staleness threshold in days — chunks beyond this are flagged.
    staleness_days: int = field(
        default_factory=lambda: int(os.environ.get("STALENESS_DAYS", "90"))
    )


@dataclass(frozen=True)
class WatsonxSettings:
    # watsonx.ai REST base URL — region: us-south (Dallas).
    # Other regions: eu-de (Frankfurt), eu-gb (London), jp-tok (Tokyo).
    # Confirmed endpoint: https://us-south.ml.cloud.ibm.com/ml/v1/text/embeddings?version=2024-09-01
    api_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "WATSONX_API_URL", "https://us-south.ml.cloud.ibm.com/ml/v1"
        )
    )
    # API version date — pin to avoid breaking changes.
    api_version: str = field(
        default_factory=lambda: os.environ.get("WATSONX_API_VERSION", "2024-09-01")
    )
    # IBM Cloud IAM API key — used to obtain a Bearer token.
    # Get from: https://cloud.ibm.com/iam/apikeys
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("WATSONX_API_KEY")
    )
    # watsonx.ai project ID — visible in your project URL on cloud.ibm.com.
    # Required for all inference calls.
    project_id: str | None = field(
        default_factory=lambda: os.environ.get("WATSONX_PROJECT_ID")
    )
    # Generation model — ibm/granite-4-h-small confirmed available on your account.
    generation_model: str = field(
        default_factory=lambda: os.environ.get(
            "GENERATION_MODEL", "ibm/granite-4-h-small"
        )
    )


@dataclass(frozen=True)
class BobAPISettings:
    # IBM Bob internal API — URL and key are obtained from the Bob platform team.
    # These cannot be fetched externally; set via environment variable.
    # See: IBM internal Bob documentation / your team's Bob workspace settings.
    base_url: str = field(default_factory=lambda: os.environ.get("BOB_API_URL", ""))
    api_key: str | None = field(default_factory=lambda: os.environ.get("BOB_API_KEY"))
    timeout_seconds: int = field(
        default_factory=lambda: int(os.environ.get("BOB_TIMEOUT", "60"))
    )


@dataclass(frozen=True)
class GroqSettings:
    # Groq API — free tier, used as fallback when watsonx.ai quota is exhausted.
    # Get a free key at: https://console.groq.com/keys
    api_key: str | None = field(default_factory=lambda: os.environ.get("GROQ_API_KEY"))
    model: str = field(
        default_factory=lambda: os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    )


@dataclass(frozen=True)
class Settings:
    elasticsearch: ElasticsearchSettings = field(default_factory=ElasticsearchSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    watsonx: WatsonxSettings = field(default_factory=WatsonxSettings)
    bob: BobAPISettings = field(default_factory=BobAPISettings)
    groq: GroqSettings = field(default_factory=GroqSettings)


# Module-level singleton — import this everywhere.
settings = Settings()
