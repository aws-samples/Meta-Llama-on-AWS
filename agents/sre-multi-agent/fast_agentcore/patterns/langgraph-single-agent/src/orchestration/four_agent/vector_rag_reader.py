"""Vector-based RAG knowledge retrieval for RCA agent.

This module implements true Retrieval-Augmented Generation using:
1. Vector embeddings (AWS Bedrock Titan Embeddings)
2. FAISS vector store for semantic search
3. Document chunking strategies
4. Top-K retrieval with relevance scoring

Replaces the previous keyword-based pattern matching approach.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import boto3
    import faiss  # Check FAISS library itself
    from langchain_community.vectorstores import FAISS
    from langchain_core.embeddings import Embeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    VECTOR_DEPS_AVAILABLE = True
except ImportError as e:
    VECTOR_DEPS_AVAILABLE = False
    logger.warning(
        f"Vector search dependencies not available ({e}). "
        "Install with: uv add langchain langchain-community langchain-text-splitters faiss-cpu"
    )


class BedrockEmbeddings(Embeddings):
    """AWS Bedrock embeddings using Titan Embed Text model."""

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v1",
        region_name: str = "us-east-1",
    ):
        """Initialize Bedrock embeddings client.

        Args:
            model_id: Bedrock model ID for embeddings
            region_name: AWS region
        """
        if not VECTOR_DEPS_AVAILABLE:
            raise ImportError(
                "Vector search dependencies required. "
                "Install with: pip install langchain langchain-community faiss-cpu boto3"
            )

        self.model_id = model_id
        self.region_name = region_name
        self.client = boto3.client("bedrock-runtime", region_name=region_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self._embed_single(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return self._embed_single(text)

    def _embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.invoke_model(
                modelId=self.model_id, body=json.dumps({"inputText": text})
            )

            response_body = json.loads(response["body"].read())
            return response_body.get("embedding", [])
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback (Titan Embed Text v1 = 1024 dimensions)
            return [0.0] * 1024


class VectorRAGKnowledgeReader:
    """Vector-based RAG knowledge retrieval for SRE troubleshooting.

    This class replaces keyword-based pattern matching with semantic search using:
    - Vector embeddings for all policy documents
    - FAISS for efficient similarity search
    - Chunking strategy for long documents
    - Top-K retrieval with relevance scoring
    """

    def __init__(
        self,
        runbooks_path: str | None = None,
        patterns_path: str | None = None,
        cache_dir: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ):
        """Initialize vector-based RAG reader.

        Args:
            runbooks_path: Path to troubleshooting-runbooks.md
            patterns_path: Path to known-failure-patterns.md
            cache_dir: Directory to cache vector store
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
            top_k: Number of top results to retrieve
        """
        if not VECTOR_DEPS_AVAILABLE:
            raise ImportError(
                "Vector search dependencies required. "
                "Install with: pip install langchain langchain-community faiss-cpu boto3"
            )

        # Set default paths
        if runbooks_path is None:
            root = Path(__file__).resolve().parents[3]
            runbooks_path = root / "docs" / "policies" / "troubleshooting-runbooks.md"

        if patterns_path is None:
            root = Path(__file__).resolve().parents[3]
            patterns_path = root / "docs" / "policies" / "known-failure-patterns.md"

        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[3] / ".vector_cache"

        self.runbooks_path = Path(runbooks_path)
        self.patterns_path = Path(patterns_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Initialize components
        self.embeddings = BedrockEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )

        # Initialize or load vector stores
        self.runbooks_store: FAISS | None = None
        self.patterns_store: FAISS | None = None
        self._initialize_vector_stores()

    def _initialize_vector_stores(self):
        """Initialize vector stores from documents or cache."""
        # Check if cached vector stores exist
        runbooks_index_name = "runbooks_vectorstore"
        patterns_index_name = "patterns_vectorstore"

        # Calculate content hashes to detect changes
        runbooks_hash = self._calculate_file_hash(self.runbooks_path)
        patterns_hash = self._calculate_file_hash(self.patterns_path)

        hash_file = self.cache_dir / "content_hashes.json"
        cached_hashes = {}
        if hash_file.exists():
            with open(hash_file) as f:
                cached_hashes = json.load(f)

        # Check if runbooks cache is valid
        runbooks_faiss_file = self.cache_dir / f"{runbooks_index_name}.faiss"
        if (
            runbooks_faiss_file.exists()
            and cached_hashes.get("runbooks") == runbooks_hash
        ):
            logger.info("Loading runbooks vector store from cache")
            try:
                self.runbooks_store = self._load_vector_store(runbooks_index_name)
            except Exception as e:
                logger.warning(f"Failed to load runbooks cache, rebuilding: {e}")
                self.runbooks_store = self._build_vector_store(self.runbooks_path)
                self._save_vector_store(self.runbooks_store, runbooks_index_name)
                cached_hashes["runbooks"] = runbooks_hash
        else:
            logger.info("Building runbooks vector store")
            self.runbooks_store = self._build_vector_store(self.runbooks_path)
            self._save_vector_store(self.runbooks_store, runbooks_index_name)
            cached_hashes["runbooks"] = runbooks_hash

        # Check if patterns cache is valid
        patterns_faiss_file = self.cache_dir / f"{patterns_index_name}.faiss"
        if (
            patterns_faiss_file.exists()
            and cached_hashes.get("patterns") == patterns_hash
        ):
            logger.info("Loading patterns vector store from cache")
            try:
                self.patterns_store = self._load_vector_store(patterns_index_name)
            except Exception as e:
                logger.warning(f"Failed to load patterns cache, rebuilding: {e}")
                self.patterns_store = self._build_vector_store(self.patterns_path)
                self._save_vector_store(self.patterns_store, patterns_index_name)
                cached_hashes["patterns"] = patterns_hash
        else:
            logger.info("Building patterns vector store")
            self.patterns_store = self._build_vector_store(self.patterns_path)
            self._save_vector_store(self.patterns_store, patterns_index_name)
            cached_hashes["patterns"] = patterns_hash

        # Save updated hashes
        with open(hash_file, "w") as f:
            json.dump(cached_hashes, f)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content."""
        if not file_path.exists():
            return ""

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _build_vector_store(self, file_path: Path) -> FAISS:
        """Build FAISS vector store from markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            FAISS vector store with embedded chunks
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Policy document not found: {file_path}")

        # Load and chunk document
        content = file_path.read_text()
        chunks = self.text_splitter.split_text(content)

        logger.info(f"Split {file_path.name} into {len(chunks)} chunks")

        # Create vector store with chunks
        vector_store = FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            metadatas=[
                {"source": file_path.name, "chunk_id": i} for i in range(len(chunks))
            ],
        )

        return vector_store

    def _save_vector_store(self, vector_store: FAISS, index_name: str):
        """Save vector store to cache.

        Args:
            vector_store: FAISS vector store to save
            index_name: Name for the index files (without extension)
        """
        try:
            vector_store.save_local(str(self.cache_dir), index_name=index_name)
            logger.info(f"Saved vector store to {self.cache_dir}/{index_name}")
        except Exception as e:
            logger.warning(f"Failed to save vector store: {e}")

    def _load_vector_store(self, index_name: str) -> FAISS:
        """Load vector store from cache.

        Args:
            index_name: Name of the index files (without extension)

        Returns:
            Loaded FAISS vector store
        """
        try:
            vector_store = FAISS.load_local(
                str(self.cache_dir),
                self.embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,  # We control the cache
            )
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load vector store from cache: {e}")
            raise

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert L2 distance to similarity score (0-1 range, higher is better).

        Args:
            distance: L2 distance from FAISS (0 = identical, higher = more different)

        Returns:
            Similarity score between 0 and 1
        """
        # Use inverse function to convert distance to similarity
        # This ensures: distance=0 -> similarity=1, large distance -> similarity≈0
        # Convert to Python float to avoid numpy type serialization issues
        return float(1.0 / (1.0 + float(distance)))

    def _sanitize_for_json(self, obj):
        """Recursively sanitize objects to ensure JSON serializability.

        Handles numpy types, nested dicts/lists, and other objects.
        """
        try:
            import numpy as np

            if isinstance(obj, dict):
                return {k: self._sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [self._sanitize_for_json(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)
        except (ImportError, AttributeError):
            # Numpy not available or obj doesn't have expected attrs
            if isinstance(obj, dict):
                return {k: self._sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [self._sanitize_for_json(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)

    def get_troubleshooting_steps(self, error_pattern: str) -> dict[str, Any]:
        """Retrieve diagnostic steps using semantic search.

        Args:
            error_pattern: Error message or symptom description

        Returns:
            Dictionary with troubleshooting information
        """
        if self.runbooks_store is None:
            return self._get_fallback_response()

        # Perform semantic search
        results = self.runbooks_store.similarity_search_with_score(
            error_pattern, k=self.top_k
        )

        if not results:
            return self._get_fallback_response()

        # Extract and format results
        top_chunks = []
        for doc, distance in results:
            top_chunks.append(
                {
                    "content": doc.page_content,
                    "relevance_score": self._distance_to_similarity(distance),
                    "source": doc.metadata.get("source", "unknown"),
                }
            )

        # Combine top chunks into structured response
        combined_content = "\n\n".join([chunk["content"] for chunk in top_chunks])

        return {
            "section_name": "Semantic Search Results",
            "query": error_pattern,
            "retrieved_chunks": top_chunks,
            "combined_context": combined_content,
            "retrieval_method": "vector_semantic_search",
            "top_k": len(top_chunks),
            "policy_reference": "POL-SRE-003 Troubleshooting Runbooks",
        }

    def get_failure_pattern(self, service_or_symptom: str) -> dict[str, Any]:
        """Retrieve failure patterns using semantic search.

        Args:
            service_or_symptom: Service name or symptom description

        Returns:
            Dictionary with failure pattern information
        """
        if self.patterns_store is None:
            return self._get_fallback_response()

        # Perform semantic search
        results = self.patterns_store.similarity_search_with_score(
            service_or_symptom, k=self.top_k
        )

        if not results:
            return self._get_fallback_response()

        # Extract and format results
        top_chunks = []
        for doc, distance in results:
            top_chunks.append(
                {
                    "content": doc.page_content,
                    "relevance_score": self._distance_to_similarity(distance),
                    "source": doc.metadata.get("source", "unknown"),
                }
            )

        combined_content = "\n\n".join([chunk["content"] for chunk in top_chunks])

        return {
            "section_name": "Failure Pattern Analysis",
            "query": service_or_symptom,
            "retrieved_chunks": top_chunks,
            "combined_context": combined_content,
            "retrieval_method": "vector_semantic_search",
            "top_k": len(top_chunks),
            "policy_reference": "POL-SRE-004 Known Failure Patterns",
        }

    def get_similar_incidents(self, symptoms: list[str]) -> list[dict[str, Any]]:
        """Retrieve similar incidents using semantic search.

        Args:
            symptoms: List of symptom descriptions

        Returns:
            List of similar incidents
        """
        if not symptoms or self.runbooks_store is None:
            return []

        # Combine symptoms into query
        query = " ".join(symptoms)

        # Search runbooks for incident references
        results = self.runbooks_store.similarity_search_with_score(query, k=10)

        incidents = []
        for doc, distance in results:
            content = doc.page_content

            # Extract incident IDs (INC-YYYY-MM-DD pattern)
            incident_pattern = r"INC-\d{4}-\d{2}-\d{2}"
            matches = re.findall(incident_pattern, content)

            for incident_id in matches:
                # Extract context around incident ID
                idx = content.find(incident_id)
                if idx != -1:
                    # Get surrounding context
                    start = max(0, idx - 50)
                    end = min(len(content), idx + 150)
                    context = content[start:end].strip()

                    incidents.append(
                        {
                            "incident_id": incident_id,
                            "description": context,
                            "relevance_score": self._distance_to_similarity(distance),
                        }
                    )

        # Deduplicate and sort by relevance
        seen_ids = set()
        unique_incidents = []
        for inc in sorted(incidents, key=lambda x: x["relevance_score"], reverse=True):
            if inc["incident_id"] not in seen_ids:
                seen_ids.add(inc["incident_id"])
                unique_incidents.append(inc)

        return unique_incidents[:5]  # Return top 5

    def get_error_code_guidance(self, error_code: str) -> dict[str, Any]:
        """Retrieve error code guidance using semantic search.

        This method provides compatibility with the RCAKnowledgeReader interface.

        Args:
            error_code: Error code or pattern to search for

        Returns:
            Dictionary with error code guidance
        """
        if not error_code or self.runbooks_store is None:
            return {
                "error_code": error_code,
                "guidance": "No guidance available",
                "retrieval_method": "fallback",
            }

        # Search for error code in runbooks
        results = self.runbooks_store.similarity_search_with_score(
            f"error code {error_code}", k=3
        )

        if not results:
            return {
                "error_code": error_code,
                "guidance": "No guidance found for this error code",
                "retrieval_method": "vector_semantic_search",
            }

        # Combine top results
        guidance_texts = []
        for doc, distance in results:
            if self._distance_to_similarity(distance) > 0.3:  # Relevance threshold
                guidance_texts.append(doc.page_content)

        return {
            "error_code": error_code,
            "guidance": (
                "\n\n".join(guidance_texts)
                if guidance_texts
                else "No specific guidance found"
            ),
            "retrieval_method": "vector_semantic_search",
            "sources_found": len(guidance_texts),
        }

    def get_all_error_patterns(self) -> list[str]:
        """Extract all error patterns from knowledge base.

        This method provides compatibility with the RCAKnowledgeReader interface.
        Note: This is less useful with vector RAG since we search semantically,
        but provided for backwards compatibility.

        Returns:
            List of error pattern strings
        """
        patterns = []

        # Load runbooks content
        if self.runbooks_path.exists():
            content = self.runbooks_path.read_text()

            # Extract error patterns using regex
            # Look for common error pattern formats
            error_patterns = [
                r"SQLSTATE\[\w+\]",  # SQL error codes
                r"HTTP [45]\d{2}",  # HTTP error codes
                r"ERROR: .+",  # Generic error messages
                r"Exception: .+",  # Exception messages
            ]

            for pattern in error_patterns:
                matches = re.findall(pattern, content)
                patterns.extend(matches)

        return list(set(patterns))[:50]  # Return unique patterns, max 50

    def search_by_semantic_query(
        self, query: str, source: str = "both"
    ) -> list[dict[str, Any]]:
        """General semantic search across knowledge base.

        Args:
            query: Natural language query
            source: "runbooks", "patterns", or "both"

        Returns:
            List of relevant chunks with scores
        """
        results = []

        if source in ("runbooks", "both") and self.runbooks_store:
            runbooks_results = self.runbooks_store.similarity_search_with_score(
                query, k=self.top_k
            )
            for doc, distance in runbooks_results:
                results.append(
                    {
                        "content": doc.page_content,
                        "relevance_score": self._distance_to_similarity(distance),
                        "source": "troubleshooting-runbooks.md",
                        "metadata": self._sanitize_for_json(doc.metadata),
                    }
                )

        if source in ("patterns", "both") and self.patterns_store:
            patterns_results = self.patterns_store.similarity_search_with_score(
                query, k=self.top_k
            )
            for doc, distance in patterns_results:
                results.append(
                    {
                        "content": doc.page_content,
                        "relevance_score": self._distance_to_similarity(distance),
                        "source": "known-failure-patterns.md",
                        "metadata": self._sanitize_for_json(doc.metadata),
                    }
                )

        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return results[: self.top_k]

    def _get_fallback_response(self) -> dict[str, Any]:
        """Provide fallback response when vector search unavailable."""
        return {
            "section_name": "Fallback Mode",
            "error": "Vector search unavailable",
            "combined_context": "Vector search system is initializing or unavailable. Please check logs.",
            "retrieval_method": "fallback",
            "policy_reference": "POL-SRE-003",
        }


__all__ = ["VectorRAGKnowledgeReader", "BedrockEmbeddings"]
