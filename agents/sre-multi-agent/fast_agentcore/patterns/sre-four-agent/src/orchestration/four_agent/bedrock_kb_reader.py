"""Bedrock Knowledge Base integration for SRE troubleshooting RAG.

This module provides semantic search capabilities using AWS-managed Bedrock Knowledge Base,
replacing the local FAISS vector store implementation.

Key features:
- Bedrock Agent Runtime API integration for retrieval
- Configuration loading from environment variables or config file
- Retry logic with exponential backoff for transient failures
- Response format conversion to maintain compatibility with existing agents
- Comprehensive error handling and fallback responses
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any
from threading import local

logger = logging.getLogger(__name__)

# Thread-local storage for KB retrieval tracking
_kb_tracking = local()

def get_kb_retrievals():
    """Get list of KB retrievals for current thread."""
    if not hasattr(_kb_tracking, 'retrievals'):
        _kb_tracking.retrievals = []
    return _kb_tracking.retrievals

def clear_kb_retrievals():
    """Clear KB retrieval tracking for current thread."""
    if hasattr(_kb_tracking, 'retrievals'):
        _kb_tracking.retrievals = []

def track_kb_retrieval(query: str, sources: list[str], result_count: int):
    """Track a KB retrieval for display to user."""
    if not hasattr(_kb_tracking, 'retrievals'):
        _kb_tracking.retrievals = []
    _kb_tracking.retrievals.append({
        'query': query,
        'sources': sources,
        'result_count': result_count
    })

try:
    import boto3
    from botocore.exceptions import ClientError, ReadTimeoutError

    BOTO3_AVAILABLE = True
except ImportError as e:
    BOTO3_AVAILABLE = False
    logger.warning(
        f"boto3 not available ({e}). "
        "Install with: pip install boto3"
    )


class BedrockKnowledgeBaseReader:
    """Bedrock Knowledge Base integration for SRE troubleshooting RAG.

    Provides semantic search capabilities using AWS-managed Bedrock Knowledge Base.
    Maintains compatibility with VectorRAGKnowledgeReader interface for seamless
    integration with existing RCA, Impact, and Mitigation agents.

    Attributes:
        knowledge_base_id: Bedrock KB ID (from env or config)
        region_name: AWS region for Bedrock services
        inference_model_arn: Model ARN for inference (Llama 3.3/3.7)
        top_k: Number of results to retrieve from semantic search
        client: boto3 bedrock-agent-runtime client
    """

    def __init__(
        self,
        knowledge_base_id: str | None = None,
        region_name: str = "us-west-2",
        inference_model_arn: str | None = None,
        top_k: int = 5,
    ):
        """Initialize Bedrock Knowledge Base reader.

        Args:
            knowledge_base_id: Bedrock KB ID (from env or config if not provided)
            region_name: AWS region for Bedrock services
            inference_model_arn: Model ARN for inference (Llama 3.3/3.7)
            top_k: Number of results to retrieve from semantic search

        Raises:
            ImportError: If boto3 is not available
            ValueError: If required configuration is missing after loading
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for Bedrock Knowledge Base integration. "
                "Install with: pip install boto3"
            )

        # Load configuration from environment variables or parameters
        self.knowledge_base_id = self._load_config_value(
            "BEDROCK_KB_ID", knowledge_base_id, required=True
        )
        self.region_name = self._load_config_value(
            "AWS_REGION", region_name, required=False
        )
        self.inference_model_arn = self._load_config_value(
            "BEDROCK_INFERENCE_MODEL",
            inference_model_arn,
            required=False,
            default="arn:aws:bedrock:us-west-2::foundation-model/meta.llama3-3-70b-instruct-v1:0",
        )
        self.top_k = int(os.getenv("BEDROCK_KB_TOP_K", str(top_k)))

        # Validate configuration
        self._validate_configuration()

        # Initialize boto3 client
        try:
            self.client = boto3.client(
                "bedrock-agent-runtime", region_name=self.region_name
            )
            logger.info(
                f"Initialized Bedrock Knowledge Base reader: "
                f"KB_ID={self.knowledge_base_id}, "
                f"Region={self.region_name}, "
                f"Top_K={self.top_k}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize bedrock-agent-runtime client: {e}")
            raise

    def _load_config_value(
        self,
        env_var: str,
        param_value: str | None,
        required: bool = False,
        default: str | None = None,
    ) -> str:
        """Load configuration value from environment variable or parameter.

        Priority order:
        1. Parameter value (if provided)
        2. Environment variable
        3. Default value (if provided)

        Args:
            env_var: Environment variable name
            param_value: Parameter value (takes precedence)
            required: Whether this configuration is required
            default: Default value if not found

        Returns:
            Configuration value

        Raises:
            ValueError: If required configuration is missing
        """
        # Check parameter first
        if param_value is not None:
            return param_value

        # Check environment variable
        env_value = os.getenv(env_var)
        if env_value:
            return env_value

        # Use default if provided
        if default is not None:
            logger.warning(
                f"Configuration {env_var} not found, using default: {default}"
            )
            return default

        # Raise error if required
        if required:
            raise ValueError(
                f"Required configuration {env_var} is missing. "
                f"Set environment variable or pass as parameter."
            )

        return ""

    def _validate_configuration(self):
        """Validate that all required configuration is present.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        errors = []

        if not self.knowledge_base_id:
            errors.append(
                "BEDROCK_KB_ID is required. Set environment variable or pass as parameter."
            )

        if not self.region_name:
            errors.append("AWS_REGION is required.")

        if self.top_k < 1:
            errors.append(f"top_k must be >= 1, got {self.top_k}")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Configuration validation passed")


    def _retrieve_from_kb(
        self,
        query: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
    ) -> dict[str, Any]:
        """Core retrieval method that calls bedrock_agent_runtime.retrieve().

        Implements retry logic with exponential backoff for timeouts and throttling.
        Handles various API errors including authentication, not found, and throttling.

        Args:
            query: Natural language query for semantic search
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Initial delay in seconds for exponential backoff (default: 1.0)
            max_delay: Maximum delay in seconds between retries (default: 10.0)

        Returns:
            Bedrock KB API response dictionary containing retrievalResults

        Raises:
            ValueError: If query is empty or invalid
            ClientError: For non-retryable API errors (auth, not found, validation)
            ReadTimeoutError: If all retries are exhausted for timeout errors
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Prepare retrieval configuration
        retrieval_config = {
            "vectorSearchConfiguration": {"numberOfResults": self.top_k}
        }

        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    f"Attempting Bedrock KB retrieve (attempt {attempt + 1}/{max_retries + 1}): "
                    f"query='{query[:50]}...', top_k={self.top_k}"
                )

                # Call Bedrock Agent Runtime retrieve API
                response = self.client.retrieve(
                    knowledgeBaseId=self.knowledge_base_id,
                    retrievalQuery={"text": query},
                    retrievalConfiguration=retrieval_config,
                )

                result_count = len(response.get('retrievalResults', []))
                logger.info(
                    f"Successfully retrieved {result_count} results "
                    f"from Knowledge Base (attempt {attempt + 1})"
                )
                
                # Extract sources for tracking
                sources = []
                for result in response.get('retrievalResults', [])[:5]:  # Track first 5 sources
                    location = result.get('location', {})
                    if 's3Location' in location:
                        uri = location['s3Location'].get('uri', 'unknown')
                        # Extract filename from S3 URI
                        filename = uri.split('/')[-1] if '/' in uri else uri
                        if filename not in sources:
                            sources.append(filename)
                
                # Track this retrieval for display to user
                track_kb_retrieval(query[:100], sources, result_count)
                
                if sources:
                    logger.info(f"KB Query: '{query[:80]}...' | Sources: {', '.join(sources)}")
                
                return response

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                error_message = e.response.get("Error", {}).get("Message", str(e))
                last_exception = e

                # Handle different error types
                if error_code == "ResourceNotFoundException":
                    logger.error(
                        f"Knowledge Base not found: {self.knowledge_base_id}. "
                        f"Verify the KB ID is correct and exists in region {self.region_name}. "
                        f"Error: {error_message}"
                    )
                    raise  # Don't retry for not found errors

                elif error_code in ["AccessDeniedException", "UnauthorizedException"]:
                    logger.error(
                        f"Authentication/Authorization failed for Knowledge Base {self.knowledge_base_id}. "
                        f"Check IAM role permissions for bedrock:Retrieve action. "
                        f"Error: {error_message}"
                    )
                    raise  # Don't retry for auth errors

                elif error_code == "ValidationException":
                    logger.error(
                        f"Invalid request parameters for Knowledge Base retrieve. "
                        f"Query: '{query[:100]}', Error: {error_message}"
                    )
                    raise  # Don't retry for validation errors

                elif error_code == "ThrottlingException":
                    if attempt < max_retries:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            f"Throttled by Bedrock KB API (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Retrying in {delay:.1f}s... Error: {error_message}"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(
                            f"Throttling persisted after {max_retries} retries. "
                            f"Consider implementing request rate limiting. Error: {error_message}"
                        )
                        raise

                elif error_code in ["ServiceUnavailableException", "InternalServerException"]:
                    if attempt < max_retries:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            f"Bedrock KB service unavailable (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Retrying in {delay:.1f}s... Error: {error_message}"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(
                            f"Service unavailable after {max_retries} retries. "
                            f"Error: {error_message}"
                        )
                        raise

                else:
                    # Unknown error - log and raise
                    logger.error(
                        f"Unexpected ClientError from Bedrock KB: "
                        f"Code={error_code}, Message={error_message}"
                    )
                    raise

            except ReadTimeoutError as e:
                last_exception = e
                if attempt < max_retries:
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        f"Network timeout during Bedrock KB retrieve (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay:.1f}s... Error: {e}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Network timeout persisted after {max_retries} retries. "
                        f"Check network connectivity and Bedrock service status. Error: {e}"
                    )
                    raise

            except Exception as e:
                # Catch-all for unexpected errors
                logger.error(
                    f"Unexpected error during Bedrock KB retrieve: {type(e).__name__}: {e}"
                )
                raise

        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry loop completed without success or exception")

    def _convert_bedrock_response(
        self,
        bedrock_response: dict[str, Any],
        query: str,
        policy_ref: str = "N/A",
    ) -> dict[str, Any]:
        """Convert Bedrock KB response to VectorRAGKnowledgeReader compatible format.

        Extracts text content and metadata from Bedrock KB results, maps confidence
        scores to relevance scores, and formats the response to match the expected
        dictionary structure for agent compatibility.

        Args:
            bedrock_response: Raw response from Bedrock Agent Runtime retrieve API
            query: Original query string used for retrieval
            policy_ref: Policy reference identifier (default: "N/A")

        Returns:
            Dictionary with VectorRAGKnowledgeReader compatible format:
            {
                "section_name": "Semantic Search Results",
                "query": str,
                "retrieved_chunks": [
                    {
                        "content": str,
                        "relevance_score": float (0-1),
                        "source": str (filename)
                    }
                ],
                "combined_context": str (concatenated chunks),
                "retrieval_method": "bedrock_knowledge_base",
                "top_k": int,
                "policy_reference": str
            }

        Note:
            - Bedrock confidence scores (0-1) are directly mapped to relevance scores
            - Higher scores indicate better relevance
            - Source filenames are extracted from S3 URIs in location metadata
        """
        retrieved_chunks = []

        # Extract and convert each retrieval result
        for result in bedrock_response.get("retrievalResults", []):
            # Extract text content from result
            content = result.get("content", {}).get("text", "")

            # Extract confidence score (0-1 range, higher is better)
            # This directly maps to relevance_score as both use 0-1 scale
            relevance_score = result.get("score", 0.0)

            # Extract source document name from S3 URI
            location = result.get("location", {})
            s3_location = location.get("s3Location", {})
            s3_uri = s3_location.get("uri", "")

            # Parse filename from S3 URI (e.g., s3://bucket/policies/doc.md -> doc.md)
            if s3_uri:
                source_file = Path(s3_uri).name
            else:
                source_file = "unknown"

            retrieved_chunks.append({
                "content": content,
                "relevance_score": relevance_score,
                "source": source_file,
            })

        # Concatenate all chunk content for combined context
        combined_context = "\n\n".join([
            chunk["content"] for chunk in retrieved_chunks if chunk["content"]
        ])

        # Return in VectorRAGKnowledgeReader compatible format
        return {
            "section_name": "Semantic Search Results",
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "combined_context": combined_context,
            "retrieval_method": "bedrock_knowledge_base",
            "top_k": len(retrieved_chunks),
            "policy_reference": policy_ref,
        }

    def get_troubleshooting_steps(self, error_pattern: str) -> dict[str, Any]:
        """Retrieve diagnostic steps using Bedrock KB semantic search.

        Args:
            error_pattern: Error message or symptom description

        Returns:
            Dictionary with troubleshooting information matching VectorRAGKnowledgeReader format

        Raises:
            ValueError: If error_pattern is empty or invalid
            ClientError: For non-retryable Bedrock KB API errors
            ReadTimeoutError: If all retries are exhausted for timeout errors
        """
        try:
            # Call core retrieval method with error pattern query
            bedrock_response = self._retrieve_from_kb(error_pattern)
            
            # Convert response to expected format with troubleshooting policy reference
            result = self._convert_bedrock_response(
                bedrock_response,
                query=error_pattern,
                policy_ref="POL-SRE-003 Troubleshooting Runbooks"
            )
            
            logger.info(
                f"Retrieved {result['top_k']} troubleshooting steps for error pattern: "
                f"'{error_pattern[:50]}...'"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Failed to retrieve troubleshooting steps for error pattern "
                f"'{error_pattern[:50]}...': {type(e).__name__}: {e}"
            )
            raise

    def get_failure_pattern(self, service_or_symptom: str) -> dict[str, Any]:
        """Retrieve failure patterns using Bedrock KB semantic search.

        Args:
            service_or_symptom: Service name or symptom description

        Returns:
            Dictionary with failure pattern information matching VectorRAGKnowledgeReader format

        Raises:
            ValueError: If service_or_symptom is empty or invalid
            ClientError: For non-retryable Bedrock KB API errors
            ReadTimeoutError: If all retries are exhausted for timeout errors
        """
        try:
            # Call core retrieval method with service/symptom query
            bedrock_response = self._retrieve_from_kb(service_or_symptom)
            
            # Convert response to expected format with failure pattern policy reference
            result = self._convert_bedrock_response(
                bedrock_response,
                query=service_or_symptom,
                policy_ref="POL-SRE-001 Known Failure Patterns"
            )
            
            logger.info(
                f"Retrieved {result['top_k']} failure patterns for service/symptom: "
                f"'{service_or_symptom[:50]}...'"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Failed to retrieve failure patterns for service/symptom "
                f"'{service_or_symptom[:50]}...': {type(e).__name__}: {e}"
            )
            raise

    def get_similar_incidents(self, symptoms: list[str]) -> list[dict[str, Any]]:
        """Retrieve similar incidents using Bedrock KB semantic search.

        Searches for past incidents that match the provided symptoms by combining
        symptom descriptions into a semantic query and extracting incident references
        from the retrieved knowledge base content.

        Args:
            symptoms: List of symptom descriptions (e.g., ["connection timeout", "high latency"])

        Returns:
            List of similar incidents, each containing:
            - incident_id: Incident identifier (e.g., "INC-2024-01-15")
            - description: Context around the incident reference
            - relevance_score: Similarity score (0-1, higher is better)

        Raises:
            ValueError: If symptoms list is empty or contains only empty strings
            ClientError: For non-retryable Bedrock KB API errors
            ReadTimeoutError: If all retries are exhausted for timeout errors
        """
        try:
            # Validate symptoms list
            if not symptoms:
                logger.warning("Empty symptoms list provided to get_similar_incidents")
                return []
            
            # Filter out empty strings and combine symptoms into query
            valid_symptoms = [s.strip() for s in symptoms if s and s.strip()]
            if not valid_symptoms:
                logger.warning("No valid symptoms after filtering empty strings")
                return []
            
            query = " ".join(valid_symptoms)
            
            # Call core retrieval method with combined symptoms query
            bedrock_response = self._retrieve_from_kb(query)
            
            # Extract incidents from retrieved content
            incidents = []
            import re
            
            # Incident ID pattern: INC-YYYY-MM-DD
            incident_pattern = r"INC-\d{4}-\d{2}-\d{2}"
            
            for result in bedrock_response.get("retrievalResults", []):
                content = result.get("content", {}).get("text", "")
                relevance_score = result.get("score", 0.0)
                
                # Find all incident IDs in this chunk
                matches = re.findall(incident_pattern, content)
                
                for incident_id in matches:
                    # Extract context around incident ID
                    idx = content.find(incident_id)
                    if idx != -1:
                        # Get surrounding context (50 chars before, 150 chars after)
                        start = max(0, idx - 50)
                        end = min(len(content), idx + 150)
                        context = content[start:end].strip()
                        
                        incidents.append({
                            "incident_id": incident_id,
                            "description": context,
                            "relevance_score": relevance_score,
                        })
            
            # Deduplicate by incident_id and sort by relevance score
            seen_ids = set()
            unique_incidents = []
            for inc in sorted(incidents, key=lambda x: x["relevance_score"], reverse=True):
                if inc["incident_id"] not in seen_ids:
                    seen_ids.add(inc["incident_id"])
                    unique_incidents.append(inc)
            
            # Return top 5 most relevant incidents
            top_incidents = unique_incidents[:5]
            
            logger.info(
                f"Retrieved {len(top_incidents)} similar incidents for symptoms: "
                f"'{query[:50]}...'"
            )
            
            return top_incidents
            
        except Exception as e:
            logger.error(
                f"Failed to retrieve similar incidents for symptoms "
                f"'{' '.join(symptoms[:3])}...': {type(e).__name__}: {e}"
            )
            raise

    def get_error_code_guidance(self, error_code: str) -> dict[str, Any]:
        """Retrieve error code guidance using Bedrock KB semantic search.

        Searches the knowledge base for guidance related to a specific error code,
        formatting the query to optimize semantic search results.

        Args:
            error_code: Error code or pattern to search for (e.g., "SQLSTATE[08006]", "HTTP 503")

        Returns:
            Dictionary with error code guidance matching VectorRAGKnowledgeReader format

        Raises:
            ValueError: If error_code is empty or invalid
            ClientError: For non-retryable Bedrock KB API errors
            ReadTimeoutError: If all retries are exhausted for timeout errors
        """
        try:
            # Format query with error code for better semantic search
            query = f"error code {error_code}"
            
            # Call core retrieval method with formatted query
            bedrock_response = self._retrieve_from_kb(query)
            
            # Convert response to expected format with error code policy reference
            result = self._convert_bedrock_response(
                bedrock_response,
                query=query,
                policy_ref="POL-SRE-002 Error Code Reference"
            )
            
            logger.info(
                f"Retrieved {result['top_k']} error code guidance entries for: "
                f"'{error_code}'"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Failed to retrieve error code guidance for '{error_code}': "
                f"{type(e).__name__}: {e}"
            )
            raise

    def search_by_semantic_query(
        self, query: str, source: str = "both"
    ) -> list[dict[str, Any]]:
        """General semantic search using Bedrock KB.

        This is a general-purpose semantic search method that provides compatibility
        with VectorRAGKnowledgeReader interface. Unlike the specialized methods
        (get_troubleshooting_steps, get_failure_pattern, etc.), this returns a
        simplified list format rather than the full dictionary structure.

        Args:
            query: Natural language query for semantic search
            source: Ignored (for compatibility with VectorRAGKnowledgeReader).
                   Bedrock KB searches across all indexed documents.

        Returns:
            List of relevant chunks with scores. Each chunk is a dictionary with:
            - content: Text content of the chunk
            - relevance_score: Similarity score (0-1, higher is better)
            - source: Source document filename

        Raises:
            ValueError: If query is empty or invalid
            ClientError: For non-retryable Bedrock KB API errors
            ReadTimeoutError: If all retries are exhausted for timeout errors

        Note:
            The 'source' parameter is accepted for interface compatibility but
            is ignored. Bedrock KB searches across all indexed documents in the
            knowledge base, regardless of the source parameter value.
        """
        try:
            # Call core retrieval method with query
            bedrock_response = self._retrieve_from_kb(query)
            
            # Extract chunks from response in simplified list format
            chunks = []
            for result in bedrock_response.get("retrievalResults", []):
                # Extract text content from result
                content = result.get("content", {}).get("text", "")
                
                # Extract confidence score (0-1 range, higher is better)
                relevance_score = result.get("score", 0.0)
                
                # Extract source document name from S3 URI
                location = result.get("location", {})
                s3_location = location.get("s3Location", {})
                s3_uri = s3_location.get("uri", "")
                
                # Parse filename from S3 URI (e.g., s3://bucket/policies/doc.md -> doc.md)
                if s3_uri:
                    source_file = Path(s3_uri).name
                else:
                    source_file = "unknown"
                
                chunks.append({
                    "content": content,
                    "relevance_score": relevance_score,
                    "source": source_file,
                })
            
            logger.info(
                f"Retrieved {len(chunks)} chunks for semantic query: "
                f"'{query[:50]}...'"
            )
            
            return chunks
            
        except Exception as e:
            logger.error(
                f"Failed to execute semantic search for query "
                f"'{query[:50]}...': {type(e).__name__}: {e}"
            )
            raise


__all__ = ["BedrockKnowledgeBaseReader"]
