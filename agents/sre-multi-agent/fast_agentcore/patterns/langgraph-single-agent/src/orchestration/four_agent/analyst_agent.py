"""Analyst agent for unstructured log analysis and LLM-based anomaly detection."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..observability import emit_event, wrap_payload

# CloudWatch observability removed for simplified demo
CLOUDWATCH_AVAILABLE = False
# Evidence generator removed for simplified demo
from .llm import BaseLLMAgent
from .schema import AgentRole, EvidenceReference, MessageType, PayloadModel
from .settings import get_default_model
from .state import IncidentState


@dataclass(frozen=True)
class AnalystAgentConfig:
    """Configuration for analyst agent anomaly detection thresholds."""

    # LLM-based anomaly detection thresholds
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.6
    orchestration_trigger_threshold: float = 0.60  # Lowered to trigger with current logs

    # Log analysis parameters
    max_log_lines_analyzed: int = 500
    include_temporal_context: bool = True
    include_service_correlation: bool = True

    # Autonomous triggering controls
    enable_autonomous_triggering: bool = True
    rate_limit_minutes: int = 15
    max_triggers_per_hour: int = 4


@dataclass
class DetectedAnomaly:
    """Represents an anomaly detected by LLM analysis."""

    pattern: str
    confidence: float
    severity: str
    affected_services: List[str]
    evidence_snippet: str
    timestamp_range: str


@dataclass
class AnalysisResult:
    """Result of log analysis with anomaly detection."""

    anomalies: List[DetectedAnomaly]
    overall_confidence: float
    severity_assessment: str
    log_summary: str


class AnalystAgent(BaseLLMAgent):
    """Derives incident insights via LLM-based unstructured log analysis and anomaly detection."""

    name = (
        AgentRole.ANALYST.value
    )  # Using ANALYST alias (resolves to "Signals" for compatibility)

    def __init__(
        self,
        config: AnalystAgentConfig | None = None,
        *,
        unstructured_mode: bool = True,
        orchestrator: Optional[
            Any
        ] = None,  # PhaseTwoOrchestrator for direct triggering
        real_time_mode: bool = False,
        llm=None,
        model: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 2000,
        stream_updates: bool = True,
    ) -> None:
        self._config = config or AnalystAgentConfig()
        self.unstructured_mode = unstructured_mode
        self.orchestrator = orchestrator
        self.real_time_mode = real_time_mode

        # Tracking for autonomous triggering
        self._last_trigger_time: Optional[float] = None
        self._triggers_this_hour: int = 0
        self._trigger_reset_time: Optional[float] = None

        # CloudWatch observability
        self._observability = None
        if CLOUDWATCH_AVAILABLE:
            try:
                self._observability = get_analyst_observability()
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Failed to initialize CloudWatch observability: {e}")

        # Performance tracking
        self._analysis_start_time: Optional[float] = None

        # Initialize real-time scenario loader if in real-time mode
        self.scenario_loader = None
        if real_time_mode:
            try:
                from src.config import get_opensearch_config
                from src.integrations import (
                    AnomalyDetector,
                    OpenSearchClient,
                    RealTimeScenarioLoader,
                )

                opensearch_config = get_opensearch_config()
                opensearch_client = OpenSearchClient(
                    endpoint=opensearch_config.endpoint,
                    region=opensearch_config.region,
                    timeout=opensearch_config.timeout,
                    max_retries=opensearch_config.max_retries,
                )

                anomaly_detector = AnomalyDetector()

                self.scenario_loader = RealTimeScenarioLoader(
                    opensearch_client=opensearch_client,
                    anomaly_detector=anomaly_detector,
                )
            except Exception as e:
                # Fall back to demo mode if real-time setup fails
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Failed to initialize real-time mode, falling back to demo mode: {e}"
                )
                self.real_time_mode = False

        super().__init__(
            role=AgentRole.ANALYST.value,
            llm=llm,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stream_updates=stream_updates,
        )

    # ------------------------------------------------------------------
    # Main Analysis Methods
    # ------------------------------------------------------------------
    async def analyze_logs(
        self, log_text: str, metadata: Dict[str, Any]
    ) -> AnalysisResult:
        """Main method for unstructured log analysis using LLM."""

        # Build prompt for LLM analysis
        system_prompt = """You are an expert SRE analyst. Analyze logs for anomalies and incidents.

CRITICAL: You MUST respond with valid JSON only. No other text.

JSON format:
{
    "anomalies": [
        {
            "pattern": "Description of the anomaly pattern",
            "confidence": 0.85,
            "severity": "MEDIUM",
            "affected_services": ["service-name"],
            "evidence_snippet": "Example log entry showing the issue",
            "timestamp_range": "Time period when observed"
        }
    ],
    "overall_confidence": 0.5,
    "severity_assessment": "LOW",
    "log_summary": "System operating normally"
}

IMPORTANT: Each anomaly in the "anomalies" array MUST be a JSON object with the fields shown above.
DO NOT use plain strings for anomalies - always use the structured format.

Use severity levels: LOW, MEDIUM, HIGH, CRITICAL
Confidence values: 0.0 to 1.0

DO NOT include recommended_action - only detect and report anomalies."""

        user_prompt = f"""Analyze these logs from a {metadata.get("window_size_minutes", "unknown")} minute window:

Window: {metadata.get("window_start", "unknown")}
Log Count: {metadata.get("log_count", "unknown")}

Logs:
{log_text}

Additional Context:
{json.dumps(metadata.get("metrics", {}), indent=2)}

Identify any anomalies, errors, or concerning patterns. Focus on:
- Error patterns and frequencies
- Service correlation issues
- Performance degradation
- Security concerns
- Unusual traffic patterns"""

        try:
            # Use the LLM runner from parent class
            runner = self._ensure_runner()

            # Create LLM request
            from .llm import LLMRequest

            request = LLMRequest(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.25,
                max_tokens=self._max_tokens,
                stream=False,
            )

            # Get LLM response
            result = await runner.run(request)
            response = result.text

            # Parse JSON response - handle markdown code blocks
            if not response or not response.strip():
                raise ValueError("LLM returned empty response")

            # Clean the response - remove markdown code blocks and extra tokens
            clean_response = response.strip()

            # Remove markdown code block markers
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]  # Remove ```json
            if clean_response.startswith("```"):
                clean_response = clean_response[3:]  # Remove ```

            # Remove ending markers and model tokens
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]  # Remove ending ```

            # Remove common LLM end tokens
            for end_token in ["<|eot_id|>", "<|end|>", "<|endoftext|>", "```"]:
                if end_token in clean_response:
                    clean_response = clean_response.split(end_token)[0]

            clean_response = clean_response.strip()

            # Find the last complete JSON object by finding the last }
            last_brace = clean_response.rfind("}")
            if last_brace != -1:
                clean_response = clean_response[: last_brace + 1]

            # Show cleaned JSON output
            print("📋 Cleaned JSON Analysis:")
            print(clean_response)
            print()

            result_data = json.loads(clean_response)

            # Convert to AnalysisResult - handle flexible field names
            anomalies = []
            for anomaly_data in result_data.get("anomalies", []):
                try:
                    # Handle both dictionary and string anomalies
                    if isinstance(anomaly_data, str):
                        # LLM returned a plain string instead of structured object
                        # Create a structured anomaly from the string
                        anomaly = self._parse_string_anomaly(anomaly_data)
                    elif isinstance(anomaly_data, dict):
                        # Expected case: structured dictionary
                        anomaly = DetectedAnomaly(
                            pattern=anomaly_data.get(
                                "pattern",
                                anomaly_data.get("description", "Unknown pattern"),
                            ),
                            confidence=float(anomaly_data.get("confidence", 0.7)),
                            severity=anomaly_data.get("severity", "MEDIUM"),
                            affected_services=anomaly_data.get(
                                "affected_services",
                                [anomaly_data.get("service", "unknown")],
                            ),
                            evidence_snippet=anomaly_data.get(
                                "evidence_snippet",
                                anomaly_data.get("example_log", "No evidence"),
                            ),
                            timestamp_range=anomaly_data.get(
                                "timestamp_range", "Unknown time range"
                            ),
                        )
                    else:
                        # Unexpected type - skip
                        print(f"⚠️ Unexpected anomaly type: {type(anomaly_data)} - {anomaly_data}")
                        continue
                    
                    anomalies.append(anomaly)
                except Exception as e:
                    print(f"⚠️ Could not parse anomaly: {anomaly_data} - {e}")
                    continue

            return AnalysisResult(
                anomalies=anomalies,
                overall_confidence=result_data.get("overall_confidence", 0.0),
                severity_assessment=result_data.get("severity_assessment", "UNKNOWN"),
                log_summary=result_data.get("log_summary", "No issues detected"),
            )

        except json.JSONDecodeError as e:
            # Fallback if LLM doesn't return valid JSON - treat as plain text analysis
            print(f"⚠️ JSON parsing failed, treating as plain text analysis")

            # Extract basic insights from plain text response
            confidence = 0.3 if response and len(response) > 50 else 0.1
            severity = (
                "MEDIUM"
                if any(
                    word in response.lower()
                    for word in ["error", "fail", "critical", "alert"]
                )
                else "LOW"
            )

            return AnalysisResult(
                anomalies=[],
                overall_confidence=confidence,
                severity_assessment=severity,
                log_summary=response[:200] + "..." if len(response) > 200 else response,
            )
        except Exception as e:
            # General error handling with more details
            import traceback

            error_details = traceback.format_exc()
            print(f"LLM Analysis Error: {str(e)}")
            print(f"Full traceback: {error_details}")

            return AnalysisResult(
                anomalies=[],
                overall_confidence=0.0,
                severity_assessment="ERROR",
                log_summary="Unable to analyze logs due to error",
            )

    async def detect_anomalies(self, log_text: str) -> List[DetectedAnomaly]:
        """LLM-based anomaly detection in logs."""
        # This would contain the core anomaly detection logic
        raise NotImplementedError(
            "Direct anomaly detection not yet implemented - use handle() method"
        )

    def _parse_string_anomaly(self, anomaly_str: str) -> DetectedAnomaly:
        """Parse a plain string anomaly into a structured DetectedAnomaly object.
        
        This is a fallback for when the LLM returns anomalies as strings instead of
        structured JSON objects. We extract what information we can from the text.
        """
        # Default values
        confidence = 0.7
        severity = "MEDIUM"
        affected_services = ["unknown"]
        
        # Try to infer severity from keywords
        anomaly_lower = anomaly_str.lower()
        if any(word in anomaly_lower for word in ["critical", "severe", "fatal", "emergency"]):
            severity = "HIGH"
            confidence = 0.85
        elif any(word in anomaly_lower for word in ["error", "fail", "timeout", "exception"]):
            severity = "MEDIUM"
            confidence = 0.75
        elif any(word in anomaly_lower for word in ["warn", "warning", "slow", "degraded"]):
            severity = "MEDIUM"
            confidence = 0.65
        else:
            severity = "LOW"
            confidence = 0.6
        
        # Try to extract service names (common patterns)
        import re
        service_patterns = [
            r'(\w+)-service',
            r'(\w+)-api',
            r'service[:\s]+(\w+)',
            r'from\s+(\w+)\s+service',
        ]
        
        found_services = []
        for pattern in service_patterns:
            matches = re.findall(pattern, anomaly_lower)
            found_services.extend(matches)
        
        if found_services:
            affected_services = list(set(found_services))[:3]  # Limit to 3 services
        
        # Use the string as both pattern and evidence
        pattern = anomaly_str[:200] + "..." if len(anomaly_str) > 200 else anomaly_str
        evidence = anomaly_str[:300] + "..." if len(anomaly_str) > 300 else anomaly_str
        
        return DetectedAnomaly(
            pattern=pattern,
            confidence=confidence,
            severity=severity,
            affected_services=affected_services,
            evidence_snippet=evidence,
            timestamp_range="Unknown time range",
        )

    async def trigger_incident_if_needed(
        self, analysis: AnalysisResult, scenario=None
    ) -> bool:
        """Trigger orchestration for high-confidence anomalies."""
        # Autonomous triggering disabled in simplified version
        # This functionality was part of the concurrent orchestration system
        return False

        # Check if we should trigger based on confidence and rate limiting
        if analysis.overall_confidence < self._config.orchestration_trigger_threshold:
            return False

        if not self._should_trigger_now():
            return False

        # Create incident through MultiIncidentCoordinator if available
        try:
            # Check if orchestrator is a MultiIncidentCoordinator
            if hasattr(self.orchestrator, "start_incident"):
                if scenario is None:
                    # Create a synthetic scenario from analysis
                    scenario = self._create_scenario_from_analysis(analysis)

                # Start incident through coordinator
                incident_id = await self.orchestrator.start_incident(
                    scenario=scenario,
                    source="analyst_agent_autonomous",
                    priority=self._determine_priority_from_analysis(analysis),
                )

                self._record_trigger()

                from ..observability import emit_event, wrap_payload

                emit_event(
                    "agent.analyst",
                    "autonomous_incident_triggered",
                    wrap_payload(
                        incident_id=incident_id,
                        confidence=analysis.overall_confidence,
                        severity=analysis.severity_assessment,
                        trigger_source="analyst_agent",
                    ),
                )

                return True
            else:
                # Fall back to legacy single orchestrator mode
                # This would be the original PhaseTwoOrchestrator
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    "Legacy orchestrator detected - multi-incident coordination not available. "
                    "Consider upgrading to MultiIncidentCoordinator for concurrent processing."
                )
                self._record_trigger()
                return False

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to trigger autonomous incident: {e}")
            return False

    # ------------------------------------------------------------------
    # BaseLLMAgent hooks
    # ------------------------------------------------------------------
    def _system_prompt(self, incoming, state) -> str:
        base_prompt = (
            "You are an expert SRE Analyst specializing in unstructured log analysis and anomaly detection. "
            "Your job is to analyze raw log data from production systems and identify potential incidents, "
            "their severity, and root cause hypotheses. Focus on error patterns, unusual frequencies, "
            "service correlations, and temporal anomalies. Use your expertise to detect issues that "
            "statistical methods might miss. Always respond with JSON matching the requested schema."
        )

        if self.real_time_mode:
            real_time_context = (
                "\n\nREAL-TIME MODE: You are analyzing live production logs from streaming data sources. "
                "This data represents current system state and may indicate ongoing incidents. "
                "Focus on the unstructured log analysis and incident narrative provided, especially "
                "looking for patterns that suggest system degradation or failures."
            )
            return base_prompt + real_time_context

        if self.unstructured_mode:
            unstructured_context = (
                "\n\nUNSTRUCTURED MODE: You are analyzing raw log text and incident narratives rather "
                "than structured metrics. Look for patterns in log messages, error frequencies, "
                "service interactions, and temporal correlations. Detect anomalies through "
                "contextual understanding rather than statistical thresholds."
            )
            return base_prompt + unstructured_context

        return base_prompt

    def _build_user_prompt(self, incoming, state) -> str:
        # Start timing analysis
        self._analysis_start_time = time.time()

        log_analysis, incident_narrative, context_data = (
            self._analyze_logs_from_incoming(incoming)
        )
        prior = incoming.payload.details.get("prior_messages", [])

        context = {
            "incident_id": incoming.incident_id,
            "severity": getattr(state.severity, "value", str(state.severity)),
            "log_analysis": log_analysis,
            "incident_narrative": incident_narrative,
            "context_data": context_data,
            "prior_messages": prior,
        }

        instructions = {
            "summary": "Concise paragraph describing what the log analysis reveals about system health.",
            "details": {
                "severity_score": "Number between 0 and 1 representing incident urgency based on log patterns.",
                "anomaly_confidence": "Number between 0 and 1 indicating confidence in anomaly detection.",
                "detected_patterns": {
                    "error_patterns": "List of error patterns found in logs",
                    "frequency_anomalies": "Unusual frequencies or spikes detected",
                    "service_correlations": "Services showing correlated issues",
                    "temporal_patterns": "Time-based patterns or progressions",
                },
                "severity_indicators": "List of specific log evidence supporting severity assessment.",
                "initial_hypothesis": "Primary hypothesis about root cause based on log analysis.",
                "recommended_action": "Immediate recommended response based on analysis.",
            },
            "evidence": "List of supporting evidence objects with title/href/summary from log analysis.",
        }

        return (
            "Analyze the unstructured log data and incident narrative to assess system health and detect anomalies. "
            "Use your expertise to identify patterns, correlations, and issues that indicate potential incidents. "
            "Respond with JSON only.\n"
            "Context:```json\n"
            f"{json.dumps(context, indent=2)}\n```\n"
            "Required JSON schema:```json\n"
            f"{json.dumps(instructions, indent=2)}\n```"
        )

    def _message_type(
        self, parsed: Mapping[str, object], incoming, state
    ) -> MessageType:
        return MessageType.OPEN

    def _build_payload(
        self, parsed: Mapping[str, object], incoming, state
    ) -> PayloadModel:
        summary = parsed.get("summary")
        details = parsed.get("details") or {}

        # Generate simple evidence references from log analysis (simplified demo)
        evidence = None
        try:
            log_analysis, incident_narrative, context_data = (
                self._analyze_logs_from_incoming(incoming)
            )

            # Create simple evidence references without external file generation
            evidence = [
                EvidenceReference(
                    title="Log Analysis Summary",
                    summary=(
                        log_analysis[:200] + "..."
                        if len(log_analysis) > 200
                        else log_analysis
                    ),
                ),
                EvidenceReference(
                    title="Incident Narrative",
                    summary=(
                        incident_narrative[:200] + "..."
                        if len(incident_narrative) > 200
                        else incident_narrative
                    ),
                ),
            ]
        except Exception as e:
            # Gracefully fall back to empty evidence if generation fails
            evidence = None

        return PayloadModel(
            summary=str(summary) if summary is not None else None,
            details=dict(details),
            evidence=evidence or None,
        )

    def _after_message(
        self,
        message,
        parsed: Mapping[str, object],
        incoming,
        state: IncidentState,
        *,
        used_fallback: bool,
        raw_text: str | None,
    ) -> None:
        details = message.payload.details
        anomaly_confidence = details.get("anomaly_confidence", 0.0)
        severity_score = details.get("severity_score", 0.0)

        # Calculate analysis duration if tracking was started
        analysis_duration_ms = 0.0
        if self._analysis_start_time:
            analysis_duration_ms = (time.time() - self._analysis_start_time) * 1000
            self._analysis_start_time = None  # Reset

        # Emit CloudWatch metrics
        if self._observability:
            # Set incident context
            self._observability.set_trace_context(incident_id=message.incident_id)

            # Emit anomaly detection metrics
            anomaly_detected = 1 if anomaly_confidence > 0.5 else 0
            self._observability.emit_anomaly_metrics(
                anomalies_detected=anomaly_detected,
                confidence_avg=anomaly_confidence,
                detection_time_ms=(
                    analysis_duration_ms if analysis_duration_ms > 0 else 1000.0
                ),
            )

            # Log agent execution
            self._observability.log_agent_execution(
                agent_name="AnalystAgent",
                stage="log_analysis",
                success=not used_fallback,
                duration_ms=analysis_duration_ms,
                error_message=None if not used_fallback else "Used LLM fallback",
            )

            # Log anomaly detection if significant confidence
            if anomaly_confidence > 0.6:
                detected_patterns = details.get("detected_patterns", {})
                affected_services = detected_patterns.get("service_correlations", [])

                self._observability.log_anomaly_detected(
                    anomaly_type=(
                        detected_patterns.get("error_patterns", ["unknown"])[0]
                        if detected_patterns.get("error_patterns")
                        else "unknown"
                    ),
                    confidence=anomaly_confidence,
                    affected_services=(
                        affected_services if isinstance(affected_services, list) else []
                    ),
                    evidence=str(details.get("initial_hypothesis", "")),
                )

        # Emit original observability event
        emit_event(
            "agent.analyst",
            "agent_completed",
            wrap_payload(
                incident_id=message.incident_id,
                severity=message.severity.value if message.severity else None,
                severity_score=severity_score,
                anomaly_confidence=anomaly_confidence,
                unstructured_mode=self.unstructured_mode,
                used_fallback=used_fallback,
                analysis_duration_ms=analysis_duration_ms,
            ),
        )

        # Check if we should trigger autonomous orchestration
        if (
            self._config.enable_autonomous_triggering
            and anomaly_confidence >= self._config.orchestration_trigger_threshold
            and self._should_trigger_now()
        ):
            try:
                # Create analysis result from message details
                analysis_result = AnalysisResult(
                    anomalies=[],  # Would be populated from detected_patterns
                    overall_confidence=anomaly_confidence,
                    severity_assessment=details.get("severity_score", 0.5),
                    log_summary=str(message.payload.summary or ""),
                )

                # Schedule async triggering in the background since _after_message is not async
                import asyncio

                async def trigger_async():
                    try:
                        incident_triggered = await self.trigger_incident_if_needed(
                            analysis_result
                        )

                        import logging

                        logger = logging.getLogger(__name__)

                        if incident_triggered:
                            logger.info(
                                f"Autonomous incident triggered (confidence={anomaly_confidence:.2f}) "
                                f"for analysis of {message.incident_id}."
                            )
                        else:
                            logger.info(
                                f"High-confidence anomaly detected (confidence={anomaly_confidence:.2f}) "
                                f"for incident {message.incident_id} but autonomous triggering failed or not configured."
                            )
                    except Exception as e:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.warning(f"Failed to trigger autonomous incident: {e}")

                # Create task to run async triggering in background
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If event loop is running, schedule the task
                        asyncio.create_task(trigger_async())
                    else:
                        # If no loop is running, just log that we would trigger
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.info(
                            f"High-confidence anomaly detected (confidence={anomaly_confidence:.2f}) "
                            f"for incident {message.incident_id}. Autonomous triggering requires async context."
                        )
                except RuntimeError:
                    # No event loop available, log the detection
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"High-confidence anomaly detected (confidence={anomaly_confidence:.2f}) "
                        f"for incident {message.incident_id}. No event loop for autonomous triggering."
                    )
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to trigger autonomous orchestration: {e}")

    # ------------------------------------------------------------------
    # Log Analysis Methods (replacing metrics extraction)
    # ------------------------------------------------------------------
    def _analyze_logs_from_incoming(self, incoming) -> Tuple[str, str, Dict[str, Any]]:
        """Analyze unstructured logs from incoming data - handles both demo and real-time modes."""
        if self.real_time_mode:
            return self._extract_realtime_logs(incoming)
        return self._extract_demo_logs(incoming)

    def _extract_demo_logs(self, incoming) -> Tuple[str, str, Dict[str, Any]]:
        """Extract unstructured log data from demo scenario data."""
        monitoring = incoming.payload.details.get("monitoring", {})

        # Primary unstructured data sources
        log_analysis = monitoring.get("log_analysis", "")
        incident_narrative = monitoring.get("incident_narrative", "")

        # Additional context for log analysis
        additional = incoming.payload.details.get("additional_sources", {})
        context_data = {
            "incident_details": additional.get("incident_details", []),
            "affected_services": additional.get("affected_services", []),
            "timeline": additional.get("timeline", {}),
            "data_source": "kafka_demo",
        }

        return log_analysis, incident_narrative, context_data

    def _extract_realtime_logs(self, incoming) -> Tuple[str, str, Dict[str, Any]]:
        """Extract unstructured log data from real-time OpenSearch data."""
        # Get the base log data like demo mode
        log_analysis, incident_narrative, context_data = self._extract_demo_logs(
            incoming
        )

        # Enhance with real-time context if available
        real_time_context = incoming.payload.details.get("real_time_context", {})
        if real_time_context:
            # Add real-time anomaly context
            context_data.update(
                {
                    "anomaly_confidence": real_time_context.get(
                        "confidence_score", 0.0
                    ),
                    "anomaly_type": real_time_context.get("anomaly_type", "unknown"),
                    "baseline_deviation": real_time_context.get(
                        "baseline_deviation", 0.0
                    ),
                    "detection_timestamp": real_time_context.get("detection_time", ""),
                    "affected_services": real_time_context.get("affected_services", []),
                    "data_source": "opensearch_realtime",
                }
            )

        return log_analysis, incident_narrative, context_data

    def _determine_scenario_from_logs(
        self, details: Dict[str, Any], state: IncidentState
    ) -> str:
        """Determine scenario type from log analysis for evidence generation."""
        # Try to infer from detected patterns
        detected_patterns = details.get("detected_patterns", {})
        if detected_patterns:
            error_patterns = detected_patterns.get("error_patterns", [])
            if any("database" in pattern.lower() for pattern in error_patterns):
                return "database"
            elif any("payment" in pattern.lower() for pattern in error_patterns):
                return "payment"
            elif any(
                "traffic" in pattern.lower() or "overload" in pattern.lower()
                for pattern in error_patterns
            ):
                return "traffic"

        # Fallback to severity-based mapping
        severity_mapping = {"SEV-1": "critical", "SEV-2": "moderate", "SEV-3": "minor"}
        severity_str = getattr(state.severity, "value", str(state.severity))
        return severity_mapping.get(severity_str, "unknown")

    # ------------------------------------------------------------------
    # Autonomous Triggering Helpers
    # ------------------------------------------------------------------
    def _should_trigger_now(self) -> bool:
        """Check if we should trigger orchestration based on rate limiting."""
        import time

        current_time = time.time()

        # Reset hourly trigger count if needed
        if (
            self._trigger_reset_time is None
            or current_time - self._trigger_reset_time >= 3600
        ):  # 1 hour
            self._triggers_this_hour = 0
            self._trigger_reset_time = current_time

        # Check hourly limit
        if self._triggers_this_hour >= self._config.max_triggers_per_hour:
            return False

        # Check minimum interval
        if (
            self._last_trigger_time is not None
            and current_time - self._last_trigger_time
            < self._config.rate_limit_minutes * 60
        ):
            return False

        return True

    def _record_trigger(self) -> None:
        """Record that we triggered orchestration for rate limiting."""
        import time

        self._last_trigger_time = time.time()
        self._triggers_this_hour += 1

    def _create_scenario_from_analysis(
        self, analysis: AnalysisResult
    ) -> "ScenarioSnapshot":
        """Create a synthetic scenario snapshot from analysis results."""
        import uuid
        from datetime import timedelta

        from ...data_ingestion import IncidentWindow
        from .scenario_loader import ScenarioMetadata, ScenarioSnapshot
        from .schema import Severity

        # Generate synthetic incident ID
        incident_id = f"auto_{uuid.uuid4().hex[:8]}"

        # Map severity assessment to enum
        severity_mapping = {
            "critical": Severity.SEV_1,
            "high": Severity.SEV_1,
            "medium": Severity.SEV_2,
            "moderate": Severity.SEV_2,
            "low": Severity.SEV_3,
            "minor": Severity.SEV_3,
        }

        # Determine severity from analysis
        if isinstance(analysis.severity_assessment, str):
            severity = severity_mapping.get(
                analysis.severity_assessment.lower(), Severity.SEV_2
            )
        else:
            # Assume numeric severity score
            score = float(analysis.severity_assessment)
            if score >= 0.8:
                severity = Severity.SEV_1
            elif score >= 0.5:
                severity = Severity.SEV_2
            else:
                severity = Severity.SEV_3

        # Create metadata
        metadata = ScenarioMetadata(
            key=f"autonomous_{incident_id}",
            severity=severity,
            description=f"Autonomous incident triggered by anomaly detection: {analysis.log_summary[:100]}...",
        )

        # Create time window (current time window for autonomous detection)
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        window = IncidentWindow(
            start=now - timedelta(minutes=30),  # 30-minute analysis window
            end=now,
        )

        # Create monitoring data from analysis
        monitoring_data = {
            "log_analysis": analysis.log_summary,
            "incident_narrative": f"Autonomous anomaly detection with confidence {analysis.overall_confidence:.2f}",
            "autonomous_detection": True,
            "anomaly_confidence": analysis.overall_confidence,
        }

        # Create additional sources
        additional_sources = {
            "incident_details": [f"Autonomous detection at {now.isoformat()}"],
            "affected_services": [],  # Would be populated from analysis
            "timeline": {"detection_time": now.isoformat()},
        }

        return ScenarioSnapshot(
            metadata=metadata,
            window=window,
            monitoring=monitoring_data,
            additional_sources=additional_sources,
        )

    def _determine_priority_from_analysis(self, analysis: AnalysisResult) -> int:
        """Determine incident priority from analysis results."""
        # Map confidence and severity to priority (1=highest, 5=lowest)
        if analysis.overall_confidence >= 0.9:
            return 1  # Highest priority for very high confidence
        elif analysis.overall_confidence >= 0.8:
            return 2  # High priority
        elif analysis.overall_confidence >= 0.6:
            return 3  # Medium priority
        else:
            return 4  # Lower priority for marginal confidence
