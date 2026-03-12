"""RCA knowledge reader for RAG-based root cause analysis."""

from __future__ import annotations

import re
from pathlib import Path


class RCAKnowledgeReader:
    """Reads and extracts troubleshooting knowledge from policy documents.

    This class implements RAG (Retrieval-Augmented Generation) for RCA,
    providing diagnostic procedures, known failure patterns, and similar incidents
    to help identify root causes faster and more accurately.
    """

    def __init__(
        self, runbooks_path: str | None = None, patterns_path: str | None = None
    ):
        """Initialize the RCA knowledge reader.

        Args:
            runbooks_path: Path to troubleshooting-runbooks.md. If None, uses default.
            patterns_path: Path to known-failure-patterns.md. If None, uses default.
        """
        if runbooks_path is None:
            root = Path(__file__).resolve().parents[3]
            runbooks_path = root / "docs" / "policies" / "troubleshooting-runbooks.md"

        if patterns_path is None:
            root = Path(__file__).resolve().parents[3]
            patterns_path = root / "docs" / "policies" / "known-failure-patterns.md"

        self._runbooks_path = Path(runbooks_path)
        self._patterns_path = Path(patterns_path)
        self._runbooks_content: str | None = None
        self._patterns_content: str | None = None

    def _load_runbooks_content(self) -> str:
        """Load the troubleshooting runbooks content."""
        if self._runbooks_content is None:
            if not self._runbooks_path.exists():
                raise FileNotFoundError(
                    f"Troubleshooting runbooks not found at {self._runbooks_path}"
                )
            self._runbooks_content = self._runbooks_path.read_text()
        return self._runbooks_content

    def _load_patterns_content(self) -> str:
        """Load the known failure patterns content."""
        if self._patterns_content is None:
            if not self._patterns_path.exists():
                raise FileNotFoundError(
                    f"Known failure patterns not found at {self._patterns_path}"
                )
            self._patterns_content = self._patterns_path.read_text()
        return self._patterns_content

    def get_troubleshooting_steps(self, error_pattern: str) -> dict[str, any]:
        """Retrieve diagnostic steps for specific error patterns.

        Args:
            error_pattern: Error message or pattern to search for

        Returns:
            Dictionary containing troubleshooting information
        """
        content = self._load_runbooks_content()

        # Map common error patterns to runbook sections
        pattern_map = {
            "SQLSTATE[08006]": "1",  # Database Connection Issues
            "SQLSTATE[08001]": "1",
            "connection pool": "1",
            "Gateway Timeout": "2",  # Payment Gateway Failures
            "504": "2",
            "502": "2",
            "OutOfMemoryError": "3",  # Memory Leaks
            "OOMKilled": "3",
            "429": "4",  # Rate Limiting
            "Too Many Requests": "4",
            "ImagePullBackOff": "5",  # Deployment Failures
            "Connection refused": "6",  # Cross-Service Communication
            "Cache miss": "7",  # CDN Issues
        }

        # Find best matching section
        section_num = None
        for pattern, num in pattern_map.items():
            if pattern.lower() in error_pattern.lower():
                section_num = num
                break

        if not section_num:
            # Fallback: return general guidance
            return self._get_fallback_troubleshooting()

        # Extract the section
        section_content = self._extract_section(content, section_num)

        # Parse the section into structured data
        return self._parse_troubleshooting_section(section_content, section_num)

    def get_failure_pattern(self, service_or_symptom: str) -> dict[str, any]:
        """Retrieve known failure patterns for a service or symptom.

        Args:
            service_or_symptom: Service name or symptom keyword

        Returns:
            Dictionary containing failure pattern information
        """
        content = self._load_patterns_content()

        # Map keywords to pattern sections
        keyword_map = {
            "payment": "2",
            "database": "3",
            "memory": "4",
            "api": "5",
            "auth": "6",
            "deploy": "7",
            "service": "8",
        }

        # Find best matching section
        section_num = None
        for keyword, num in keyword_map.items():
            if keyword.lower() in service_or_symptom.lower():
                section_num = num
                break

        if not section_num:
            # Try section 2 (Payment) as default
            section_num = "2"

        # Extract and parse section
        section_content = self._extract_section(content, section_num)
        return self._parse_failure_pattern_section(section_content, section_num)

    def get_similar_incidents(self, symptoms: list[str]) -> list[dict[str, any]]:
        """Retrieve similar past incidents based on symptoms.

        Args:
            symptoms: List of symptom keywords

        Returns:
            List of similar incidents with root causes
        """
        content = self._load_runbooks_content()

        incidents = []

        # Search for incident references (INC-YYYY-MM-DD pattern)
        incident_pattern = r"INC-\d{4}-\d{2}-\d{2}:[^*\n]+"
        matches = re.findall(incident_pattern, content, re.MULTILINE)

        for match in matches:
            # Check if any symptom keyword appears in the incident description
            if any(symptom.lower() in match.lower() for symptom in symptoms):
                # Parse incident details
                parts = match.split(":")
                if len(parts) >= 2:
                    incident_id = parts[0].strip()
                    description = parts[1].strip()

                    incidents.append(
                        {"incident_id": incident_id, "description": description}
                    )

        # Also search in known-failure-patterns for historical references
        patterns_content = self._load_patterns_content()
        pattern_matches = re.findall(incident_pattern, patterns_content, re.MULTILINE)

        for match in pattern_matches:
            if any(symptom.lower() in match.lower() for symptom in symptoms):
                parts = match.split(":")
                if len(parts) >= 2:
                    incident_id = parts[0].strip()
                    description = parts[1].strip()

                    # Avoid duplicates
                    if not any(inc["incident_id"] == incident_id for inc in incidents):
                        incidents.append(
                            {"incident_id": incident_id, "description": description}
                        )

        return incidents[:5]  # Return top 5 most relevant

    def get_error_code_guidance(self, error_code: str) -> dict[str, str]:
        """Retrieve guidance for specific error codes.

        Args:
            error_code: Error code (e.g., SQLSTATE[08006], 504, etc.)

        Returns:
            Dictionary with error code meaning and guidance
        """
        content = self._load_runbooks_content()

        # Common error codes and their meanings
        error_codes = {
            "SQLSTATE[08006]": {
                "meaning": "Connection failure - Database rejected or dropped connection",
                "common_causes": [
                    "Connection pool exhaustion",
                    "Database instance restarted",
                    "Network connectivity issues",
                ],
                "diagnostic_steps": [
                    "Check pg_stat_activity for connection count",
                    "Verify database instance status",
                    "Test network connectivity",
                ],
                "policy_reference": "POL-SRE-003 Section 1",
            },
            "504": {
                "meaning": "Gateway Timeout - Backend service took too long to respond",
                "common_causes": [
                    "Backend service degraded",
                    "Slow database queries",
                    "Third-party API timeout",
                ],
                "diagnostic_steps": [
                    "Check backend service health",
                    "Review slow query logs",
                    "Test external API endpoints",
                ],
                "policy_reference": "POL-SRE-003 Section 2",
            },
            "429": {
                "meaning": "Too Many Requests - Rate limit exceeded",
                "common_causes": [
                    "Traffic spike beyond quota",
                    "Missing rate limit backoff",
                    "Retry storm amplification",
                ],
                "diagnostic_steps": [
                    "Check X-RateLimit-Remaining header",
                    "Review traffic patterns",
                    "Implement exponential backoff",
                ],
                "policy_reference": "POL-SRE-003 Section 4",
            },
        }

        # Normalize error code
        normalized_code = error_code.upper().strip()

        # Direct match
        if normalized_code in error_codes:
            return error_codes[normalized_code]

        # Partial match
        for code, guidance in error_codes.items():
            if code in normalized_code or normalized_code in code:
                return guidance

        # Fallback
        return {
            "meaning": f"Error code {error_code}",
            "common_causes": ["Check troubleshooting runbooks for details"],
            "diagnostic_steps": ["Review logs for context", "Check recent deployments"],
            "policy_reference": "POL-SRE-003",
        }

    def get_all_error_patterns(self) -> list[str]:
        """Retrieve list of all documented error patterns.

        Returns:
            List of error pattern strings
        """
        content = self._load_runbooks_content()

        patterns = []

        # Extract error patterns from section 1.1, 2.1, etc.
        error_section_pattern = r"### \d+\.\d+ Error Patterns.*?(?=###|\Z)"
        matches = re.findall(error_section_pattern, content, re.DOTALL)

        for match in matches:
            # Extract bullet points and code patterns
            lines = match.split("\n")
            for line in lines:
                line = line.strip()
                if (
                    line.startswith("-")
                    or line.startswith("*")
                    or "`" in line
                    or "SQLSTATE" in line
                ):
                    # Clean up the pattern
                    pattern = line.lstrip("- *").strip("`").strip()
                    if pattern and len(pattern) > 3:
                        patterns.append(pattern)

        return patterns

    def search_runbooks_by_keyword(self, keyword: str) -> list[dict[str, str]]:
        """Search runbooks for sections containing keyword.

        Args:
            keyword: Keyword to search for

        Returns:
            List of matching sections with context
        """
        content = self._load_runbooks_content()

        results = []
        sections = content.split("\n## ")

        for section in sections[1:]:  # Skip first (header)
            if keyword.lower() in section.lower():
                # Extract section title
                lines = section.split("\n")
                title = lines[0].strip() if lines else "Unknown"

                # Get first 200 characters as preview
                preview = section[:200].replace("\n", " ")

                results.append({"title": title, "preview": preview, "keyword": keyword})

        return results

    def _extract_section(self, content: str, section_num: str) -> str:
        """Extract a specific numbered section from markdown content.

        Args:
            content: Full markdown content
            section_num: Section number (e.g., "1", "2.1")

        Returns:
            Extracted section content
        """
        lines = content.split("\n")
        section_lines = []
        in_section = False
        section_level = len(section_num.split("."))

        for line in lines:
            # Check if this is the start of our target section
            if line.startswith(f"## {section_num} ") or line.startswith(
                f"### {section_num} "
            ):
                in_section = True
                section_lines.append(line)
                continue

            # Check if we've hit the next section
            if in_section and (line.startswith("## ") or line.startswith("### ")):
                # Extract section number from new heading
                heading_match = re.match(r"^#{2,3} (\d+(?:\.\d+)?)", line)
                if heading_match:
                    new_section_num = heading_match.group(1)
                    new_section_level = len(new_section_num.split("."))

                    # Stop if we hit a same-level or higher-level section
                    if new_section_level <= section_level:
                        break

            if in_section:
                section_lines.append(line)

        return (
            "\n".join(section_lines)
            if section_lines
            else f"Section {section_num} not found"
        )

    def _parse_troubleshooting_section(
        self, section_content: str, section_num: str
    ) -> dict[str, any]:
        """Parse troubleshooting section into structured data."""
        # Extract subsections
        error_patterns = self._extract_subsection(section_content, "Error Patterns")
        root_causes = self._extract_subsection(section_content, "Root Causes")
        diagnostic_steps = self._extract_subsection(section_content, "Diagnostic Steps")
        similar_incidents = self._extract_subsection(
            section_content, "Similar Past Incidents"
        )

        # Extract section title
        title_match = re.match(r"^#{2,3} \d+(?:\.\d+)? (.+)", section_content)
        section_name = title_match.group(1) if title_match else f"Section {section_num}"

        return {
            "section_name": section_name,
            "section_number": section_num,
            "error_patterns": error_patterns,
            "root_causes": root_causes,
            "diagnostic_steps": diagnostic_steps,
            "similar_incidents": similar_incidents,
            "policy_reference": f"POL-SRE-003 Section {section_num}",
        }

    def _parse_failure_pattern_section(
        self, section_content: str, section_num: str
    ) -> dict[str, any]:
        """Parse failure pattern section into structured data."""
        symptom_pattern = self._extract_subsection(section_content, "Symptom Pattern")
        root_cause_dist = self._extract_subsection(
            section_content, "Root Cause Distribution"
        )
        diagnostic_q = self._extract_subsection(section_content, "Diagnostic Questions")

        # Extract section title
        title_match = re.match(r"^#{2,3} \d+(?:\.\d+)? (.+)", section_content)
        section_name = title_match.group(1) if title_match else f"Section {section_num}"

        return {
            "section_name": section_name,
            "section_number": section_num,
            "symptom_pattern": symptom_pattern,
            "root_cause_distribution": root_cause_dist,
            "diagnostic_questions": diagnostic_q,
            "policy_reference": f"POL-SRE-004 Section {section_num}",
        }

    def _extract_subsection(self, content: str, subsection_name: str) -> str:
        """Extract a subsection by name (e.g., 'Error Patterns')."""
        # Match subsection heading
        pattern = rf"###+ [\d\.]* ?{re.escape(subsection_name)}.*?\n(.*?)(?=\n###|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        return f"{subsection_name} not found"

    def _get_fallback_troubleshooting(self) -> dict[str, any]:
        """Provide fallback troubleshooting guidance for unknown patterns."""
        return {
            "section_name": "General Troubleshooting",
            "section_number": "0",
            "error_patterns": "Unknown error pattern",
            "root_causes": "Check logs for context, review recent deployments",
            "diagnostic_steps": "1. Review error logs\n2. Check recent changes\n3. Verify service health",
            "similar_incidents": "No similar incidents found",
            "policy_reference": "POL-SRE-003 General Guidance",
        }


__all__ = ["RCAKnowledgeReader"]
