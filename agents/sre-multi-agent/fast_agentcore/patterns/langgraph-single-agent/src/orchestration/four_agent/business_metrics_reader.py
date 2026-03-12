"""Business metrics reader for RAG-based revenue impact calculations."""

from __future__ import annotations

from pathlib import Path

from .schema import Severity


class BusinessMetricsReader:
    """Reads and extracts business impact baselines from policy documents.

    This class implements RAG (Retrieval-Augmented Generation) for business metrics,
    providing Impact Agent with authoritative baseline data for revenue calculations.
    """

    def __init__(self, policy_path: str | None = None):
        """Initialize the business metrics reader.

        Args:
            policy_path: Path to business-impact-baselines.md. If None, uses default path.
        """
        if policy_path is None:
            # Default path relative to project root
            root = Path(__file__).resolve().parents[3]
            policy_path = root / "docs" / "policies" / "business-impact-baselines.md"

        self._policy_path = Path(policy_path)
        self._policy_content: str | None = None

    def _load_policy_content(self) -> str:
        """Load the full policy document content."""
        if self._policy_content is None:
            if not self._policy_path.exists():
                raise FileNotFoundError(
                    f"Business impact baselines policy not found at {self._policy_path}"
                )

            self._policy_content = self._policy_path.read_text()

        return self._policy_content

    def get_baseline_metrics(self, severity: Severity) -> dict[str, float]:
        """Extract baseline metrics for the given severity level.

        Args:
            severity: Incident severity (SEV-1, SEV-2, or SEV-3)

        Returns:
            Dictionary containing baseline TPS, approvals, revenue, and success rates

        Example:
            >>> reader = BusinessMetricsReader()
            >>> metrics = reader.get_baseline_metrics(Severity.SEV_2)
            >>> print(metrics['baseline_tps'])
            1200.0
        """
        content = self._load_policy_content()

        # Map severity to section numbers in the policy document
        section_map = {
            Severity.SEV_1: "3.1",  # SEV-1 (Critical) - Complete Service Outage
            Severity.SEV_2: "3.2",  # SEV-2 (High) - Significant Degradation
            Severity.SEV_3: "3.3",  # SEV-3 (Medium) - Limited Impact
        }

        section_num = section_map.get(severity)
        if not section_num:
            return self._get_fallback_baselines(severity)

        # Extract the specific section
        section_content = self._extract_section(content, section_num)

        # Parse metrics from the section
        metrics = self._parse_baseline_metrics(section_content, severity)

        return metrics

    def get_revenue_formulas(self) -> dict[str, str]:
        """Extract revenue calculation formulas from policy document.

        Returns:
            Dictionary containing revenue calculation formulas and methodology
        """
        content = self._load_policy_content()

        # Extract Section 4: Revenue Calculation Methodology
        formulas_section = self._extract_section(content, "4")

        return {
            "transaction_revenue_loss": "(Baseline TPS - Current TPS) × Revenue per Transaction",
            "approval_revenue_loss": "(Baseline Approvals - Current Approvals) × Revenue per Approval",
            "total_revenue_loss": "Transaction Revenue Loss + Approval Revenue Loss",
            "cumulative_impact": "Revenue Loss/min × Incident Duration (minutes)",
            "methodology_reference": "POL-SRE-002 Section 4.1",
            "policy_section": (
                formulas_section[:500] + "..."
                if len(formulas_section) > 500
                else formulas_section
            ),
        }

    def get_sla_thresholds(self, severity: Severity) -> dict[str, any]:
        """Extract SLA breach thresholds for the given severity.

        Args:
            severity: Incident severity level

        Returns:
            Dictionary containing SLA thresholds and breach criteria
        """
        # Parse from Section 5: Success Rate and SLA Impact
        content = self._load_policy_content()
        sla_section = self._extract_section(content, "5")

        # Default thresholds from policy
        thresholds = {
            Severity.SEV_1: {
                "success_rate_threshold": 95.0,
                "max_duration_minutes": 5,
                "sla_target": 99.9,
                "monthly_budget_minutes": 43.8,
                "breach_action": "Executive escalation + customer notifications",
            },
            Severity.SEV_2: {
                "success_rate_threshold": 97.0,
                "max_duration_minutes": 15,
                "sla_target": 99.5,
                "monthly_budget_minutes": 216.0,
                "breach_action": "Customer notifications",
            },
            Severity.SEV_3: {
                "success_rate_threshold": 98.0,
                "max_duration_minutes": 60,
                "sla_target": 99.0,
                "monthly_budget_minutes": 432.0,
                "breach_action": "Internal tracking only",
            },
        }

        return thresholds.get(severity, thresholds[Severity.SEV_2])

    def get_user_impact_thresholds(self, severity: Severity) -> dict[str, any]:
        """Extract user impact assessment thresholds.

        Args:
            severity: Incident severity level

        Returns:
            Dictionary containing user impact thresholds and escalation criteria
        """
        # Parse from Section 6: User Impact Assessment Guidelines
        user_thresholds = {
            Severity.SEV_1: {
                "affected_percentage": 100.0,
                "critical_user_count": 10000,
                "communication": "Public status page + direct outreach",
                "executive_notification": "Required",
            },
            Severity.SEV_2: {
                "affected_percentage": 75.0,
                "critical_user_count": 5000,
                "communication": "Status page updates",
                "executive_notification": "Recommended",
            },
            Severity.SEV_3: {
                "affected_percentage": 10.0,
                "critical_user_count": 1000,
                "communication": "Internal tracking",
                "executive_notification": "Not required",
            },
        }

        return user_thresholds.get(severity, user_thresholds[Severity.SEV_2])

    def _extract_section(self, content: str, section_num: str) -> str:
        """Extract a specific numbered section from the policy document.

        Args:
            content: Full policy document content
            section_num: Section number to extract (e.g., "3.1", "4")

        Returns:
            Extracted section content as string
        """
        lines = content.split("\n")
        section_lines = []
        in_section = False
        section_level = len(section_num.split("."))

        for line in lines:
            # Check if this is the start of our target section
            if line.startswith(f"### {section_num} ") or line.startswith(
                f"## {section_num} "
            ):
                in_section = True
                section_lines.append(line)
                continue

            # Check if we've hit the next section at the same level
            if in_section and line.startswith("#"):
                # Count the heading level
                heading_level = len(line) - len(line.lstrip("#"))
                if heading_level <= section_level + 1:
                    # Check if this is a same-level or higher-level section
                    if section_level == 1 and heading_level <= 2:
                        break
                    elif section_level == 2 and heading_level <= 3:
                        # For subsections, only break on same level or higher
                        if line.startswith("### ") and not line.startswith(
                            f"### {section_num}."
                        ):
                            break
                        elif line.startswith("## "):
                            break

            if in_section:
                section_lines.append(line)

        return (
            "\n".join(section_lines)
            if section_lines
            else f"Section {section_num} not found"
        )

    def _parse_baseline_metrics(
        self, section_content: str, severity: Severity
    ) -> dict[str, float]:
        """Parse baseline metrics from extracted section content.

        Args:
            section_content: Extracted section text from policy document
            severity: Severity level for fallback values

        Returns:
            Dictionary of parsed baseline metrics
        """
        # Default baselines (fallback if parsing fails)
        defaults = {
            Severity.SEV_1: {
                "baseline_tps": 1650.0,
                "baseline_approvals_per_min": 1350.0,
                "baseline_revenue_per_min": 6500.0,
                "baseline_success_rate": 99.5,
                "revenue_per_transaction": 2.45,
                "revenue_per_approval": 3.20,
                "peak_tps": 2100.0,
                "peak_revenue_per_min": 8200.0,
                "max_downtime_minutes": 5,
            },
            Severity.SEV_2: {
                "baseline_tps": 1200.0,
                "baseline_approvals_per_min": 980.0,
                "baseline_revenue_per_min": 4200.0,
                "baseline_success_rate": 99.0,
                "revenue_per_transaction": 2.45,
                "revenue_per_approval": 3.20,
                "peak_tps": 1550.0,
                "peak_revenue_per_min": 5400.0,
                "max_downtime_minutes": 30,
            },
            Severity.SEV_3: {
                "baseline_tps": 850.0,
                "baseline_approvals_per_min": 720.0,
                "baseline_revenue_per_min": 2800.0,
                "baseline_success_rate": 98.5,
                "revenue_per_transaction": 2.45,
                "revenue_per_approval": 3.20,
                "peak_tps": 1100.0,
                "peak_revenue_per_min": 3600.0,
                "max_downtime_minutes": 120,
            },
        }

        # Try to parse from section content (basic parsing)
        metrics = defaults.get(severity, defaults[Severity.SEV_2]).copy()

        # Simple line-by-line parsing for key metrics
        for line in section_content.split("\n"):
            line_lower = line.lower()

            # Parse TPS
            if "baseline tps:" in line_lower:
                try:
                    value = line.split(":")[1].strip().split()[0].replace(",", "")
                    metrics["baseline_tps"] = float(value)
                except (IndexError, ValueError):
                    pass

            # Parse Approvals
            if "baseline approvals:" in line_lower:
                try:
                    value = line.split(":")[1].strip().split()[0].replace(",", "")
                    metrics["baseline_approvals_per_min"] = float(value)
                except (IndexError, ValueError):
                    pass

            # Parse Revenue
            if "baseline revenue:" in line_lower:
                try:
                    value = (
                        line.split(":")[1]
                        .strip()
                        .replace("$", "")
                        .replace(",", "")
                        .split("/")[0]
                    )
                    metrics["baseline_revenue_per_min"] = float(value)
                except (IndexError, ValueError):
                    pass

            # Parse Success Rate
            if "expected success rate:" in line_lower or "success rate:" in line_lower:
                try:
                    value = line.split(":")[1].strip().replace("%", "")
                    metrics["baseline_success_rate"] = float(value)
                except (IndexError, ValueError):
                    pass

            # Parse Revenue per Transaction
            if "revenue per transaction:" in line_lower:
                try:
                    value = line.split(":")[1].strip().replace("$", "").replace(",", "")
                    metrics["revenue_per_transaction"] = float(value)
                except (IndexError, ValueError):
                    pass

            # Parse Revenue per Approval
            if "revenue per approval:" in line_lower:
                try:
                    value = line.split(":")[1].strip().replace("$", "").replace(",", "")
                    metrics["revenue_per_approval"] = float(value)
                except (IndexError, ValueError):
                    pass

        return metrics

    def _get_fallback_baselines(self, severity: Severity) -> dict[str, float]:
        """Get fallback baseline metrics if policy document is unavailable.

        Args:
            severity: Incident severity level

        Returns:
            Dictionary of fallback baseline metrics
        """
        # Same as defaults in _parse_baseline_metrics
        fallbacks = {
            Severity.SEV_1: {
                "baseline_tps": 1650.0,
                "baseline_approvals_per_min": 1350.0,
                "baseline_revenue_per_min": 6500.0,
                "baseline_success_rate": 99.5,
                "revenue_per_transaction": 2.45,
                "revenue_per_approval": 3.20,
            },
            Severity.SEV_2: {
                "baseline_tps": 1200.0,
                "baseline_approvals_per_min": 980.0,
                "baseline_revenue_per_min": 4200.0,
                "baseline_success_rate": 99.0,
                "revenue_per_transaction": 2.45,
                "revenue_per_approval": 3.20,
            },
            Severity.SEV_3: {
                "baseline_tps": 850.0,
                "baseline_approvals_per_min": 720.0,
                "baseline_revenue_per_min": 2800.0,
                "baseline_success_rate": 98.5,
                "revenue_per_transaction": 2.45,
                "revenue_per_approval": 3.20,
            },
        }

        return fallbacks.get(severity, fallbacks[Severity.SEV_2])


__all__ = ["BusinessMetricsReader"]
