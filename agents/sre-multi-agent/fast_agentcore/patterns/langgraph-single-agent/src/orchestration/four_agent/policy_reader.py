"""Policy document reader for RAG-based incident response procedures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from .schema import Severity


class PolicyReader:
    """Reads and extracts relevant sections from incident response policy documents."""
    
    def __init__(self, policy_path: Optional[str] = None):
        if policy_path is None:
            # Default path relative to project root
            root = Path(__file__).resolve().parents[3]
            policy_path = root / "docs" / "policies" / "incident-response-procedures.md"
        
        self._policy_path = Path(policy_path)
        self._policy_content: Optional[str] = None
    
    def _load_policy_content(self) -> str:
        """Load the full policy document content."""
        if self._policy_content is None:
            if not self._policy_path.exists():
                raise FileNotFoundError(f"Policy document not found at {self._policy_path}")
            
            self._policy_content = self._policy_path.read_text()
        
        return self._policy_content
    
    def get_severity_procedures(self, severity: Severity) -> str:
        """Extract relevant policy procedures for the given severity level."""
        content = self._load_policy_content()
        
        # Map severity to policy section numbers
        section_map = {
            Severity.SEV_1: "3.1",
            Severity.SEV_2: "3.2", 
            Severity.SEV_3: "3.3"
        }
        
        section_num = section_map.get(severity)
        if not section_num:
            return self._get_general_procedures()
        
        # Extract the specific section
        section_content = self._extract_section(content, section_num)
        
        # Also include general coordination procedures
        coordination_section = self._extract_section(content, "4")
        
        return f"""
MANDATORY PROCEDURES FOR {severity.value} INCIDENTS:

{section_content}

COORDINATION AND HANDOFF PROCEDURES:
{coordination_section}

POLICY REFERENCE: POL-SRE-001 Incident Response Procedures and Mitigation Policy
"""
    
    def _extract_section(self, content: str, section_num: str) -> str:
        """Extract a specific numbered section from the policy document."""
        lines = content.split('\n')
        section_lines = []
        in_section = False
        section_level = len(section_num.split('.'))
        
        for line in lines:
            # Check if this is the start of our target section
            if line.startswith(f"### {section_num} ") or line.startswith(f"## {section_num} "):
                in_section = True
                section_lines.append(line)
                continue
            
            # Check if we've hit the next section at the same level
            if in_section and line.startswith('#'):
                # Count the heading level
                heading_level = len(line) - len(line.lstrip('#'))
                if heading_level <= section_level + 1:
                    # Check if this is a same-level or higher-level section
                    if section_level == 1 and heading_level <= 2:
                        break
                    elif section_level == 2 and heading_level <= 3:
                        # For subsections, only break on same level or higher
                        if line.startswith('### ') and not line.startswith(f'### {section_num}.'):
                            break
                        elif line.startswith('## '):
                            break
            
            if in_section:
                section_lines.append(line)
        
        return '\n'.join(section_lines) if section_lines else f"Section {section_num} not found"
    
    def _get_general_procedures(self) -> str:
        """Get general incident response procedures when specific severity not found."""
        return """
GENERAL INCIDENT RESPONSE PROCEDURES:
1. Coordinate with appropriate teams based on incident scope
2. Obtain required approvals according to severity level
3. Execute mitigation steps with proper handoff procedures
4. Maintain communication with stakeholders
5. Document all actions for post-incident review

POLICY REFERENCE: POL-SRE-001 Incident Response Procedures and Mitigation Policy
"""
    
    def get_approval_requirements(self, severity: Severity) -> str:
        """Get approval authority requirements for the given severity."""
        approval_map = {
            Severity.SEV_1: "Dual Approval Required: Incident Commander AND Risk Manager",
            Severity.SEV_2: "Single Approval: Incident Commander sufficient for operational changes", 
            Severity.SEV_3: "Single Approval: Incident Commander for infrastructure changes"
        }
        
        return approval_map.get(severity, "Standard approval process required")
    
    def get_business_impact_thresholds(self) -> str:
        """Get business impact calculation guidelines from policy."""
        content = self._load_policy_content()
        business_section = self._extract_section(content, "5")
        return business_section


__all__ = ["PolicyReader"]
