# SRE Policy Documents

This directory contains authoritative policy documents used by the RAG-enhanced AI agents for incident response. These documents ensure compliance, accuracy, and consistency in agent outputs.

---

## Policy Documents Overview

### POL-SRE-001: Incident Response Procedures
**File:** `incident-response-procedures.md`
**Used By:** Mitigation Agent
**Purpose:** Provides step-by-step incident response procedures, escalation paths, and communication templates.

**Key Sections:**
- Communication protocols
- Escalation procedures
- Incident severity definitions
- Response timelines
- Post-incident review requirements

**Agent Integration:**
- Mitigation Agent reads this document via `PolicyReader` class
- Ensures compliance-driven mitigation planning
- Provides structured response workflows

---

### POL-SRE-002: Business Impact Baselines
**File:** `business-impact-baselines.md`
**Version:** 1.0
**Last Updated:** November 2024
**Used By:** Impact Agent
**Purpose:** Authoritative baseline metrics for revenue loss calculations to prevent LLM hallucination.

**Key Sections:**
1. **Overview** - Baseline definitions and data sources
2. **Severity Thresholds** - SEV-1/2/3 classification criteria
3. **Baseline Metrics** - TPS, revenue, success rates by severity
4. **Revenue Formulas** - Calculation methodologies
5. **SLA Thresholds** - Breach definitions by service tier

**Baseline Metrics:**
| Severity | TPS | Revenue/min | Success Rate | MTTR Target |
|----------|-----|-------------|--------------|-------------|
| SEV-1    | 1650 | $6,500     | 99.5%        | 15 min      |
| SEV-2    | 1200 | $4,200     | 99.0%        | 30 min      |
| SEV-3    | 850  | $2,800     | 98.5%        | 60 min      |

**Agent Integration:**
- Impact Agent reads this document via `BusinessMetricsReader` class
- Retrieves baseline TPS, revenue, and SLA thresholds based on incident severity
- Injects revenue formulas into LLM context for accurate calculations
- **Test Coverage:** 86% (22/22 tests passing)

---

### POL-SRE-003: Troubleshooting Runbooks
**File:** `troubleshooting-runbooks.md`
**Version:** 1.0
**Last Updated:** November 2024
**Used By:** RCA Agent
**Purpose:** Tactical diagnostic procedures for 7 common failure scenarios with historical incident references.

**Key Sections:**
1. **Database Connection Issues** - Connection pool, network, authentication
2. **Payment Gateway Failures** - Third-party degradation, rate limiting, timeouts
3. **Application Memory Leaks** - Unclosed resources, cache growth, event listeners
4. **API Rate Limiting and Throttling** - Traffic spikes, retry storms, quota exhaustion
5. **Service Deployment Failures** - Image pull errors, health check failures
6. **Cross-Service Communication Failures** - DNS, network policies, circuit breakers
7. **CDN and Caching Issues** - Cache miss rates, origin overload

**Each Section Contains:**
- **Error Patterns** - Common error codes and messages (e.g., `SQLSTATE[08006]`, `504 Gateway Timeout`)
- **Root Causes** - Distribution percentages from historical incidents
- **Diagnostic Steps** - SQL queries, bash commands, CloudWatch metrics
- **Similar Past Incidents** - Historical references (e.g., `INC-2024-01-15`)

**Example:**
```markdown
## 1. Database Connection Issues

### 1.1 Error Patterns
- SQLSTATE[08006] - Connection failure
- connection pool exhausted

### 1.2 Known Root Causes
1. Connection pool exhaustion (45% of cases)
2. Database instance resource saturation (25%)
3. Network connectivity issues (15%)

### 1.3 Diagnostic Steps
SELECT count(*) FROM pg_stat_activity WHERE state='active';

### 1.4 Similar Past Incidents
- INC-2024-01-15: Connection Pool Leak After ORM Upgrade
- INC-2024-02-03: Database Failover Connection Storm
```

**Agent Integration:**
- RCA Agent reads this document via `RCAKnowledgeReader` class
- Maps error patterns to specific runbook sections
- Injects diagnostic procedures (SQL, bash) into LLM context
- Provides historical incident context for root cause ranking
- **Test Coverage:** 90% (20/20 tests passing)

---

### POL-SRE-004: Known Failure Patterns
**File:** `known-failure-patterns.md`
**Version:** 1.0
**Last Updated:** November 2024
**Used By:** RCA Agent
**Purpose:** Historical failure pattern statistics and root cause distributions from 2024 incidents.

**Key Sections:**
1. **Overview** - Pattern catalog methodology
2. **Payment Processing Failures** - 47 incidents analyzed
3. **Database Performance Degradation** - 38 incidents analyzed
4. **Memory Leaks and OOM Errors** - 24 incidents analyzed
5. **API Gateway and Load Balancer Issues** - 19 incidents analyzed
6. **Authentication and Authorization Failures** - 16 incidents analyzed
7. **Deployment and Rollout Failures** - 31 incidents analyzed
8. **Cross-Service Communication Failures** - 22 incidents analyzed

**Appendices:**
- **Appendix A:** Incident Pattern Correlation Matrix
- **Appendix B:** Top 10 Incidents by Frequency

**Example Failure Pattern:**
```markdown
## 2. Payment Processing Failures

### 2.2 Root Cause Distribution
Based on 47 incidents (Jan-Oct 2024):

| Root Cause | Frequency | MTTR | Severity |
|------------|-----------|------|----------|
| Third-party payment gateway degradation | 45% (21 incidents) | 35 min | SEV-2 |
| Database connection pool exhaustion | 30% (14 incidents) | 25 min | SEV-2 |
| API rate limiting from payment provider | 15% (7 incidents) | 20 min | SEV-3 |

### 2.3 Diagnostic Questions
1. Is the third-party payment provider status page reporting issues?
   - If YES → 45% likely external issue, implement failover
2. Are database connection pools saturated?
   - Check: SELECT count(*) FROM pg_stat_activity WHERE state='active'
```

**Agent Integration:**
- RCA Agent reads this document via `RCAKnowledgeReader` class
- Retrieves root cause distribution percentages for hypothesis ranking
- Provides diagnostic question priorities
- Injects historical MTTR and frequency statistics
- **Test Coverage:** 90% (included in RCAKnowledgeReader tests)

---

## RAG Architecture

### Document-Based Retrieval
All policy documents use **section-based retrieval**, not semantic/vector search:

1. **Error Pattern Mapping** - Maps error codes to specific sections
   - `SQLSTATE[08006]` → Section 1 (Database Connection Issues)
   - `Gateway Timeout` → Section 2 (Payment Gateway Failures)

2. **Severity-Based Selection** - Selects appropriate baselines
   - `SEV-1` → Section 3.1 (Business Impact Baselines)
   - `SEV-2` → Section 3.2

3. **Keyword Matching** - Searches for service/symptom keywords
   - `"payment"` → Section 2 (Payment Processing Failures)
   - `"memory"` → Section 4 (Memory Leaks)

### Reader Classes

| Policy Document | Reader Class | Agent | Test Coverage |
|----------------|--------------|-------|---------------|
| POL-SRE-001 | `PolicyReader` | Mitigation Agent | 23% |
| POL-SRE-002 | `BusinessMetricsReader` | Impact Agent | 86% |
| POL-SRE-003 | `RCAKnowledgeReader` | RCA Agent | 90% |
| POL-SRE-004 | `RCAKnowledgeReader` | RCA Agent | 90% |

### Caching Strategy
- **First Load:** Policy documents read from disk
- **Subsequent Calls:** Content cached in memory (`_runbooks_content`, `_patterns_content`)
- **No Expiration:** Cache persists for agent lifetime (stateless agents recreated per request)

---

## Usage Examples

### Impact Agent - Baseline Retrieval
```python
from src.orchestration.four_agent.business_metrics_reader import BusinessMetricsReader
from src.orchestration.four_agent.schema import Severity

reader = BusinessMetricsReader()
baselines = reader.get_baseline_metrics(Severity.SEV_2)

# Returns:
{
    "baseline_tps": 1200.0,
    "baseline_revenue_per_min": 4200.0,
    "baseline_success_rate": 99.0,
    "mttr_target_minutes": 30,
    "policy_reference": "POL-SRE-002 Section 3.2"
}
```

### RCA Agent - Troubleshooting Steps
```python
from src.orchestration.four_agent.rca_knowledge_reader import RCAKnowledgeReader

reader = RCAKnowledgeReader()
steps = reader.get_troubleshooting_steps("SQLSTATE[08006]")

# Returns:
{
    "section_name": "Database Connection Issues",
    "error_patterns": ["SQLSTATE[08006] - Connection failure", ...],
    "root_causes": [
        "Connection pool exhaustion (45% of cases)",
        "Database instance resource saturation (25%)", ...
    ],
    "diagnostic_steps": [
        "Check connection pool metrics: SELECT count(*) FROM pg_stat_activity",
        "Verify instance resource utilization: CloudWatch CPUUtilization", ...
    ],
    "similar_incidents": [
        {"incident_id": "INC-2024-01-15", "description": "Connection Pool Leak..."},
        ...
    ],
    "policy_reference": "POL-SRE-003 Section 1"
}
```

### RCA Agent - Failure Patterns
```python
pattern = reader.get_failure_pattern("payment")

# Returns:
{
    "section_name": "Payment Processing Failures",
    "symptom_pattern": "Error rate spike in payment-processor service...",
    "root_cause_distribution": [
        "Third-party payment gateway degradation: 45% (21 incidents)",
        "Database connection pool exhaustion: 30% (14 incidents)", ...
    ],
    "diagnostic_questions": [
        "Is the third-party payment provider status page reporting issues?",
        "Are database connection pools saturated?", ...
    ],
    "policy_reference": "POL-SRE-004 Section 2"
}
```

---

## Maintenance Guidelines

### Updating Policy Documents

1. **Version Control:**
   - Increment version number in document header
   - Update "Last Updated" timestamp
   - Document changes in commit message

2. **Testing:**
   - Run affected agent tests after updates
   - Verify section numbers remain consistent
   - Check markdown formatting

3. **Validation:**
   ```bash
   # Test Impact Agent integration
   pytest tests/orchestration/four_agent/test_impact_agent.py -v
   pytest tests/orchestration/four_agent/test_business_metrics_reader.py -v

   # Test RCA Agent integration
   pytest tests/orchestration/four_agent/test_rca_agent.py -v
   pytest tests/orchestration/four_agent/test_rca_knowledge_reader.py -v
   ```

### Adding New Policy Documents

1. **Create Document:**
   - Use existing policy structure as template
   - Include policy ID (POL-SRE-XXX)
   - Add version and ownership metadata

2. **Create Reader Class:**
   - Implement in `src/orchestration/four_agent/`
   - Add section extraction methods
   - Implement caching strategy

3. **Write Tests:**
   - Create test file in `tests/orchestration/four_agent/`
   - Test document loading
   - Test section extraction
   - Test edge cases (missing sections, unknown patterns)

4. **Integrate into Agent:**
   - Initialize reader in agent `__init__`
   - Inject knowledge in `_build_user_prompt`
   - Update agent tests

5. **Document:**
   - Add entry to this README
   - Update PROJECT_OVERVIEW.md
   - Update main README.md

---

## Document Format Standards

### Section Numbering
```markdown
## 1. Top-Level Section
### 1.1 Subsection
### 1.2 Subsection

## 2. Top-Level Section
### 2.1 Subsection
```

### Metadata Headers
```markdown
**Policy ID:** POL-SRE-XXX
**Version:** 1.0
**Last Updated:** Month YYYY
**Owner:** Team Name
```

### Tables
```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value    | Value    | Value    |
```

### Code Blocks
```markdown
```sql
-- SQL queries with comments
SELECT * FROM table;
```

```bash
# Bash commands with comments
aws cloudwatch get-metric-statistics ...
```
```

---

## Policy Document Governance

### Review Schedule
- **Monthly:** POL-SRE-002 (Business Impact Baselines) - Update with latest metrics
- **Quarterly:** POL-SRE-003, POL-SRE-004 (Troubleshooting & Patterns) - Add new incident data
- **Annual:** POL-SRE-001 (Incident Response Procedures) - Review compliance requirements

### Approval Workflow
1. **Draft** - Author creates/updates document
2. **Review** - SRE team reviews changes
3. **Test** - Validate agent integration tests pass
4. **Approve** - Manager approval required
5. **Deploy** - Merge to main branch

### Compliance Requirements
- All policy documents must include version number
- Changes require test validation before deployment
- Historical versions archived in git history
- Critical changes trigger agent re-certification

---

## Additional Resources

- **Implementation Details:** [PROJECT_OVERVIEW.md](../../PROJECT_OVERVIEW.md)
- **Agent Architecture:** [docs/DESIGN.md](../DESIGN.md)
- **Test Results:** Run `pytest tests/orchestration/four_agent/ -v --cov`

---

**Document Classification:** Internal - SRE Operations
**Last Updated:** November 2024
**Next Review Due:** December 2024
