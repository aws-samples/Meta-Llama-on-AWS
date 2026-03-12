# Business Impact Baselines and Revenue Calculation Guidelines

**Policy ID:** POL-SRE-002
**Version:** 1.0
**Last Updated:** November 2024
**Owner:** SRE Team & Finance Operations

---

## 1. Overview

This document establishes standardized baseline metrics for calculating business impact during incident response. All SRE agents must use these documented baselines to ensure consistent, accurate revenue loss calculations and impact assessments.

## 2. Purpose

- Provide authoritative baseline metrics for incident impact calculations
- Ensure consistency across all incident response agents
- Enable accurate revenue loss estimation during service degradation
- Support executive decision-making with reliable business impact data

---

## 3. Baseline Metrics by Severity Level

### 3.1 SEV-1 (Critical) - Complete Service Outage

**Transaction Performance**
- Baseline TPS: 1650 transactions/minute
- Expected Success Rate: 99.5%
- Peak Hour TPS: 2100 transactions/minute

**Approval Performance**
- Baseline Approvals: 1350 approvals/minute
- Approval Success Rate: 99.5%
- Peak Hour Approvals: 1700 approvals/minute

**Revenue Baselines**
- Baseline Revenue: $6,500/minute
- Peak Hour Revenue: $8,200/minute
- Average Revenue per Transaction: $2.45
- Average Revenue per Approval: $3.20

**User Impact Thresholds**
- Critical User Count: All users affected (100%)
- SLA Breach: Immediate (within 5 minutes)
- Maximum Acceptable Downtime: 5 minutes

### 3.2 SEV-2 (High) - Significant Degradation

**Transaction Performance**
- Baseline TPS: 1200 transactions/minute
- Expected Success Rate: 99.0%
- Peak Hour TPS: 1550 transactions/minute

**Approval Performance**
- Baseline Approvals: 980 approvals/minute
- Approval Success Rate: 99.0%
- Peak Hour Approvals: 1250 approvals/minute

**Revenue Baselines**
- Baseline Revenue: $4,200/minute
- Peak Hour Revenue: $5,400/minute
- Average Revenue per Transaction: $2.45
- Average Revenue per Approval: $3.20

**User Impact Thresholds**
- Affected User Percentage: 50-100%
- SLA Breach: Within 15 minutes
- Maximum Acceptable Duration: 30 minutes

### 3.3 SEV-3 (Medium) - Limited Impact

**Transaction Performance**
- Baseline TPS: 850 transactions/minute
- Expected Success Rate: 98.5%
- Peak Hour TPS: 1100 transactions/minute

**Approval Performance**
- Baseline Approvals: 720 approvals/minute
- Approval Success Rate: 98.5%
- Peak Hour Approvals: 920 approvals/minute

**Revenue Baselines**
- Baseline Revenue: $2,800/minute
- Peak Hour Revenue: $3,600/minute
- Average Revenue per Transaction: $2.45
- Average Revenue per Approval: $3.20

**User Impact Thresholds**
- Affected User Percentage: <10%
- SLA Breach: Within 60 minutes
- Maximum Acceptable Duration: 120 minutes

---

## 4. Revenue Calculation Methodology

### 4.1 Standard Revenue Formulas

**Transaction-Based Revenue Loss**
```
Revenue Loss/min = (Baseline TPS - Current TPS) × Revenue per Transaction
```

**Approval-Based Revenue Loss**
```
Revenue Loss/min = (Baseline Approvals - Current Approvals) × Revenue per Approval
```

**Combined Revenue Loss**
```
Total Revenue Loss/min = Transaction Revenue Loss + Approval Revenue Loss
```

**Cumulative Revenue Impact**
```
Total Revenue Impact = Revenue Loss/min × Incident Duration (minutes)
```

### 4.2 Time-Based Adjustments

**Business Hours (9 AM - 5 PM ET)**
- Use baseline metrics as-is
- No adjustment factor required

**Peak Hours (12 PM - 2 PM ET)**
- Apply 1.25× multiplier to baseline TPS
- Apply 1.25× multiplier to baseline approvals
- Use peak hour revenue baselines

**Off-Peak Hours (5 PM - 9 AM ET)**
- Apply 0.7× multiplier to baseline TPS
- Apply 0.7× multiplier to baseline approvals
- Adjust revenue calculations proportionally

**Weekends**
- Apply 0.4× multiplier to all baselines
- Use weekend revenue model (if available)

### 4.3 Service-Specific Adjustments

**Payment Processing Service**
- High-value transaction multiplier: 1.5×
- Critical path for revenue: Priority 1
- Average transaction value: $2.45

**Account Approval Service**
- High-value approval multiplier: 1.3×
- Critical for new customer acquisition
- Average approval value: $3.20

**Authentication Service**
- Indirect revenue impact: Calculate based on dependent services
- Assume 80% of transactions require authentication

---

## 5. Success Rate and SLA Impact

### 5.1 Success Rate Calculation

**Formula**
```
Success Rate = (Successful Transactions / Total Attempted Transactions) × 100
```

**SLA Breach Thresholds**
- SEV-1: Success rate < 95% for > 5 minutes
- SEV-2: Success rate < 97% for > 15 minutes
- SEV-3: Success rate < 98% for > 60 minutes

### 5.2 SLA Impact Assessment

**P1 SLA (99.9% uptime)**
- Monthly downtime budget: 43.8 minutes
- Breach impact: Executive escalation + customer notifications

**P2 SLA (99.5% uptime)**
- Monthly downtime budget: 3.6 hours
- Breach impact: Customer notifications

**P3 SLA (99.0% uptime)**
- Monthly downtime budget: 7.2 hours
- Breach impact: Internal tracking only

---

## 6. User Impact Assessment Guidelines

### 6.1 User Count Estimation

**Direct Impact**
- Count of users actively experiencing errors
- Measured by: Error logs, support tickets, monitoring alerts

**Indirect Impact**
- Users unable to complete workflows due to downstream dependencies
- Measured by: Session drop rates, abandoned transactions

**Total Affected Users Formula**
```
Total Affected = Direct Impact Users + (Indirect Impact Users × 0.5)
```

### 6.2 Impact Severity Mapping

**Critical (>10,000 users)**
- Severity: SEV-1
- Communication: Public status page + direct outreach
- Executive notification: Required

**High (1,000-10,000 users)**
- Severity: SEV-2
- Communication: Status page updates
- Executive notification: Recommended

**Medium (<1,000 users)**
- Severity: SEV-3
- Communication: Internal tracking
- Executive notification: Not required

---

## 7. Examples and Case Studies

### 7.1 Example: Database Connection Pool Exhaustion (SEV-2)

**Observed Metrics**
- Current TPS: 450 (down from baseline 1200)
- Current Approvals: 380 (down from baseline 980)
- Success Rate: 62%
- Duration: 25 minutes

**Revenue Calculation**
```
Transaction Revenue Loss = (1200 - 450) × $2.45 = $1,837.50/min
Approval Revenue Loss = (980 - 380) × $3.20 = $1,920/min
Total Revenue Loss = $1,837.50 + $1,920 = $3,757.50/min
Total Impact = $3,757.50 × 25 minutes = $93,937.50
```

**SLA Impact**
- Baseline: 99.0% success rate
- Actual: 62% success rate
- Breach: YES (< 97% for > 15 minutes)
- Action: Customer notifications required

### 7.2 Example: Payment Gateway Timeout (SEV-3)

**Observed Metrics**
- Current TPS: 720 (down from baseline 850)
- Current Approvals: 650 (down from baseline 720)
- Success Rate: 91%
- Duration: 45 minutes

**Revenue Calculation**
```
Transaction Revenue Loss = (850 - 720) × $2.45 = $318.50/min
Approval Revenue Loss = (720 - 650) × $3.20 = $224/min
Total Revenue Loss = $318.50 + $224 = $542.50/min
Total Impact = $542.50 × 45 minutes = $24,412.50
```

**SLA Impact**
- Baseline: 98.5% success rate
- Actual: 91% success rate
- Breach: NO (> 98% threshold but within grace period)
- Action: Monitor for escalation

---

## 8. Data Sources and Validation

### 8.1 Authoritative Data Sources

**Transaction Metrics**
- Source: Prometheus metrics (payment_transactions_total)
- Retention: 90 days
- Update Frequency: Real-time (30-second intervals)

**Approval Metrics**
- Source: PostgreSQL analytics database
- Retention: 365 days
- Update Frequency: 1-minute aggregates

**Revenue Data**
- Source: Finance data warehouse
- Retention: 7 years (compliance requirement)
- Update Frequency: End-of-day reconciliation

### 8.2 Baseline Validation Schedule

**Monthly Review**
- Validate baseline TPS against 30-day rolling average
- Adjust for seasonal trends (holidays, month-end processing)
- Update this document if baselines drift > 10%

**Quarterly Audit**
- Finance team validates revenue correlation factors
- Compare calculated revenue loss to actual financial impact
- Adjust formulas if variance > 5%

---

## 9. Usage Requirements for SRE Agents

### 9.1 Impact Agent Requirements

**MUST Use These Baselines**
- Retrieve severity-appropriate baselines from this document
- Apply documented revenue calculation formulas
- Reference specific section numbers in output (e.g., "Section 3.2")

**MUST NOT**
- Estimate or guess baseline metrics
- Use different formulas than documented here
- Skip time-based adjustments when applicable

### 9.2 Validation and Consistency

**Cross-Agent Validation**
- Impact Agent calculations must match Mitigation Agent references
- All agents must cite "POL-SRE-002" when using business metrics
- Discrepancies must be logged and investigated

---

## 10. Policy Governance

### 10.1 Document Ownership

**Primary Owner:** SRE Lead
**Secondary Owner:** Finance Operations Manager
**Review Frequency:** Quarterly
**Approval Required For:** Baseline changes > 10%, formula modifications

### 10.2 Change Control

**Minor Updates (< 10% change)**
- Requires: SRE Lead approval
- Notification: Engineering team email
- Effective: Next business day

**Major Updates (> 10% change)**
- Requires: SRE Lead + Finance approval
- Notification: All-hands email + documentation update
- Effective: 7-day notice period

### 10.3 Compliance and Audit

**Audit Trail**
- All baseline changes must be version-controlled
- Document revision history maintained in git
- Financial audit team access required

**Regulatory Requirements**
- Revenue calculation methodology must be auditable
- Baselines must be traceable to source data
- Incident impact reports must reference policy version

---

## Appendix A: Quick Reference Table

| Metric | SEV-1 | SEV-2 | SEV-3 |
|--------|-------|-------|-------|
| **Baseline TPS** | 1650 | 1200 | 850 |
| **Baseline Approvals/min** | 1350 | 980 | 720 |
| **Baseline Revenue/min** | $6,500 | $4,200 | $2,800 |
| **Success Rate** | 99.5% | 99.0% | 98.5% |
| **Revenue per Transaction** | $2.45 | $2.45 | $2.45 |
| **Revenue per Approval** | $3.20 | $3.20 | $3.20 |
| **Max Downtime** | 5 min | 30 min | 120 min |

---

**Policy Reference:** POL-SRE-002
**Document Classification:** Internal - SRE Operations
**Last Reviewed:** November 2024
**Next Review Due:** February 2025
