# Payments API Mitigation Runbook

This runbook summarises the standard mitigation flow used by the demo agents.

## 1. Trigger Conditions
- Payments API latency or error rate breaching SEV-3+ thresholds
- Infra metrics showing pod restarts or sustained memory pressure

## 2. Immediate Actions
1. Drain and recycle impacted pods (node-b)
2. Enable autoscaler guardrails and monitor memory for 30 minutes
3. Review latest deploys and roll back if correlated with the incident window

## 3. Validation
- Confirm latency p95 < 250 ms for 15 minutes
- Ensure pod restarts fall back to baseline (< 1 per hour)
- Verify payments success rate returns to contractual threshold

## 4. Communication
- Provide internal update via incident channel
- Prepare external status page message if customer impact persists beyond 30 minutes

This file is referenced by the Mitigation agent to provide context links in the console UI.
