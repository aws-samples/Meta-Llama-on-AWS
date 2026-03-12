# Known Failure Patterns

**Policy ID:** POL-SRE-004
**Version:** 1.0
**Last Updated:** November 2024
**Owner:** SRE Team & Incident Response

---

## 1. Overview

This document catalogs recurring failure patterns observed in production systems, their root cause distributions, and diagnostic approaches. Use this as a reference during incident response to quickly identify likely root causes based on symptom patterns.

---

## 2. Payment Processing Failures

### 2.1 Symptom Pattern

**Observable Indicators:**
- Error rate spike in `payment-processor` service
- Gateway timeout errors (504) from payment API
- Transaction success rate drops below 95%
- Customer reports of "payment not processing"
- Increased latency in `/api/payments/process` endpoint

**Typical Timeline:**
- Onset: Sudden (within 2-5 minutes)
- Duration: 15-45 minutes average
- Peak Impact: 10-15 minutes after onset

### 2.2 Root Cause Distribution

Based on 47 incidents (Jan-Oct 2024):

| Root Cause | Frequency | MTTR | Severity |
|------------|-----------|------|----------|
| **Third-party payment gateway degradation** | 45% (21 incidents) | 35 min | SEV-2 |
| **Database connection pool exhaustion** | 30% (14 incidents) | 25 min | SEV-2 |
| **API rate limiting from payment provider** | 15% (7 incidents) | 20 min | SEV-3 |
| **Authentication token expiry** | 10% (5 incidents) | 15 min | SEV-3 |

### 2.3 Diagnostic Questions

**Priority 1 (Check First):**
1. Is the third-party payment provider status page reporting issues?
   - Check: https://status.stripe.com, https://status.paypal.com
   - If YES → 45% likely external issue, implement failover

2. Are database connection pools saturated?
   - Check: `SELECT count(*) FROM pg_stat_activity WHERE state='active'`
   - If >80% capacity → Connection pool issue

**Priority 2 (Secondary Checks):**
3. Have we exceeded API rate limits?
   - Check response headers: `X-RateLimit-Remaining`
   - Review traffic spike in last 30 minutes

4. When were payment API credentials last rotated?
   - Check credential expiration date
   - Test authentication with direct API call

### 2.4 Historical Incident References

**INC-2024-04-08: Stripe Regional Outage**
- **Symptoms:** 504 errors, 100% failure rate
- **Root Cause:** Stripe us-east-1 region degraded
- **Resolution Time:** 42 minutes
- **Resolution:** Failover to eu-west-1 endpoint
- **Prevention:** Multi-region routing configuration

**INC-2024-05-20: Payment Rate Limit Breach**
- **Symptoms:** 429 errors, 60% success rate
- **Root Cause:** Marketing campaign caused 10x traffic spike
- **Resolution Time:** 18 minutes
- **Resolution:** Implemented request queuing with backoff
- **Prevention:** Rate limit monitoring alerts

**INC-2024-08-15: Database Connection Pool Leak**
- **Symptoms:** Gradual degradation over 20 minutes
- **Root Cause:** Payment service not releasing connections
- **Resolution Time:** 28 minutes
- **Resolution:** Service restart + connection pool tuning
- **Prevention:** Connection pool monitoring

---

## 3. Database Performance Degradation

### 3.1 Symptom Pattern

**Observable Indicators:**
- Query response time >500ms (baseline: 50-100ms)
- Database CPU utilization >80%
- Connection pool exhaustion warnings
- Read replica lag >5 seconds
- Application timeout errors

**Typical Timeline:**
- Onset: Gradual (15-30 minute degradation) or sudden (instant)
- Duration: 20-90 minutes average
- Recovery: Usually requires intervention

### 3.2 Root Cause Distribution

Based on 38 incidents (Jan-Oct 2024):

| Root Cause | Frequency | MTTR | Pattern |
|------------|-----------|------|---------|
| **Unoptimized query introduced in deployment** | 35% (13 incidents) | 45 min | Gradual |
| **Connection pool size insufficient for traffic** | 25% (10 incidents) | 30 min | Sudden |
| **Expensive report/analytics query** | 20% (8 incidents) | 20 min | Sudden |
| **Database instance CPU saturation** | 15% (6 incidents) | 60 min | Gradual |
| **Replication lag cascade** | 5% (2 incidents) | 90 min | Gradual |

### 3.3 Diagnostic Questions

**Priority 1:**
1. Was there a deployment in the last 2 hours?
   - Check git log: `git log --since="2 hours ago"`
   - If YES → 35% likely new query issue

2. What are the slowest running queries right now?
   ```sql
   SELECT pid, now() - query_start AS duration, query
   FROM pg_stat_activity
   WHERE state = 'active' AND now() - query_start > interval '1 second'
   ORDER BY duration DESC;
   ```

**Priority 2:**
3. Is this a scheduled batch job or report time?
   - Check cron jobs: `crontab -l`
   - Review analytics dashboard usage

4. Has traffic increased significantly?
   - Check request rate: CloudWatch `RequestCount` metric
   - Compare to 7-day average

### 3.4 Known Query Patterns

**Problematic Pattern 1: N+1 Query**
```python
# BAD: Generates N queries
for user in users:
    user.orders.count()  # Separate query per user

# GOOD: Single query with join
users = User.objects.prefetch_related('orders')
```

**Problematic Pattern 2: Missing Index**
```sql
-- Slow: No index on created_at
SELECT * FROM orders WHERE created_at > '2024-01-01';

-- Fix: Add index
CREATE INDEX idx_orders_created_at ON orders(created_at);
```

**Problematic Pattern 3: Large IN clause**
```sql
-- Slow: IN with 10,000 values
SELECT * FROM products WHERE id IN (1,2,3,...,10000);

-- Better: Use temp table or array
SELECT * FROM products WHERE id = ANY(ARRAY[1,2,3,...]);
```

---

## 4. Memory Leaks and OOM Errors

### 4.1 Symptom Pattern

**Observable Indicators:**
- Gradual memory growth over hours/days
- Container restarts with `OOMKilled` status
- Increasing GC frequency
- Performance degradation over time
- Application unresponsiveness

**Typical Timeline:**
- Onset: Very gradual (hours to days)
- Detection: Usually after restart/crash
- Resolution: 45-120 minutes (requires code fix)

### 4.2 Root Cause Distribution

Based on 24 incidents (Jan-Oct 2024):

| Root Cause | Frequency | Detection Time | Fix Time |
|------------|-----------|----------------|----------|
| **Unclosed database connections** | 40% (10 incidents) | 4-8 hours | 30 min |
| **Unbounded in-memory cache** | 30% (7 incidents) | 12-24 hours | 60 min |
| **Event listener accumulation** | 15% (4 incidents) | 2-6 hours | 90 min |
| **Large object retention in sessions** | 10% (2 incidents) | 6-12 hours | 45 min |
| **Third-party library memory leak** | 5% (1 incident) | Variable | 120 min |

### 4.3 Diagnostic Steps

**Check 1: Memory growth rate**
```bash
# Monitor container memory over 5 minutes
watch -n 30 'docker stats --no-stream | grep <container-name>'

# CloudWatch: MemoryUtilization metric trend
```

**Check 2: Object retention**
```python
# Python: Check object counts
import gc
from collections import Counter
print(Counter(type(obj).__name__ for obj in gc.get_objects()).most_common(10))
```

**Check 3: Connection leaks**
```sql
-- PostgreSQL: Connections not released
SELECT count(*), state, wait_event_type
FROM pg_stat_activity
GROUP BY state, wait_event_type;
```

---

## 5. API Gateway and Load Balancer Issues

### 5.1 Symptom Pattern

**Observable Indicators:**
- 502 Bad Gateway errors
- 504 Gateway Timeout errors
- Intermittent connection failures
- Specific routes failing while others work
- Regional outage reports

**Typical Timeline:**
- Onset: Immediate
- Impact: Can affect 100% of traffic
- Resolution: 10-30 minutes

### 5.2 Root Cause Distribution

Based on 19 incidents (Jan-Oct 2024):

| Root Cause | Frequency | User Impact | MTTR |
|------------|-----------|-------------|------|
| **Backend service unhealthy** | 50% (10 incidents) | Regional | 20 min |
| **Load balancer configuration error** | 25% (5 incidents) | Total | 25 min |
| **SSL certificate expiration** | 15% (3 incidents) | Total | 15 min |
| **DDoS or traffic spike** | 10% (2 incidents) | Partial | 40 min |

### 5.3 Diagnostic Questions

1. Are backend health checks passing?
   ```bash
   # Check target group health
   aws elbv2 describe-target-health --target-group-arn <arn>
   ```

2. Is the SSL certificate valid?
   ```bash
   echo | openssl s_client -connect api.example.com:443 2>/dev/null | openssl x509 -noout -dates
   ```

3. What's the current traffic rate?
   ```bash
   # CloudWatch: RequestCount per minute
   # Compare to historical baseline
   ```

---

## 6. Authentication and Authorization Failures

### 6.1 Symptom Pattern

**Observable Indicators:**
- 401 Unauthorized errors
- 403 Forbidden errors
- "Invalid token" or "Token expired" messages
- Users logged out unexpectedly
- SSO integration failures

**Typical Timeline:**
- Onset: Can be gradual or sudden
- Scope: Can affect all users or specific cohort
- Resolution: 15-45 minutes

### 6.2 Root Cause Distribution

Based on 16 incidents (Jan-Oct 2024):

| Root Cause | Frequency | Scope | MTTR |
|------------|-----------|-------|------|
| **JWT secret rotation without grace period** | 40% (6 incidents) | All users | 20 min |
| **OAuth token refresh failure** | 30% (5 incidents) | Specific provider | 30 min |
| **Session store (Redis) unavailable** | 20% (3 incidents) | All users | 25 min |
| **RBAC policy misconfiguration** | 10% (2 incidents) | User subset | 40 min |

### 6.3 Diagnostic Steps

**Check 1: Token validity**
```bash
# Decode JWT token
echo <token> | jwt decode -

# Check expiration
jwt decode <token> | jq .exp
```

**Check 2: Session store**
```bash
# Redis health
redis-cli ping

# Check session count
redis-cli DBSIZE
```

**Check 3: OAuth provider status**
```bash
# Test OAuth endpoint
curl -v https://oauth.provider.com/.well-known/openid-configuration
```

---

## 7. Deployment and Rollout Failures

### 7.1 Symptom Pattern

**Observable Indicators:**
- Error rate spike immediately after deployment
- New version containers crash-looping
- Health checks failing for new pods
- Rollback triggered automatically
- Configuration validation errors

**Typical Timeline:**
- Onset: 0-5 minutes after deployment
- Detection: Immediate (automated alerts)
- Resolution: 10-30 minutes (rollback or fix)

### 7.2 Root Cause Distribution

Based on 31 incidents (Jan-Oct 2024):

| Root Cause | Frequency | Prevention | MTTR |
|------------|-----------|------------|------|
| **Breaking API change** | 35% (11 incidents) | Integration tests | 25 min |
| **Missing environment variable** | 30% (9 incidents) | Deployment checks | 15 min |
| **Database migration failure** | 20% (6 incidents) | Migration tests | 45 min |
| **Dependency version conflict** | 10% (3 incidents) | Lock files | 30 min |
| **Container image build failure** | 5% (2 incidents) | CI checks | 20 min |

### 7.3 Rollback Criteria

**Automatic Rollback Triggers:**
- Error rate >10% higher than pre-deployment baseline
- Health check failure rate >25%
- Response time >2x baseline for 5 minutes
- Zero successful requests in 2-minute window

**Manual Rollback Indicators:**
- Critical functionality broken
- Data corruption detected
- Security vulnerability introduced
- Customer escalations >5 in 10 minutes

---

## 8. Cross-Service Communication Failures

### 8.1 Symptom Pattern

**Observable Indicators:**
- `Connection refused` errors
- `Name resolution failed` (DNS errors)
- Circuit breaker opened
- Service mesh errors
- Timeout cascades

**Typical Timeline:**
- Onset: Sudden
- Propagation: Can cascade to dependent services
- Resolution: 20-60 minutes

### 8.2 Root Cause Distribution

Based on 22 incidents (Jan-Oct 2024):

| Root Cause | Frequency | Impact | MTTR |
|------------|-----------|--------|------|
| **Service discovery failure** | 40% (9 incidents) | Multiple services | 35 min |
| **Network policy blocking traffic** | 30% (7 incidents) | Specific routes | 25 min |
| **Downstream service degradation** | 20% (4 incidents) | Cascade | 45 min |
| **DNS resolution timeout** | 10% (2 incidents) | All services | 30 min |

### 8.3 Diagnostic Commands

```bash
# Test service DNS
nslookup payment-service.default.svc.cluster.local

# Check service endpoints
kubectl get endpoints payment-service

# Test connectivity
kubectl run -it --rm debug --image=nicolaka/netshoot -- \
  curl payment-service:8080/health

# Review network policies
kubectl get networkpolicies -A
```

---

## Appendix A: Incident Pattern Correlation Matrix

| Primary Symptom | Likely Root Causes (in order) | Diagnostic Priority |
|----------------|-------------------------------|-------------------|
| 504 Gateway Timeout | 1. Backend slow<br>2. Connection pool<br>3. Load balancer | Backend health → Connections → LB config |
| Database errors | 1. Connection pool<br>2. Slow query<br>3. Resource saturation | Connections → Query log → CPU/Memory |
| Memory growth | 1. Connection leak<br>2. Cache unbounded<br>3. Event listeners | Connection count → Heap dump → GC log |
| Authentication failures | 1. Token expiry<br>2. Session store<br>3. OAuth issue | Token validity → Redis → OAuth |
| Payment failures | 1. Provider outage<br>2. Rate limit<br>3. DB pool | Status page → Rate headers → DB connections |

---

## Appendix B: Quick Reference - Top 10 Incidents by Frequency

1. **Database connection pool exhaustion** - 23 incidents (2024 YTD)
2. **Third-party API degradation** - 21 incidents
3. **Deployment breaking changes** - 18 incidents
4. **Unoptimized queries** - 13 incidents
5. **Memory leaks (connection-related)** - 10 incidents
6. **API rate limiting** - 9 incidents
7. **Service discovery failures** - 9 incidents
8. **Network policy misconfigurations** - 7 incidents
9. **Cache unbounded growth** - 7 incidents
10. **OAuth/SSO provider issues** - 6 incidents

---

**Policy Reference:** POL-SRE-004 Known Failure Patterns
**Document Classification:** Internal - SRE Operations
**Data Source:** Incident tracking system (2024-01-01 to 2024-10-31)
**Last Reviewed:** November 2024
**Next Review Due:** February 2025
