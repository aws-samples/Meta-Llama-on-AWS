# Troubleshooting Runbooks

**Policy ID:** POL-SRE-003
**Version:** 1.0
**Last Updated:** November 2024
**Owner:** SRE Team

---

## 1. Database Connection Issues

### 1.1 Error Patterns

**Common Error Codes:**
- `SQLSTATE[08006]` - Connection failure
- `SQLSTATE[08001]` - Unable to establish connection
- `SQLSTATE[08004]` - Server rejected connection
- `connection pool exhausted`
- `too many connections`
- `FATAL: remaining connection slots are reserved`

### 1.2 Known Root Causes

**Distribution from Historical Incidents:**
1. **Connection pool exhaustion** (45% of cases)
   - Application not releasing connections
   - Connection leak in ORM layer
   - Insufficient pool size for traffic

2. **Database instance resource saturation** (25% of cases)
   - CPU utilization >90%
   - Memory pressure causing slowdowns
   - Disk I/O bottleneck

3. **Network connectivity issues** (15% of cases)
   - Security group misconfigurations
   - DNS resolution failures
   - Network partitions between availability zones

4. **Authentication/credential issues** (10% of cases)
   - Expired credentials
   - Password rotation not propagated
   - IAM role permission changes

5. **Database failover events** (5% of cases)
   - Automatic RDS failover
   - Manual maintenance operations
   - Multi-AZ synchronization delays

### 1.3 Diagnostic Steps

**Step 1: Check connection pool metrics**
```sql
-- PostgreSQL: View active connections
SELECT
    pid,
    usename,
    application_name,
    client_addr,
    state,
    state_change,
    query_start,
    wait_event_type,
    wait_event
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start;

-- Check total connections vs max
SELECT count(*) as current_connections,
       (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_connections;
```

**Step 2: Verify instance resource utilization**
```bash
# CloudWatch metrics to check:
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name CPUUtilization \
  --dimensions Name=DBInstanceIdentifier,Value=prod-db \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average
```

**Step 3: Test network connectivity**
```bash
# From application server
telnet db-endpoint.rds.amazonaws.com 5432
nc -zv db-endpoint.rds.amazonaws.com 5432

# DNS resolution
nslookup db-endpoint.rds.amazonaws.com
dig db-endpoint.rds.amazonaws.com
```

**Step 4: Review recent changes**
- Check deployments in last 2 hours
- Review database parameter group changes
- Verify security group modifications
- Check ORM library version updates

### 1.4 Similar Past Incidents

**INC-2024-01-15: Connection Pool Leak After ORM Upgrade**
- Root Cause: SQLAlchemy 2.0 upgrade changed connection handling
- Resolution: Downgrade to 1.4.x, add explicit connection closure
- Prevention: Connection pool monitoring alerts

**INC-2024-02-03: Database Failover Connection Storm**
- Root Cause: RDS automatic failover triggered connection retry storm
- Resolution: Implement exponential backoff in connection retry logic
- Prevention: Connection circuit breaker pattern

**INC-2024-03-12: Security Group Misconfiguration**
- Root Cause: Terraform apply removed database security group rule
- Resolution: Add rule back, implement drift detection
- Prevention: Require manual approval for security group changes

---

## 2. Payment Gateway Failures

### 2.1 Error Patterns

**Common Error Messages:**
- `Gateway Timeout (504)`
- `Bad Gateway (502)`
- `Service Temporarily Unavailable (503)`
- `Connection timeout to payment-api.example.com`
- `SSL certificate verification failed`
- `API rate limit exceeded`

### 2.2 Known Root Causes

**Distribution from Historical Incidents:**
1. **Third-party gateway degradation** (50% of cases)
   - Provider experiencing outage
   - Regional routing issues
   - Scheduled maintenance windows

2. **API rate limiting** (20% of cases)
   - Exceeded transaction quota
   - Burst traffic beyond limit
   - Missing rate limit headers handling

3. **Authentication token expiry** (15% of cases)
   - OAuth tokens not refreshed
   - API key rotation missed
   - Certificate expiration

4. **Network timeouts** (10% of cases)
   - Insufficient timeout configuration
   - Firewall blocking outbound traffic
   - DNS resolution delays

5. **Request payload issues** (5% of cases)
   - Invalid data format
   - Missing required fields
   - Encoding/charset mismatches

### 2.3 Diagnostic Steps

**Step 1: Check third-party status**
```bash
# Check status page
curl -s https://status.stripe.com/api/v2/status.json | jq .

# Test API endpoint directly
curl -v https://api.stripe.com/v1/charges \
  -u sk_test_xxx: \
  -d amount=100 \
  -d currency=usd
```

**Step 2: Review rate limit headers**
```python
# Check response headers from last successful call
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0  # ⚠️ Rate limit exhausted
X-RateLimit-Reset: 1699564800
```

**Step 3: Verify authentication**
```bash
# Test API key validity
curl https://api.stripe.com/v1/balance \
  -u sk_live_xxx:

# Check token expiration
jwt decode <token> | jq .exp
```

**Step 4: Analyze request/response patterns**
```sql
-- Check error distribution
SELECT
    error_code,
    COUNT(*) as error_count,
    AVG(response_time_ms) as avg_response_time
FROM payment_logs
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY error_code
ORDER BY error_count DESC;
```

### 2.4 Similar Past Incidents

**INC-2024-04-08: Stripe Regional Outage**
- Root Cause: Stripe's us-east-1 region degraded
- Resolution: Temporarily routed to eu-west-1 endpoint
- Prevention: Multi-region failover configuration

**INC-2024-05-20: Rate Limit Burst Exceeded**
- Root Cause: Marketing campaign caused 10x traffic spike
- Resolution: Implemented request queuing and backoff
- Prevention: Rate limit monitoring and auto-scaling

---

## 3. Application Memory Leaks

### 3.1 Error Patterns

**Common Symptoms:**
- `OutOfMemoryError`
- `GC overhead limit exceeded`
- Gradual performance degradation over time
- Container restarts with OOMKilled status
- Heap dumps showing increasing object count

### 3.2 Known Root Causes

**Distribution from Historical Incidents:**
1. **Unclosed resources** (40% of cases)
   - File handles not closed
   - Database connections leaked
   - HTTP client connections not released

2. **Cache growth without eviction** (30% of cases)
   - In-memory cache unbounded
   - LRU eviction not configured
   - TTL not set on cached objects

3. **Event listener accumulation** (15% of cases)
   - Event listeners not deregistered
   - Observer pattern memory leaks
   - Circular references in callbacks

4. **Large object retention** (10% of cases)
   - Logging entire request/response bodies
   - Holding references to large datasets
   - Session data not garbage collected

5. **Library-specific issues** (5% of cases)
   - Known memory leaks in dependencies
   - Outdated library versions
   - Framework-specific bugs

### 3.3 Diagnostic Steps

**Step 1: Analyze heap dump**
```bash
# Generate heap dump
jmap -dump:live,format=b,file=heap.bin <pid>

# Analyze with Eclipse MAT or VisualVM
# Look for:
# - Objects with high retained heap
# - Growing object collections
# - Duplicate strings/arrays
```

**Step 2: Monitor GC behavior**
```bash
# Enable GC logging
-XX:+PrintGCDetails
-XX:+PrintGCDateStamps
-Xloggc:gc.log

# Analyze GC logs
# Watch for:
# - Increasing full GC frequency
# - Decreasing reclaimed memory per GC
# - Growing old generation size
```

**Step 3: Profile memory allocation**
```python
# Python: Use memory_profiler
from memory_profiler import profile

@profile
def problematic_function():
    # Code that may leak memory
    pass
```

**Step 4: Check resource cleanup**
```python
# Verify resources are closed
with open('file.txt') as f:  # ✅ Good
    data = f.read()

# Check connection pool metrics
SELECT * FROM pg_stat_activity;  # Look for idle connections
```

### 3.4 Similar Past Incidents

**INC-2024-06-15: Redis Client Connection Leak**
- Root Cause: Redis client not using connection pooling
- Resolution: Migrate to connection pool, add max connections limit
- Prevention: Connection count monitoring alerts

**INC-2024-07-22: LRU Cache Growth**
- Root Cause: Cache had no max size, grew to 8GB over 3 days
- Resolution: Set maxsize=10000, add eviction policy
- Prevention: Memory usage alerts per service

---

## 4. API Rate Limiting and Throttling

### 4.1 Error Patterns

**Common Errors:**
- `429 Too Many Requests`
- `Rate limit exceeded`
- `Quota exhausted for service X`
- `Retry-After: 60` (header indicating cooldown)

### 4.2 Known Root Causes

1. **Traffic spike beyond quota** (50%)
2. **Retry storm amplification** (25%)
3. **Missing backoff logic** (15%)
4. **Quota misconfiguration** (10%)

### 4.3 Diagnostic Steps

**Step 1: Check current rate limit status**
```bash
# Review rate limit headers
curl -I https://api.example.com/endpoint
# Look for: X-RateLimit-Remaining, Retry-After
```

**Step 2: Analyze traffic patterns**
```sql
SELECT
    DATE_TRUNC('minute', timestamp) as minute,
    COUNT(*) as requests_per_minute
FROM api_logs
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY minute
ORDER BY requests_per_minute DESC;
```

**Step 3: Identify retry behavior**
```bash
# Check for retry loops
grep "Retrying request" app.log | wc -l

# Look for exponential backoff
grep "backoff.*ms" app.log
```

---

## 5. Service Deployment Failures

### 5.1 Error Patterns

- `ImagePullBackOff` (Kubernetes)
- `ECS task failed to start`
- `Health check failed`
- `Port already in use`
- `Configuration validation error`

### 5.2 Diagnostic Steps

**Step 1: Check deployment logs**
```bash
# Kubernetes
kubectl describe pod <pod-name>
kubectl logs <pod-name> --previous

# ECS
aws ecs describe-tasks --tasks <task-arn>
```

**Step 2: Verify image availability**
```bash
# Check image exists in registry
docker pull <image>:<tag>

# Verify image digest
aws ecr describe-images --repository-name <repo>
```

**Step 3: Review configuration**
```bash
# Validate environment variables
kubectl get configmap <name> -o yaml

# Check secrets
kubectl get secret <name> -o yaml
```

---

## 6. Cross-Service Communication Failures

### 6.1 Error Patterns

- `Connection refused`
- `Service unavailable`
- `DNS resolution failed`
- `Circuit breaker open`
- `Timeout waiting for response`

### 6.2 Diagnostic Steps

**Step 1: Test service discovery**
```bash
# Kubernetes DNS
nslookup <service-name>.<namespace>.svc.cluster.local

# Check service endpoints
kubectl get endpoints <service-name>
```

**Step 2: Verify network policies**
```bash
# Check network policies
kubectl get networkpolicies

# Test connectivity
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -- bash
curl http://<service>:<port>/health
```

**Step 3: Review circuit breaker status**
```python
# Check circuit breaker metrics
circuit_breaker.state  # CLOSED, OPEN, HALF_OPEN
circuit_breaker.failure_count
circuit_breaker.last_failure_time
```

---

## 7. CDN and Caching Issues

### 7.1 Error Patterns

- `Cache miss rate >50%`
- `Origin server overload`
- `Stale content served`
- `CloudFront 504 Gateway Timeout`

### 7.2 Diagnostic Steps

**Step 1: Check cache hit ratio**
```bash
# CloudFront metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/CloudFront \
  --metric-name CacheHitRate \
  --dimensions Name=DistributionId,Value=<dist-id>
```

**Step 2: Verify cache headers**
```bash
curl -I https://cdn.example.com/asset.js
# Look for: Cache-Control, ETag, Age
```

**Step 3: Test cache invalidation**
```bash
# Invalidate CloudFront cache
aws cloudfront create-invalidation \
  --distribution-id <id> \
  --paths "/*"
```

---

## Appendix A: Quick Reference Commands

### Database Health Check
```bash
# PostgreSQL connection count
psql -c "SELECT count(*) FROM pg_stat_activity;"

# MySQL processlist
mysql -e "SHOW PROCESSLIST;"

# Redis info
redis-cli INFO stats
```

### Application Health
```bash
# Check process memory
ps aux | grep <app-name> | awk '{print $6}'

# Java heap usage
jmap -heap <pid>

# Python memory profiling
python -m memory_profiler script.py
```

### Network Diagnostics
```bash
# Test endpoint
curl -v -w "@curl-format.txt" https://api.example.com/health

# DNS lookup
dig +trace example.com

# Trace route
traceroute api.example.com
```

---

**Policy Reference:** POL-SRE-003 Troubleshooting Runbooks
**Document Classification:** Internal - SRE Operations
**Last Reviewed:** November 2024
**Next Review Due:** February 2025
