# Bank Log Generator Lambda

Lambda function that generates realistic banking system logs and publishes them to CloudWatch Logs at `/aws/banking/system-logs`.

## Overview

Generates synthetic logs from 5 banking microservices with embedded incidents:
- auth-service
- payments-service
- accounts-service
- trading-service
- notification-service

Logs include 3 embedded incidents (unlabeled) that SRE agents can detect and analyze.

## Structure

```
lambda_log_generator/
├── lambda_function.py      # Lambda handler
├── cdk/
│   ├── app.py              # CDK app entry point
│   ├── cdk.json            # CDK config
│   ├── lambda_stack.py     # CDK stack (Lambda + API Gateway)
│   └── requirements.txt    # CDK dependencies
└── README.md
```

## Deployment

This component is deployed automatically by the root `deploy.sh` script (Step 3/4). No manual deployment needed.

## Lambda Parameters

The Lambda accepts these event parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hours`   | 1       | Duration of logs to generate |
| `rate`    | 20      | Events per minute per service |
| `seed`    | random  | Random seed for reproducibility |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_GROUP_NAME` | `/aws/banking/system-logs` | CloudWatch log group |
| `LOG_STREAM_PREFIX` | `service` | Log stream name prefix |

## Example Log Entry

```json
{
  "timestamp": "2025-10-09T09:15:23.456",
  "service": "payments-service",
  "host": "pay-1.bank.local",
  "environment": "production",
  "region": "us-west-2",
  "event_type": "request",
  "http": {
    "method": "POST",
    "path": "/api/payments/request",
    "status_code": 200,
    "response_time_ms": 85
  },
  "message": "payments-service request handled with status 200 in 85ms"
}
```
