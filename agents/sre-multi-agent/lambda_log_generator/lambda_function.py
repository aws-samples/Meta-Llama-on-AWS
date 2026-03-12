#!/usr/bin/env python3
"""
Lambda function to generate bank logs and publish to CloudWatch Logs.

This Lambda generates realistic banking system logs using the BankLogGenerator
and publishes them to CloudWatch Logs for analysis by SRE agents.
"""

import json
import os
import random
import uuid
from datetime import datetime, timedelta
from typing import Any

import boto3

# CloudWatch Logs client
logs_client = boto3.client('logs')

# Configuration from environment
LOG_GROUP_NAME = os.environ.get('LOG_GROUP_NAME', '/aws/banking/system-logs')
LOG_STREAM_PREFIX = os.environ.get('LOG_STREAM_PREFIX', 'service')


class ServiceCfg:
    def __init__(self, name: str, topic: str, base_latency_ms: int, base_error_rate: float, hosts: list[str]):
        self.name = name
        self.topic = topic
        self.base_latency_ms = base_latency_ms
        self.base_error_rate = base_error_rate
        self.hosts = hosts


class BankLogGenerator:
    def __init__(
        self,
        start_time: datetime,
        hours: int = 5,
        rate_per_minute_per_service: int = 20,
        seed: int = 42,
    ):
        self.start_time = start_time
        self.end_time = start_time + timedelta(hours=hours)
        self.hours = hours
        self.rate = rate_per_minute_per_service
        random.seed(seed)

        # Define the 5 services
        self.services: dict[str, ServiceCfg] = {
            "auth-service": ServiceCfg(
                name="auth-service",
                topic="auth-events",
                base_latency_ms=30,
                base_error_rate=0.01,
                hosts=[f"auth-{i}.bank.local" for i in range(1, 4)],
            ),
            "payments-service": ServiceCfg(
                name="payments-service",
                topic="payments-events",
                base_latency_ms=80,
                base_error_rate=0.01,
                hosts=[f"pay-{i}.bank.local" for i in range(1, 5)],
            ),
            "accounts-service": ServiceCfg(
                name="accounts-service",
                topic="accounts-events",
                base_latency_ms=25,
                base_error_rate=0.005,
                hosts=[f"acct-{i}.bank.local" for i in range(1, 3)],
            ),
            "trading-service": ServiceCfg(
                name="trading-service",
                topic="trading-events",
                base_latency_ms=60,
                base_error_rate=0.008,
                hosts=[f"trade-{i}.bank.local" for i in range(1, 5)],
            ),
            "notification-service": ServiceCfg(
                name="notification-service",
                topic="notification-events",
                base_latency_ms=20,
                base_error_rate=0.004,
                hosts=[f"notify-{i}.bank.local" for i in range(1, 3)],
            ),
        }

        self.partitions = {cfg.topic: list(range(3)) for cfg in self.services.values()}
        self.offsets = {cfg.topic: 0 for cfg in self.services.values()}

        # Incident windows
        self.incidents = [
            {
                "id": "INC-001",
                "title": "Auth cache outage → latency spike & 5xx",
                "services": ["auth-service"],
                "start": self.start_time + timedelta(minutes=30),
                "end": self.start_time + timedelta(minutes=45),
                "effects": {
                    "latency_mult": 4.0,
                    "err_boost": 0.25,
                    "cache_timeouts": True,
                },
            },
            {
                "id": "INC-002",
                "title": "Payments schema mismatch → DLQ growth",
                "services": ["payments-service", "notification-service"],
                "start": self.start_time + timedelta(minutes=75),
                "end": self.start_time + timedelta(minutes=100),
                "effects": {
                    "dlq_growth": True,
                    "err_boost": 0.18,
                    "schema_expected": "v6",
                    "schema_observed": "v5",
                },
            },
            {
                "id": "INC-003",
                "title": "Trading CPU/memory saturation → 5xx/timeouts",
                "services": ["trading-service"],
                "start": self.start_time + timedelta(minutes=140),
                "end": self.start_time + timedelta(minutes=155),
                "effects": {
                    "latency_mult": 3.0,
                    "err_boost": 0.22,
                    "backpressure": True,
                },
            },
        ]

        self.event_types = [
            "request",
            "db_query",
            "cache_op",
            "external_call",
            "kafka_produce",
            "kafka_consume",
            "job_run",
        ]
        self.http_methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        self.payment_providers = ["visa", "mastercard", "amex", "ach"]
        self.currencies = ["USD", "EUR", "GBP"]

    def _incident_effects(self, service: str, ts: datetime) -> dict[str, Any]:
        eff: dict[str, Any] = {}
        for inc in self.incidents:
            if service in inc["services"] and inc["start"] <= ts <= inc["end"]:
                eff.update(inc["effects"])
        return eff

    def _latency_ms(self, svc: ServiceCfg, effects: dict[str, Any]) -> int:
        base = svc.base_latency_ms
        mult = effects.get("latency_mult", 1.0)
        lat = base * mult
        lat = max(1, int(random.gauss(lat, lat * 0.25 if lat > 4 else 1)))
        return lat

    def _status_code(self, svc: ServiceCfg, effects: dict[str, Any]) -> int:
        p_err = svc.base_error_rate + effects.get("err_boost", 0.0)
        if random.random() < p_err:
            return random.choice([500, 503, 504, 429])
        return random.choice([200, 200, 200, 201, 202, 204])

    def _add_kafka(self, log: dict[str, Any], topic: str, dlq: bool = False) -> None:
        base_topic = topic
        if dlq:
            topic = f"{topic}.DLQ"
        partition = random.choice(self.partitions[base_topic])
        offset = self.offsets[base_topic]
        self.offsets[base_topic] += 1
        log["kafka"] = {
            "topic": topic,
            "partition": partition,
            "offset": offset,
            "key": uuid.uuid4().hex,
        }

    def _base_record(self, ts: datetime, svc: ServiceCfg) -> dict[str, Any]:
        region = random.choice(["us-east-1", "us-west-2", "eu-west-1"])
        az = random.choice(["a", "b", "c"])
        return {
            "timestamp": ts.isoformat(),
            "service": svc.name,
            "host": random.choice(svc.hosts),
            "environment": "production",
            "region": region,
            "availability_zone": f"{region}{az}",
            "instance_id": f"i-{uuid.uuid4().hex[:12]}",
            "container_id": f"{svc.name}-{uuid.uuid4().hex[:8]}",
            "trace_id": uuid.uuid4().hex,
            "span_id": uuid.uuid4().hex[:16],
            "request_id": uuid.uuid4().hex[:12],
        }

    def _http_block(
        self, svc: ServiceCfg, ts: datetime, effects: dict[str, Any]
    ) -> dict[str, Any]:
        method = random.choice(self.http_methods)
        namespace = svc.name.split("-")[0]
        path = f"/api/{namespace}/{random.choice(self.event_types)}"
        status = self._status_code(svc, effects)
        return {
            "method": method,
            "path": path,
            "status_code": status,
            "response_time_ms": self._latency_ms(svc, effects),
            "client_ip": f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            "user_agent": random.choice(
                [
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0) AppleWebKit/605.1.15",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/118.0.0.0",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15",
                    "MobileApp/4.5.0 (iOS)",
                    "PaymentSDK/3.2.1 (Android 13)",
                ]
            ),
            "request_size_bytes": random.randint(200, 2000),
            "response_size_bytes": random.randint(200, 8000),
        }

    def _attach_domain_noise(
        self, rec: dict[str, Any], svc: ServiceCfg, effects: dict[str, Any]
    ) -> None:
        level = (
            "ERROR"
            if rec["http"]["status_code"] >= 400
            else random.choices(["DEBUG", "INFO", "WARN"], weights=[1, 6, 2], k=1)[0]
        )
        rec["level"] = level

        if svc.name == "payments-service":
            rec["payment"] = {
                "amount_cents": random.randint(199, 250000),
                "currency": random.choice(self.currencies),
                "provider": random.choice(self.payment_providers),
            }
            if "dlq_growth" in effects and random.random() < 0.35:
                rec["error_detail"] = random.choice(
                    [
                        "SchemaValidationError: field amount_cents missing",
                        "Avro schema mismatch",
                        "SignatureVerificationFailed",
                    ]
                )
                rec["schema"] = {
                    "expected": effects.get("schema_expected", "v6"),
                    "observed": effects.get("schema_observed", "v5"),
                }

        if svc.name == "auth-service" and effects.get("cache_timeouts"):
            rec["cache"] = {
                "endpoint": "redis://cache-auth:6379",
                "result": random.choice(["MISS", "TIMEOUT", "ERROR", "MISS", "MISS"]),
            }

        if svc.name == "trading-service" and effects.get("backpressure"):
            rec["system"] = {
                "cpu_percent": round(random.uniform(88, 99), 1),
                "memory_mb": random.randint(7800, 8200),
                "queue_depth": random.randint(1200, 2000),
            }

        if svc.name == "notification-service" and "dlq_growth" in effects:
            rec["dlq_depth"] = random.randint(500, 5000)
            rec["kafka_consume"] = {"from_topic": "payments-events.DLQ"}

    def _maybe_route_dlq(self, svc: ServiceCfg, effects: dict[str, Any]) -> bool:
        if "dlq_growth" in effects:
            if svc.name in ("payments-service", "notification-service"):
                return random.random() < 0.25
        return False

    def _generate_event(self, ts: datetime, svc: ServiceCfg) -> dict[str, Any]:
        effects = self._incident_effects(svc.name, ts)
        rec = self._base_record(ts, svc)
        rec["event_type"] = random.choice(self.event_types)
        rec["http"] = self._http_block(svc, ts, effects)
        rec["message"] = (
            f"{svc.name} {rec['event_type']} handled with status {rec['http']['status_code']} in {rec['http']['response_time_ms']}ms"
        )
        self._attach_domain_noise(rec, svc, effects)

        to_dlq = self._maybe_route_dlq(svc, effects)
        self._add_kafka(rec, svc.topic, dlq=to_dlq)
        return rec

    def generate_logs(self, minutes: int = 60) -> list[dict[str, Any]]:
        """Generate logs for specified duration in minutes."""
        logs = []
        for minute in range(minutes):
            minute_start = self.start_time + timedelta(minutes=minute)
            for svc in self.services.values():
                n = max(1, int(random.gauss(self.rate, self.rate * 0.15)))
                for _ in range(n):
                    ts = minute_start + timedelta(
                        seconds=random.randint(0, 59),
                        milliseconds=random.randint(0, 999),
                    )
                    rec = self._generate_event(ts, svc)
                    logs.append(rec)
        return logs


def ensure_log_group_exists(log_group_name: str) -> None:
    """Create log group if it doesn't exist."""
    try:
        logs_client.create_log_group(logGroupName=log_group_name)
        print(f"Created log group: {log_group_name}")
    except logs_client.exceptions.ResourceAlreadyExistsException:
        print(f"Log group already exists: {log_group_name}")


def ensure_log_stream_exists(log_group_name: str, log_stream_name: str) -> None:
    """Create log stream if it doesn't exist."""
    try:
        logs_client.create_log_stream(
            logGroupName=log_group_name,
            logStreamName=log_stream_name
        )
        print(f"Created log stream: {log_stream_name}")
    except logs_client.exceptions.ResourceAlreadyExistsException:
        print(f"Log stream already exists: {log_stream_name}")


def publish_logs_to_cloudwatch(logs: list[dict[str, Any]], log_group_name: str, log_stream_name: str) -> None:
    """Publish logs to CloudWatch Logs."""
    if not logs:
        print("No logs to publish")
        return
    
    # Sort logs by timestamp
    logs.sort(key=lambda x: x['timestamp'])
    
    # Convert to CloudWatch log events
    log_events = []
    for log in logs:
        timestamp_ms = int(datetime.fromisoformat(log['timestamp']).timestamp() * 1000)
        log_events.append({
            'timestamp': timestamp_ms,
            'message': json.dumps(log)
        })
    
    # Publish in batches (CloudWatch limit: 10,000 events or 1MB per request)
    batch_size = 1000
    for i in range(0, len(log_events), batch_size):
        batch = log_events[i:i + batch_size]
        try:
            logs_client.put_log_events(
                logGroupName=log_group_name,
                logStreamName=log_stream_name,
                logEvents=batch
            )
            print(f"Published {len(batch)} log events to CloudWatch")
        except Exception as e:
            print(f"Error publishing batch: {e}")
            raise


def lambda_handler(event, context):
    """
    Lambda handler to generate and publish bank logs to CloudWatch.
    
    Event parameters:
    - hours: Duration in hours (default: 1)
    - rate: Events per minute per service (default: 20)
    - seed: Random seed (default: 42)
    """
    try:
        # Parse event parameters
        hours = event.get('hours', 1)
        rate = event.get('rate', 20)
        seed = event.get('seed', random.randint(1, 10000))
        
        print(f"Generating logs: hours={hours}, rate={rate}, seed={seed}")
        
        # Generate logs starting from now
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        generator = BankLogGenerator(
            start_time=start_time,
            hours=hours,
            rate_per_minute_per_service=rate,
            seed=seed
        )
        
        # Generate logs
        minutes = hours * 60
        logs = generator.generate_logs(minutes=minutes)
        
        print(f"Generated {len(logs)} log events")
        
        # Ensure log group and stream exist
        log_stream_name = f"{LOG_STREAM_PREFIX}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ensure_log_group_exists(LOG_GROUP_NAME)
        ensure_log_stream_exists(LOG_GROUP_NAME, log_stream_name)
        
        # Publish to CloudWatch
        publish_logs_to_cloudwatch(logs, LOG_GROUP_NAME, log_stream_name)
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                'Access-Control-Allow-Methods': 'POST,OPTIONS'
            },
            'body': json.dumps({
                'message': 'Logs published successfully',
                'log_group': LOG_GROUP_NAME,
                'log_stream': log_stream_name,
                'log_count': len(logs),
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': generator.end_time.isoformat()
                }
            })
        }
        
    except Exception as e:
        print(f"Error in lambda_handler: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                'Access-Control-Allow-Methods': 'POST,OPTIONS'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }
