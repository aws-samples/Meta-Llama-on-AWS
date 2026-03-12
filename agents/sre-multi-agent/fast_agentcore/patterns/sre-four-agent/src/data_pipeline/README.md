# Log Data Pipeline

This pipeline bridges Kafka-style synthetic streaming logs with the existing signals agent and orchestration workflow.

## Overview

The log data pipeline transforms unstructured log data from Kafka-style streaming sources into `ScenarioSnapshot` objects compatible with the existing orchestration system, while providing rich unstructured text for analyst agent analysis.

## Architecture

```
Kafka Logs → log_data_pipeline.py → ScenarioSnapshot → AnalystAgent → Orchestration
```

## Components

### Core Pipeline Components

1. **`log_window_processor.py`** - 15-minute window extraction from streaming logs
2. **`unstructured_aggregator.py`** - Convert structured logs to analysis text
3. **`incident_detector.py`** - Identify anomaly patterns in log windows
4. **`scenario_mapper.py`** - Convert log data to ScenarioSnapshot format
5. **`pipeline_orchestrator.py`** - Main pipeline controller and simulation

### Key Features

- ✅ **Kafka-style Log Processing** - Reads JSONL streaming logs
- ✅ **15-minute Window Simulation** - Extracts logs for specific time windows
- ✅ **Incident Detection** - Automatically detects 3 embedded incidents
- ✅ **ScenarioSnapshot Compatibility** - Maintains interface with existing orchestration
- ✅ **Unstructured Analysis** - Provides text for LLM-based analyst agent
- ✅ **Backward Compatibility** - Existing CSV scenarios still work

## Usage

### Basic Pipeline Usage

```python
from src.data_pipeline import LogDataPipeline

# Initialize pipeline
pipeline = LogDataPipeline()

# Load logs
logs = pipeline.load_logs()
print(f"Loaded {len(logs)} log entries")

# Get available windows
windows = pipeline.get_available_windows()
print(f"Available windows: {len(windows)}")

# Process a specific window
scenario = pipeline.process_window(windows[0])
print(f"Generated scenario: {scenario.metadata.key}")
```

### Demo Mode Usage

```python
from src.data_pipeline import get_kafka_scenario_snapshot

# Get scenario for specific incident time
scenario = get_kafka_scenario_snapshot("09:45")  # Payment failures
scenario = get_kafka_scenario_snapshot("11:15")  # Database issues
scenario = get_kafka_scenario_snapshot("12:45")  # Traffic surge

# Or use severity keys
scenario = get_kafka_scenario_snapshot("sev1")  # Maps to 12:45
scenario = get_kafka_scenario_snapshot("sev2")  # Maps to 11:15
scenario = get_kafka_scenario_snapshot("sev3")  # Maps to 09:45
```

### Streaming Simulation

```python
from src.data_pipeline import simulate_kafka_ingestion

# Simulate 15-minute ingestion cycles
scenarios = simulate_kafka_ingestion(max_windows=10)
print(f"Generated {len(scenarios)} scenarios")

# Check for incidents
incident_scenarios = [s for s in scenarios
                     if s.additional_sources.get("incident_details")]
print(f"Detected {len(incident_scenarios)} incidents")
```

## Data Flow

### Input: Kafka-Style Logs
- **Format**: JSONL (JSON Lines)
- **Source**: `data/kafka_style/streaming_logs.jsonl`
- **Duration**: 5 hours (09:00 - 14:00 UTC)
- **Entries**: 1,344 log entries
- **Services**: 10 microservices

### Processing: 15-Minute Windows
- **Window Size**: 15 minutes (configurable)
- **Overlap Detection**: Handles time boundary edge cases
- **Service Correlation**: Groups logs by service
- **Anomaly Detection**: Pattern matching for incidents

### Output: ScenarioSnapshot Objects
- **Interface**: Compatible with existing orchestration
- **Monitoring**: Includes unstructured log analysis text
- **Evidence**: Raw log messages preserved
- **Metrics**: Synthetic metrics for backward compatibility

## Embedded Incidents

The synthetic dataset contains 3 embedded incidents:

### 1. Payment Processing Failures (SEV-3)
- **Time**: 09:45 - 10:15 UTC
- **Cause**: External payment gateway issues
- **Symptoms**: High error rate, processing timeouts
- **Detection**: ERROR logs in payment-processor service

### 2. Database Connection Issues (SEV-2)
- **Time**: 11:15 - 11:45 UTC
- **Cause**: Connection pool exhaustion
- **Symptoms**: Query timeouts, cascading failures
- **Detection**: CONNECTION_TIMEOUT errors across services

### 3. Traffic Surge Overload (SEV-1)
- **Time**: 12:45 - 13:15 UTC
- **Cause**: Unexpected traffic spike
- **Symptoms**: System-wide timeouts, resource exhaustion
- **Detection**: High latency, HTTP 504 responses

## Integration with Existing System

### ScenarioSnapshot Compatibility
The pipeline generates `ScenarioSnapshot` objects with the same interface as the existing CSV-based system:

```python
# Both approaches produce compatible objects
csv_scenario = load_snapshot("sev2")        # Original CSV approach
kafka_scenario = get_kafka_scenario_snapshot("11:15")  # New Kafka approach

# Same interface
assert type(csv_scenario) == type(kafka_scenario)
assert hasattr(kafka_scenario, 'metadata')
assert hasattr(kafka_scenario, 'monitoring')
assert hasattr(kafka_scenario, 'additional_sources')
```

### Enhanced Data for Analyst Agent
The Kafka-style scenarios provide additional unstructured data:

```python
scenario = get_kafka_scenario_snapshot("09:45")

# Unstructured log analysis for LLM processing
log_analysis = scenario.monitoring["log_analysis"]
incident_narrative = scenario.monitoring["incident_narrative"]

# Rich evidence and context
incident_details = scenario.additional_sources["incident_details"]
raw_logs = scenario.additional_sources["raw_logs"]
```

## Demo Scripts

### Run Full Demo
```bash
python3 src/data_pipeline/demo.py
```

### Run Specific Demo Components
```bash
python3 src/data_pipeline/demo.py basic      # Basic pipeline functionality
python3 src/data_pipeline/demo.py scenarios  # Scenario generation
python3 src/data_pipeline/demo.py streaming  # Streaming simulation
python3 src/data_pipeline/demo.py analysis   # Unstructured analysis
python3 src/data_pipeline/demo.py integration # Orchestrator integration
python3 src/data_pipeline/demo.py comparison  # CSV vs Kafka comparison
python3 src/data_pipeline/demo.py list       # Available scenarios
```

## Performance Considerations

- **Log Loading**: ~1,344 entries loaded in <1 second
- **Window Processing**: 15-minute windows processed in <2 seconds
- **Incident Detection**: Pattern matching completes in <5 seconds
- **Memory Usage**: ~50MB for full dataset in memory
- **Scalability**: Designed for streaming ingestion at production scale

## Future Enhancements

### Real-Time Mode
```python
# Future: OpenSearch integration
pipeline = LogDataPipeline(mode="real_time")
pipeline.connect_to_opensearch(endpoint="https://opensearch.example.com")

# Stream processing
for scenario in pipeline.stream_analysis():
    if scenario.has_incidents():
        await trigger_orchestration(scenario)
```

### Advanced Analytics
- **ML-based Anomaly Detection**: Replace pattern matching with ML models
- **Cross-Service Correlation**: Detect distributed system failures
- **Predictive Alerting**: Identify incidents before they escalate
- **Auto-Remediation**: Trigger automated responses

## Testing

### Unit Tests
```bash
# Test individual components
python3 -m pytest tests/data_pipeline/test_log_window_processor.py
python3 -m pytest tests/data_pipeline/test_incident_detector.py
python3 -m pytest tests/data_pipeline/test_scenario_mapper.py
```

### Integration Tests
```bash
# Test full pipeline
python3 -m pytest tests/integration/test_log_data_pipeline.py
```

### Manual Testing
```bash
# Quick validation
python3 -c "
from src.data_pipeline import LogDataPipeline
pipeline = LogDataPipeline()
assert len(pipeline.load_logs()) == 1344
print('✅ Pipeline validation passed')
"
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Timezone Issues**: All timestamps are UTC-aware
3. **Memory Issues**: Large datasets may require streaming processing
4. **Performance**: Consider caching for repeated analysis

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = LogDataPipeline()
logs = pipeline.load_logs()  # Will show debug info
```

## API Reference

See individual module docstrings for detailed API documentation:
- [`log_window_processor.py`](./log_window_processor.py)
- [`incident_detector.py`](./incident_detector.py)
- [`scenario_mapper.py`](./scenario_mapper.py)
- [`pipeline_orchestrator.py`](./pipeline_orchestrator.py)

---

**Status**: ✅ **Ready for Integration**
**Compatibility**: ✅ **Maintains ScenarioSnapshot Interface**
**Testing**: ✅ **Basic Validation Complete**
**Documentation**: ✅ **Comprehensive API Coverage**