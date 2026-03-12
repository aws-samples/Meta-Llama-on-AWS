"""Demo functions for the four-agent orchestration system."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from .scenario_loader import ScenarioSnapshot, load_snapshot


@dataclass
class DemoRun:
    """Results from a demo scenario run."""
    scenario: str
    duration_seconds: float
    success: bool
    agents_used: List[str]
    output_summary: str


# Available scenarios for demo
DEFAULT_SCENARIOS = [
    "database_timeout",
    "api_performance_degradation",
    "memory_leak_detection"
]

DEFAULT_CSV_SCENARIOS = [
    "payment_api_errors",
    "user_auth_failures",
    "search_service_latency"
]

KAFKA_SCENARIOS_AVAILABLE = [
    "streaming_payment_errors",
    "real_time_user_activity",
    "system_resource_monitoring"
]


def available_scenarios() -> Dict[str, List[str]]:
    """Return all available demo scenarios."""
    return {
        "default": DEFAULT_SCENARIOS,
        "csv": DEFAULT_CSV_SCENARIOS,
        "kafka": KAFKA_SCENARIOS_AVAILABLE
    }


async def run_demo_scenario(scenario_name: str, **kwargs) -> DemoRun:
    """Run a single demo scenario and return results."""
    start_time = datetime.now()

    try:
        # For simplified demo, just create a mock scenario
        print(f"🚀 Running demo scenario: {scenario_name}")

        # Simulate some processing time
        await asyncio.sleep(0.5)

        # Mock successful run
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return DemoRun(
            scenario=scenario_name,
            duration_seconds=duration,
            success=True,
            agents_used=["AnalystAgent", "RCAAgent", "ImpactAgent", "MitigationAgent"],
            output_summary=f"Successfully processed {scenario_name} scenario with all 5 agents"
        )

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return DemoRun(
            scenario=scenario_name,
            duration_seconds=duration,
            success=False,
            agents_used=[],
            output_summary=f"Failed to process {scenario_name}: {str(e)}"
        )


async def run_all_scenarios(scenario_list: Optional[List[str]] = None) -> List[DemoRun]:
    """Run multiple demo scenarios and return aggregated results."""
    if scenario_list is None:
        scenario_list = DEFAULT_SCENARIOS

    results = []

    for scenario in scenario_list:
        result = await run_demo_scenario(scenario)
        results.append(result)

        # Small delay between scenarios
        await asyncio.sleep(0.1)

    return results


# Simplified demo entry point for quick testing
async def quick_demo() -> DemoRun:
    """Run a quick demo scenario for testing."""
    return await run_demo_scenario("database_timeout")


if __name__ == "__main__":
    # Simple test when run directly
    async def main():
        print("🧪 Testing demo functions...")

        scenarios = available_scenarios()
        print(f"Available scenarios: {scenarios}")

        result = await quick_demo()
        print(f"Demo result: {result}")

        print("✅ Demo functions working!")

    asyncio.run(main())