# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Configure UV for container environment
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    DOCKER_CONTAINER=1 \
    OTEL_PYTHON_LOG_CORRELATION=true \
    PYTHONUNBUFFERED=1 \
    BEDROCK_KB_ID=YOUR_KB_ID \
    AWS_REGION=us-west-2

# Copy pyproject.toml for FAST installation
COPY pyproject.toml .

# Copy and install agent-specific requirements first
COPY patterns/sre-four-agent/requirements.txt requirements.txt
RUN uv pip install --no-cache -r requirements.txt && \
    uv pip install --no-cache aws-opentelemetry-distro==0.10.1

# Install FAST package with only core dependencies (no dev/agent optional deps)
RUN uv pip install --no-cache -e . --no-deps && \
    uv pip install --no-cache requests>=2.31.0

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 8080

# Copy agent code and SRE 4-agent system
COPY patterns/sre-four-agent/sre_agent.py .
COPY patterns/sre-four-agent/src/ src/
COPY patterns/sre-four-agent/utils/ utils/

# Healthcheck using Python (no extra dependencies needed)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/ping', timeout=2)" || exit 1

# Start agent with OpenTelemetry instrumentation
CMD ["opentelemetry-instrument", "python", "-m", "sre_agent"]
