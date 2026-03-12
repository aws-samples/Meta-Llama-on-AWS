# Architecture

## System Overview

```
User → Amplify Frontend (React/Vite) → Cognito Auth (JWT)
                                            ↓
                                   Bedrock AgentCore Runtime
                                            ↓
                                   sre_agent.py (entrypoint)
                                            ↓
                              ┌─────────────┴─────────────┐
                              ↓                           ↓
                     CloudWatch Logs              Bedrock Knowledge Base
                  (/aws/banking/system-logs)        (6 SRE policy docs)
                              ↓                           ↓
                    ┌─────────┴─────────────────────────────┐
                    ↓              ↓            ↓           ↓
               Analyst Agent → RCA Agent → Impact Agent → Mitigation Agent
                    ↓
              Streaming Response → Frontend Chat UI
```

## Components

### 1. Bedrock Knowledge Base (`bedrock_kb/`)

- **Stack**: `SimpleBedrockStack`
- **CDK**: TypeScript (`bedrock_kb/lib/simple-bedrock-stack.ts`)
- **Vector Store**: OpenSearch Serverless collection
- **Embedding Model**: Amazon Titan Embed Text v2 (1024 dimensions)
- **Documents**: 6 SRE policy documents in `docs/policies/`:
  - `incident-response-procedures.md` — POL-SRE-001
  - `business-impact-baselines.md` — POL-SRE-002 (TPS baselines, revenue formulas)
  - `troubleshooting-runbooks.md` — POL-SRE-003
  - `known-failure-patterns.md` — POL-SRE-004
  - `communication-templates.md` — SEV-1/2/3 templates
  - `mitigation-playbooks.md` — Mitigation procedures

### 2. AgentCore Backend (`fast_agentcore/`)

- **Stack**: Configured via `infra-cdk/config.yaml` (default: `meta-sre-agent`)
- **CDK**: TypeScript (`infra-cdk/lib/`)
- **Pattern**: `sre-four-agent` (Docker deployment)
- **LLM**: Meta Llama 3.3 70B Instruct (`us.meta.llama3-3-70b-instruct-v1:0`)

What the CDK creates:
- **Cognito User Pool** — user authentication with OAuth2
- **AgentCore Runtime** — runs the Docker container with the 4-agent workflow
- **AgentCore Memory** — conversation persistence (short-term)
- **AgentCore Gateway** — MCP protocol gateway with Lambda tool targets
- **Amplify Hosting** — static site hosting for the frontend
- **S3 Staging Bucket** — for Amplify manual deployments
- **Feedback API** — API Gateway + Lambda + DynamoDB for user feedback
- **IAM Roles** — execution role with permissions for Bedrock, CloudWatch, ECR, Memory, etc.

#### Agent Entrypoint (`patterns/sre-four-agent/sre_agent.py`)

The entrypoint handles two types of requests:
- **Incident analysis** (triggered by "Test System" button or keywords like "analyze", "investigate") — runs the full 4-agent workflow
- **Conversational queries** (any other chat input) — responds using Llama 3.3 70B directly

The 4-agent workflow:
1. Fetches logs from CloudWatch (`/aws/banking/system-logs`)
2. Queries Knowledge Base for relevant policies
3. Runs agents sequentially: Analyst → RCA → Impact → Mitigation
4. Streams results back to the frontend as each agent completes
5. Saves analysis to AgentCore Memory

#### Agent Implementations (`patterns/sre-four-agent/src/orchestration/four_agent/`)

| Agent | File | Role |
|-------|------|------|
| Analyst | `analyst_agent.py` | Analyzes CloudWatch logs, detects anomalies, assigns severity |
| RCA | `rca_agent.py` | Identifies root causes using log patterns and KB policies |
| Impact | `impact_agent.py` | Calculates business impact (TPS, revenue, approvals) using KB baselines |
| Mitigation | `mitigation_agent.py` | Generates action plans, communication templates, rollback procedures |

Supporting modules:
- `orchestrator.py` — `PhaseTwoOrchestrator` coordinates the 4-agent workflow
- `bedrock_kb_reader.py` — queries Bedrock Knowledge Base via `bedrock:Retrieve`
- `schema.py` — Pydantic models for agent messages, payloads, evidence
- `settings.py` — model configuration (max tokens: 4096)
- `llm.py` — Bedrock model invocation wrapper

### 3. Lambda Log Generator (`lambda_log_generator/`)

- **Stack**: `BankLogGeneratorStack`
- **CDK**: Python (`lambda_log_generator/cdk/lambda_stack.py`)
- **Lambda**: Generates realistic banking logs (auth, payments, trading, accounts services)
- **API Gateway**: REST API with Cognito authorizer
- **CloudWatch Log Group**: `/aws/banking/system-logs`

The Lambda generates logs with configurable parameters:
- Number of events (default: 50)
- Error rate (default: 15%)
- Services: auth-service, payments-service, trading-service, accounts-service
- Error types: 429, 500, 502, 503, 504

### 4. Frontend (`fast_agentcore/frontend/`)

- **Framework**: React + Vite + TypeScript + Tailwind CSS
- **Hosting**: AWS Amplify (manual deployment via S3)
- **Auth**: Cognito OAuth2 (Authorization Code flow)
- **Theme**: Dark (gray-900 background)

Key components:
- `ChatInterface.tsx` — main chat UI, handles message sending and streaming
- `ChatMessage.tsx` — renders individual messages with markdown support
- `LogGeneratorButton.tsx` — triggers Lambda to generate test logs
- `TestSystemButton.tsx` — triggers the full 4-agent analysis workflow
- `ChatHeader.tsx` — header with title, New Chat, Generate Logs, Test System buttons

Configuration: `frontend/public/aws-exports.json` — generated from CDK stack outputs. The `agentPattern` field must be `langgraph-single-agent` (this controls the streaming response parser).

## Data Flow

### Test System Button Click

1. Frontend sends `"Analyze the logs in /aws/banking/system-logs for incidents"` to AgentCore Runtime
2. `sre_agent.py` detects this as an incident analysis request
3. `create_scenario_from_query()` fetches last 500 log events from CloudWatch
4. Creates a `ScenarioSnapshot` with the log data
5. `stream_orchestrator_results()` runs each agent sequentially:
   - Each agent calls Bedrock (Llama 3.3 70B) with its prompt + context
   - Each agent queries Knowledge Base for relevant policies
   - Results stream back to frontend as each agent completes
6. Final summary is generated and streamed
7. Analysis is saved to AgentCore Memory

### Conversational Query

1. Frontend sends user message to AgentCore Runtime
2. `sre_agent.py` detects this is NOT an incident analysis request
3. `_handle_conversational_query()` calls Llama 3.3 70B directly
4. Response streams back to frontend

## IAM Permissions

The AgentCore Runtime execution role (`agentcore-role.ts`) has:
- `bedrock:InvokeModel` — call Llama 3.3 70B
- `bedrock:Retrieve` — query Knowledge Base (added manually, not in base template)
- `logs:FilterLogEvents`, `logs:GetLogEvents` — read CloudWatch banking logs
- `bedrock-agentcore:CreateEvent`, `GetEvent`, `ListEvents` — Memory access
- `ecr:BatchGetImage` — pull Docker image
- `logs:CreateLogStream`, `PutLogEvents` — write runtime logs
- `ssm:GetParameter` — read configuration
- `secretsmanager:GetSecretValue` — read machine client secret

## CloudWatch Log Groups

| Log Group | Purpose |
|-----------|---------|
| `/aws/banking/system-logs` | Simulated banking logs (generated by Lambda) |
| `/aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT` | AgentCore runtime logs (agent execution, errors) |
| `/aws/lambda/<stack-name>-feedback` | Feedback API Lambda logs |
| `/aws/lambda/BankLogGeneratorStack-*` | Log generator Lambda logs |

## Cost Estimate

| Resource | Estimated Cost |
|----------|---------------|
| AgentCore Runtime | ~$50-100/month (depends on usage) |
| Bedrock Llama 3.3 70B | ~$0.0024/1K tokens (input and output) |
| OpenSearch Serverless | ~$12-24/day (minimum 2 OCUs) |
| Amplify Hosting | ~$0.50/month |
| Lambda | Negligible (free tier) |
| DynamoDB | Negligible (on-demand) |
| Cognito | Free (up to 50K MAUs) |

The biggest cost driver is OpenSearch Serverless (~$350-700/month minimum). For development/demo purposes, consider destroying the KB stack when not in use.
