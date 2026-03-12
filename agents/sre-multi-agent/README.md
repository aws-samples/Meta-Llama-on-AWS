# Multi-Agent SRE Incident Response System

A production-ready multi-agent system for automated incident response using Meta Llama 3.3 70B, Amazon Bedrock AgentCore, Knowledge Base integration, and CloudWatch Logs analysis.

## 🎯 What This Does

Automatically analyzes CloudWatch logs and generates comprehensive incident response plans using 4 specialized AI agents:

1. **Analyst Agent** - Detects incidents from CloudWatch logs
2. **RCA Agent** - Identifies root causes using historical patterns
3. **Impact Agent** - Calculates business impact and revenue loss
4. **Mitigation Agent** - Generates actionable mitigation plans with communication templates

## ✨ Key Features

- ✅ **CloudWatch Logs Integration** - Analyzes real-time error logs
- ✅ **Bedrock Knowledge Base** - 6 policy documents for incident response procedures
- ✅ **Multi-Agent Orchestration** - LangGraph workflow with 4 specialized agents
- ✅ **AWS FAST Template** - Production-ready infrastructure with Amplify frontend
- ✅ **Real-time Streaming** - See agent responses as they generate
- ✅ **Cognito Authentication** - Secure user access

## 🏗️ Architecture

```
User Query → Amplify Frontend → Bedrock AgentCore Runtime
                                        ↓
                            LangGraph Orchestrator
                                        ↓
                    ┌───────────────────┴───────────────────┐
                    ↓                   ↓                   ↓
            CloudWatch Logs      Bedrock KB         Agent Logic
            (Incident Data)    (6 Policy Docs)   (4 Specialized Agents)
```

## 📋 Prerequisites

### AWS Permissions

**For the user deploying this stack, you need:**

**Option 1: AdministratorAccess** (Simplest)
```bash
# Attach AdministratorAccess policy to your IAM user/role
```

**Option 2: Minimum Required Permissions** (Least Privilege)
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "cloudformation:*",
        "iam:CreateRole",
        "iam:DeleteRole",
        "iam:AttachRolePolicy",
        "iam:DetachRolePolicy",
        "iam:PutRolePolicy",
        "iam:DeleteRolePolicy",
        "iam:GetRole",
        "iam:PassRole",
        "bedrock:*",
        "bedrock-agent:*",
        "bedrock-agentcore:*",
        "amplify:*",
        "cognito-idp:*",
        "s3:*",
        "lambda:*",
        "logs:*",
        "ssm:*",
        "ec2:DescribeVpcs",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "ecr:*",
        "sts:GetCallerIdentity"
      ],
      "Resource": "*"
    }
  ]
}
```

**Runtime Permissions (Automatically Created by CDK):**

The agent runtime will have these permissions (no action needed):
- ✅ Bedrock Model Invocation (`bedrock:InvokeModel` for Llama 3.3 70B)
- ✅ Knowledge Base Query (`bedrock-agent:Retrieve`)
- ✅ CloudWatch Logs Read (`logs:FilterLogEvents`, `logs:GetLogEvents`)
- ✅ AgentCore Memory (`bedrock-agentcore:CreateEvent`, `bedrock-agentcore:GetEvent`)
- ✅ Code Interpreter (`bedrock-agentcore:InvokeCodeInterpreter`)

### Software Requirements

- **AWS Account** with appropriate permissions (see above)
- **AWS CLI** configured with credentials
- **Node.js** 20+ and **Python** 3.11+
- **AWS CDK** installed: `npm install -g aws-cdk`
- **Docker** running (for agent packaging)

### Bedrock Model Access

Enable Llama 3.3 70B model access in your AWS account:

```bash
# Check if model is available
aws bedrock list-foundation-models \
  --region us-west-2 \
  --query 'modelSummaries[?contains(modelId, `llama3-3-70b`)]'

# If not available, request access via AWS Console:
# https://console.aws.amazon.com/bedrock/home?region=us-west-2#/modelaccess
```

## 🚀 Quick Start

See [`deployment_docs/DEPLOYMENT_GUIDE.md`](deployment_docs/DEPLOYMENT_GUIDE.md) for complete step-by-step instructions.

Summary of deployment steps:

```bash
# 1. Deploy Knowledge Base (~15 min)
cd bedrock_kb && npm install && cdk deploy

# 2. Upload policy documents to KB
aws s3 cp docs/policies/ s3://$KB_BUCKET/policies/ --recursive --exclude "README.md"

# 3. Deploy AgentCore Backend (~10 min)
cd fast_agentcore/infra-cdk && npm install && cdk deploy

# 4. Deploy Lambda Log Generator (~3 min)
cd lambda_log_generator/cdk && pip install -r requirements.txt && cdk deploy

# 5. Deploy Frontend
cd fast_agentcore && python scripts/deploy-frontend.py

# 6. Create Cognito user and sign in
```

### Test the System

1. Open the Amplify URL
2. Sign in with Cognito credentials
3. Click "Generate Logs" to create test banking logs
4. Click "Test System" to trigger the 4-agent analysis workflow

## 📁 Project Structure

```
metasreagent/
├── fast_agentcore/           # AWS FAST AgentCore template (main deployment)
│   ├── infra-cdk/                 # CDK infrastructure code
│   │   ├── config.yaml            # Deployment configuration
│   │   └── lib/                   # Stack definitions
│   ├── frontend/                  # React/Vite Amplify frontend
│   ├── patterns/
│   │   └── sre-four-agent/        # 4-agent SRE workflow
│   │       ├── sre_agent.py       # AgentCore entrypoint
│   │       ├── Dockerfile         # Container definition
│   │       └── src/orchestration/four_agent/  # Agent implementations
│   ├── gateway/                   # AgentCore Gateway tools
│   └── scripts/                   # Deployment scripts
├── bedrock_kb/                    # Knowledge Base deployment (TypeScript CDK)
│   ├── lib/simple-bedrock-stack.ts
│   └── lambda/create-index/       # OpenSearch index creation
├── lambda_log_generator/                   # Lambda log generator
│   ├── lambda_function.py         # Log generation logic
│   └── cdk/                       # CDK stack (Python)
├── docs/
│   └── policies/                  # 6 SRE policy documents (uploaded to KB)
├── deployment_docs/          # Deployment documentation
│   ├── DEPLOYMENT_GUIDE.md        # Step-by-step deployment instructions
│   ├── ARCHITECTURE.md            # System architecture
│   └── TROUBLESHOOTING.md         # Common issues and fixes
└── README.md                      # This file
```

## 📚 Documentation

- **[Deployment Guide](deployment_docs/DEPLOYMENT_GUIDE.md)** — Step-by-step deployment instructions
- **[Architecture](deployment_docs/ARCHITECTURE.md)** — System architecture and component details
- **[Troubleshooting](deployment_docs/TROUBLESHOOTING.md)** — Common issues and fixes
- **[FAST Template Docs](fast_agentcore/docs/)** — AWS FAST template documentation

## 🔧 Configuration

### Knowledge Base Documents

The system uses 6 policy documents stored in Bedrock Knowledge Base:

1. **POL-SRE-001** - Incident Response Procedures
2. **POL-SRE-002** - Business Impact Baselines (TPS: 850, Revenue formulas)
3. **POL-SRE-003** - Troubleshooting Runbooks
4. **POL-SRE-004** - Known Failure Patterns (Database connection pool exhaustion, etc.)
5. **communication-templates.md** - SEV-1/2/3 communication templates
6. **mitigation-playbooks.md** - Mitigation procedures

### CloudWatch Logs

Configure the log group in `config.yaml`:

```yaml
cloudwatch:
  log_group: /aws/sre-demo/incidents  # Your log group name
```

The log group must exist before deployment. Create it:

```bash
aws logs create-log-group --log-group-name /aws/sre-demo/incidents --region us-west-2
```

## 🧪 Testing

### Query the System

In the UI, try these queries:
- "whats going on?"
- "analyze the current incident"
- "what's the impact?"

### Expected Output

The system will:
1. ✅ Retrieve 20-40 logs from CloudWatch (last 15 minutes)
2. ✅ **Analyst Agent**: Analyze errors across multiple services
3. ✅ **RCA Agent**: Identify root cause (e.g., database connection pool exhaustion, 0.8 confidence)
4. ✅ **Impact Agent**: Calculate business impact ($2,052.50/min revenue loss, TPS: 450 vs baseline 850)
5. ✅ **Mitigation Agent**: Generate mitigation plan with:
   - Specific kubectl commands for each service
   - Success criteria and rollback procedures
   - Communication templates for internal/external stakeholders

## 🛠️ Troubleshooting

### Deployment Issues

```bash
# Check stack status
aws cloudformation describe-stacks --stack-name meta-sre-agent --region us-west-2

# View deployment events
aws cloudformation describe-stack-events --stack-name meta-sre-agent --region us-west-2 | head -50

# Check CDK bootstrap
aws cloudformation describe-stacks --stack-name CDKToolkit --region us-west-2
```

### Permission Errors

```bash
# Verify your AWS credentials
aws sts get-caller-identity

# Check if you have required permissions
aws iam get-user
aws iam list-attached-user-policies --user-name YOUR_USERNAME
```

### No Logs Retrieved

```bash
# Verify log group exists
aws logs describe-log-groups --log-group-name-prefix /aws/sre-demo --region us-west-2

# Check IAM permissions on agent role
aws iam get-role --role-name meta-sre-agent-agent-role

# Verify logs exist
aws logs filter-log-events \
  --log-group-name /aws/banking/system-logs \
  --start-time $(($(date +%s) - 3600))000 \
  --region us-west-2
```

### Knowledge Base Not Working

```bash
# Verify KB exists
aws bedrock-agent get-knowledge-base --knowledge-base-id YOUR_KB_ID --region us-west-2

# Check data source sync status
aws bedrock-agent list-data-sources --knowledge-base-id YOUR_KB_ID --region us-west-2

# View agent runtime logs
aws logs tail /aws/bedrock-agentcore/runtimes/meta_sre_agent_StrandsAgent-* \
  --since 5m --region us-west-2 --follow
```

### Bedrock Model Access

```bash
# Check if Llama 3.3 70B is enabled
aws bedrock list-foundation-models \
  --region us-west-2 \
  --query 'modelSummaries[?contains(modelId, `llama3-3-70b`)]'

```

## 🤝 Contributing

This project uses:
- Multi-agent SRE orchestration logic
- AWS FAST template infrastructure
- Bedrock Knowledge Base integration

See the [Deployment Guide](deployment_docs/DEPLOYMENT_GUIDE.md) for complete implementation details.

## 🔒 Security Best Practices

- ✅ IAM roles with least privilege
- ✅ Cognito user authentication
- ✅ Environment variables for sensitive config
- ✅ CloudWatch logging for audit trail
- ✅ No hardcoded credentials in code

## 💰 Cost Estimate

**Approximate monthly costs (us-west-2):**
- Bedrock AgentCore Runtime: ~$50-100/month (depends on usage)
- Bedrock Model Invocations (Llama 3.3 70B): ~$0.0024/1K input tokens, ~$0.0024/1K output tokens
- Bedrock Knowledge Base: ~$0.10/GB storage + $0.002/query
- Amplify Hosting: ~$15/month
- Cognito: Free tier (50,000 MAUs)
- CloudWatch Logs: ~$0.50/GB ingested

**Total estimated cost: $70-150/month** (varies with usage)


## 🆘 Support

For issues:
1. Check [Troubleshooting Guide](deployment_docs/TROUBLESHOOTING.md)
2. Review [Architecture](deployment_docs/ARCHITECTURE.md) for component details
3. Check CloudWatch logs for runtime errors
4. Contact the project maintainers

## 🔗 References

- [Full-Stack Solution Template for Amazon Bedrock AgentCore (FAST)](https://github.com/awslabs/fullstack-solution-template-for-agentcore) — The open-source template this project is built on
- [Accelerate agentic application development with a full-stack starter template for Amazon Bedrock AgentCore](https://aws.amazon.com/blogs/machine-learning/accelerate-agentic-application-development-with-a-full-stack-starter-template-for-amazon-bedrock-agentcore/) — AWS blog post introducing the FAST template

---

**Built with:** Meta Llama 3.3 70B | Amazon Bedrock AgentCore | AWS FAST Template | LangGraph | Amplify
