# SRE Agent Deployment Documentation

This folder contains the complete deployment guide for the Multi-Agent SRE Incident Response System.

## Documents

| Document | Description |
|----------|-------------|
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Step-by-step deployment instructions (start here) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, components, and how they connect |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and how to fix them |

## Quick Overview

This system deploys 4 components to AWS (us-west-2):

1. **Bedrock Knowledge Base** — OpenSearch Serverless vector store with 6 SRE policy documents
2. **AgentCore Backend** — 4-agent SRE workflow (Analyst → RCA → Impact → Mitigation) running on Bedrock AgentCore Runtime
3. **Lambda Log Generator** — Generates realistic banking logs to CloudWatch for testing
4. **Amplify Frontend** — React web app with Cognito authentication

Total deployment time: ~30-45 minutes.

## Prerequisites at a Glance

- AWS account with AdministratorAccess (or equivalent permissions)
- AWS CLI configured (`aws configure`)
- Node.js 20+, Python 3.11+, Docker running
- AWS CDK CLI: `npm install -g aws-cdk`
- Bedrock model access enabled for Llama 3.3 70B in us-west-2
