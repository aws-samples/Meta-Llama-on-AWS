#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${RED}========================================${NC}"
echo -e "${RED}  DESTROY ALL AWS RESOURCES${NC}"
echo -e "${RED}========================================${NC}"
echo ""
echo "This will destroy:"
echo "  1. Lambda Log Generator (BankLogGeneratorStack)"
echo "  2. AgentCore Backend (meta-sre-agent)"
echo "  3. Bedrock Knowledge Base (SimpleBedrockStack)"
echo ""
read -p "Are you sure? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""

# Step 1: Lambda Log Generator
echo -e "${YELLOW}[1/3] Destroying Lambda Log Generator...${NC}"
if aws cloudformation describe-stacks --stack-name BankLogGeneratorStack --region us-west-2 &>/dev/null; then
    (
        cd lambda_log_generator/cdk
        if [ ! -d ".venv" ]; then
            python3 -m venv .venv
        fi
        source .venv/bin/activate
        pip install -q -r requirements.txt
        cdk destroy --force
    )
    echo -e "${GREEN}✅ Lambda Log Generator destroyed${NC}"
else
    echo -e "${GREEN}✅ Lambda Log Generator already gone${NC}"
fi

echo ""

# Step 2: AgentCore Backend
echo -e "${YELLOW}[2/3] Destroying AgentCore Backend...${NC}"
STACK_NAME=$(grep 'stack_name_base' fast_agentcore/infra-cdk/config.yaml | awk '{print $2}')
if aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region us-west-2 &>/dev/null; then
    (
        cd fast_agentcore/infra-cdk
        npm install --silent
        cdk destroy --force
    )
    echo -e "${GREEN}✅ AgentCore Backend destroyed${NC}"
else
    echo -e "${GREEN}✅ AgentCore Backend already gone${NC}"
fi

echo ""

# Step 3: Knowledge Base
echo -e "${YELLOW}[3/3] Destroying Knowledge Base...${NC}"
if aws cloudformation describe-stacks --stack-name SimpleBedrockStack --region us-west-2 &>/dev/null; then
    (
        cd bedrock_kb
        npm install --silent
        cdk destroy --force
    )
    echo -e "${GREEN}✅ Knowledge Base destroyed${NC}"
else
    echo -e "${GREEN}✅ Knowledge Base already gone${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All resources destroyed${NC}"
echo -e "${GREEN}========================================${NC}"
