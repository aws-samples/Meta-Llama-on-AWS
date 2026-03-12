#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  SRE Agent - Full Deployment${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

for cmd in aws node python3 cdk docker; do
    if ! command -v $cmd &>/dev/null; then
        echo -e "${RED}❌ $cmd not found. Please install it first.${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✅ All tools found${NC}"

if ! aws sts get-caller-identity &>/dev/null; then
    echo -e "${RED}❌ AWS credentials not configured. Run 'aws configure' first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ AWS credentials configured${NC}"

if ! docker ps &>/dev/null; then
    echo -e "${RED}❌ Docker is not running. Start Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker running${NC}"

export AWS_REGION=us-west-2
export AWS_DEFAULT_REGION=us-west-2
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}✅ Account: ${ACCOUNT_ID}, Region: us-west-2${NC}"

# Check CDK bootstrap
if ! aws cloudformation describe-stacks --stack-name CDKToolkit --region us-west-2 &>/dev/null; then
    echo ""
    echo -e "${YELLOW}Bootstrapping CDK...${NC}"
    cdk bootstrap aws://${ACCOUNT_ID}/us-west-2
fi
echo -e "${GREEN}✅ CDK bootstrapped${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Step 1/4: Knowledge Base${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

(
    cd bedrock_kb
    npm install --silent
    cdk deploy --require-approval never
)

KB_ID=$(aws cloudformation describe-stacks \
    --stack-name SimpleBedrockStack --region us-west-2 \
    --query 'Stacks[0].Outputs[?OutputKey==`KnowledgeBaseId`].OutputValue' --output text)
KB_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name SimpleBedrockStack --region us-west-2 \
    --query 'Stacks[0].Outputs[?OutputKey==`PrimaryS3BucketName`].OutputValue' --output text)

echo -e "${GREEN}✅ Knowledge Base deployed: ${KB_ID}${NC}"

# Upload policy documents
echo -e "${YELLOW}Uploading policy documents...${NC}"
aws s3 cp docs/policies/ s3://${KB_BUCKET}/policies/ --recursive --exclude "README.md" --quiet

DATA_SOURCE_ID=$(aws bedrock-agent list-data-sources \
    --knowledge-base-id $KB_ID --region us-west-2 \
    --query 'dataSourceSummaries[0].dataSourceId' --output text)

aws bedrock-agent start-ingestion-job \
    --knowledge-base-id $KB_ID \
    --data-source-id $DATA_SOURCE_ID \
    --region us-west-2 >/dev/null

echo -e "${YELLOW}Waiting for KB sync...${NC}"
for i in $(seq 1 30); do
    STATUS=$(aws bedrock-agent list-ingestion-jobs \
        --knowledge-base-id $KB_ID \
        --data-source-id $DATA_SOURCE_ID \
        --region us-west-2 \
        --query 'ingestionJobSummaries[0].status' --output text 2>/dev/null)
    if [ "$STATUS" = "COMPLETE" ]; then
        break
    fi
    sleep 10
done

if [ "$STATUS" = "COMPLETE" ]; then
    echo -e "${GREEN}✅ Policy documents indexed${NC}"
else
    echo -e "${YELLOW}⚠️  KB sync status: ${STATUS} (may still be processing)${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Step 2/4: AgentCore Backend${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Update KB ID in Dockerfile
sed -i.bak "s/BEDROCK_KB_ID=.*/BEDROCK_KB_ID=${KB_ID} \\\\/" \
    fast_agentcore/patterns/sre-four-agent/Dockerfile
rm -f fast_agentcore/patterns/sre-four-agent/Dockerfile.bak

(
    cd fast_agentcore/infra-cdk
    npm install --silent
    cdk deploy --require-approval never
)

STACK_NAME=$(grep 'stack_name_base' fast_agentcore/infra-cdk/config.yaml | awk '{print $2}')

USER_POOL_ID=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME --region us-west-2 \
    --query 'Stacks[0].Outputs[?OutputKey==`CognitoUserPoolId`].OutputValue' --output text)
AMPLIFY_APP_ID=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME --region us-west-2 \
    --query 'Stacks[0].Outputs[?OutputKey==`AmplifyAppId`].OutputValue' --output text)

echo -e "${GREEN}✅ AgentCore Backend deployed${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Step 3/4: Lambda Log Generator${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

USER_POOL_ARN=$(aws cognito-idp describe-user-pool \
    --user-pool-id $USER_POOL_ID --region us-west-2 \
    --query 'UserPool.Arn' --output text)

(
    cd lambda_log_generator/cdk
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install -q -r requirements.txt
    cdk deploy \
        --context user_pool_id=$USER_POOL_ID \
        --context user_pool_arn=$USER_POOL_ARN \
        --require-approval never
)

LOG_API_URL=$(aws cloudformation describe-stacks \
    --stack-name BankLogGeneratorStack --region us-west-2 \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' --output text)

echo -e "${GREEN}✅ Lambda Log Generator deployed${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Step 4/4: Frontend${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

(
    cd fast_agentcore
    python3 scripts/deploy-frontend.py $STACK_NAME --extra-config logGeneratorApiUrl=$LOG_API_URL
)

AMPLIFY_URL=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME --region us-west-2 \
    --query 'Stacks[0].Outputs[?OutputKey==`AmplifyUrl`].OutputValue' --output text)

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "  Frontend:       ${GREEN}${AMPLIFY_URL}${NC}"
echo -e "  Knowledge Base:  ${KB_ID}"
echo -e "  User Pool:       ${USER_POOL_ID}"
echo -e "  Log Generator:   ${LOG_API_URL}"
echo ""
echo -e "${YELLOW}Next: Create a Cognito user to sign in:${NC}"
echo ""
echo "  aws cognito-idp admin-create-user \\"
echo "    --user-pool-id ${USER_POOL_ID} \\"
echo "    --username your-username \\"
echo "    --user-attributes Name=email,Value=your-email@example.com Name=email_verified,Value=true \\"
echo "    --temporary-password TempPass123! \\"
echo "    --region us-west-2"
echo ""
