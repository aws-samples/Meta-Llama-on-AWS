# Deployment Guide

Complete step-by-step instructions to deploy the SRE Agent system from scratch.

## Prerequisites

### Software

- **AWS CLI** — configured with credentials (`aws configure`)
- **Node.js 20+** — `node --version`
- **Python 3.11+** — `python3 --version`
- **Docker** — running (`docker ps`)
- **AWS CDK CLI** — `npm install -g aws-cdk`

### AWS Account Setup

1. Your IAM user/role needs AdministratorAccess (or the permissions listed in `fast_agentcore/infra-cdk/minimal-deploy-policy.json`)

2. Enable Llama 3.3 70B model access in us-west-2:
   - Go to https://console.aws.amazon.com/bedrock/home?region=us-west-2#/modelaccess
   - Request access to `Meta Llama 3.3 70B Instruct`
   - Wait for approval (usually instant)

3. Verify:
```bash
aws sts get-caller-identity
aws bedrock list-foundation-models --region us-west-2 \
  --query 'modelSummaries[?contains(modelId, `llama3-3-70b`)].[modelId]' --output text
```

### Bootstrap CDK (one-time per account/region)

```bash
cdk bootstrap aws://$(aws sts get-caller-identity --query Account --output text)/us-west-2
```

---

## Step 1: Deploy the Bedrock Knowledge Base

The Knowledge Base provides SRE policy documents (incident response procedures, business impact baselines, troubleshooting runbooks, etc.) that agents query via RAG.

```bash
cd bedrock_kb
npm install
cdk deploy --require-approval never
```

This creates:
- OpenSearch Serverless collection (vector store)
- S3 bucket for documents
- Bedrock Knowledge Base with data source

Deployment takes ~10-15 minutes (OpenSearch Serverless collection creation is slow).

### Save the outputs

```bash
# Get Knowledge Base ID
export KB_ID=$(aws cloudformation describe-stacks \
  --stack-name SimpleBedrockStack \
  --query 'Stacks[0].Outputs[?OutputKey==`KnowledgeBaseId`].OutputValue' \
  --output text --region us-west-2)

# Get S3 bucket name
export KB_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name SimpleBedrockStack \
  --query 'Stacks[0].Outputs[?OutputKey==`PrimaryS3BucketName`].OutputValue' \
  --output text --region us-west-2)

echo "KB_ID=$KB_ID"
echo "KB_BUCKET=$KB_BUCKET"
```

### Upload policy documents

```bash
# Upload the 6 SRE policy documents
aws s3 cp ../docs/policies/ s3://$KB_BUCKET/policies/ --recursive --exclude "README.md"

# Get the data source ID
export DATA_SOURCE_ID=$(aws bedrock-agent list-data-sources \
  --knowledge-base-id $KB_ID --region us-west-2 \
  --query 'dataSourceSummaries[0].dataSourceId' --output text)

# Trigger sync to index documents
aws bedrock-agent start-ingestion-job \
  --knowledge-base-id $KB_ID \
  --data-source-id $DATA_SOURCE_ID \
  --region us-west-2
```

Wait ~2 minutes for sync to complete. Verify:
```bash
aws bedrock-agent list-ingestion-jobs \
  --knowledge-base-id $KB_ID \
  --data-source-id $DATA_SOURCE_ID \
  --region us-west-2 \
  --query 'ingestionJobSummaries[0].status' --output text
```

Should return `COMPLETE`.


---

## Step 2: Configure and Deploy AgentCore Backend

This deploys the main infrastructure: Cognito authentication, AgentCore Runtime (with the 4-agent SRE workflow), AgentCore Gateway, AgentCore Memory, Amplify hosting, and a Feedback API.

### Update the Knowledge Base ID in the Dockerfile

Edit `fast_agentcore/patterns/sre-four-agent/Dockerfile` and set the `BEDROCK_KB_ID` environment variable to your Knowledge Base ID:

```dockerfile
ENV BEDROCK_KB_ID=YOUR_KB_ID_HERE \
    AWS_REGION=us-west-2
```

### Verify config.yaml

The file `fast_agentcore/infra-cdk/config.yaml` should contain:

```yaml
stack_name_base: meta-sre-agent

admin_user_email: null  # Set to your email to auto-create a Cognito user

backend:
  pattern: sre-four-agent
  deployment_type: docker
```

- `stack_name_base` — must be unique in your account, max 35 characters
- `pattern` — must be `sre-four-agent` (this is the 4-agent SRE workflow)
- `deployment_type` — must be `docker` (the SRE pattern requires container deployment)

### Deploy

```bash
cd fast_agentcore/infra-cdk
npm install
cdk deploy --require-approval never
```

Deployment takes ~5-10 minutes. The CDK will:
1. Build the Docker image from `patterns/sre-four-agent/Dockerfile`
2. Push it to ECR
3. Create the AgentCore Runtime, Memory, and Gateway
4. Create Cognito User Pool and OAuth clients
5. Create Amplify Hosting app
6. Create Feedback API (API Gateway + Lambda + DynamoDB)

### Save the outputs

```bash
export STACK_NAME=meta-sre-agent

# Amplify App ID (for frontend deployment)
export AMPLIFY_APP_ID=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME --region us-west-2 \
  --query 'Stacks[0].Outputs[?OutputKey==`AmplifyAppId`].OutputValue' --output text)

# Cognito User Pool ID (for creating users and Lambda auth)
export USER_POOL_ID=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME --region us-west-2 \
  --query 'Stacks[0].Outputs[?OutputKey==`CognitoUserPoolId`].OutputValue' --output text)

# Staging bucket (for Amplify deployments)
export STAGING_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME --region us-west-2 \
  --query 'Stacks[0].Outputs[?OutputKey==`StagingBucketName`].OutputValue' --output text)

echo "AMPLIFY_APP_ID=$AMPLIFY_APP_ID"
echo "USER_POOL_ID=$USER_POOL_ID"
echo "STAGING_BUCKET=$STAGING_BUCKET"
```

---

## Step 3: Deploy the Lambda Log Generator

This creates a Lambda function that generates realistic banking system logs and writes them to CloudWatch at `/aws/banking/system-logs`. The frontend has a "Generate Logs" button that calls this Lambda via API Gateway.

```bash
cd lambda_log_generator/cdk
pip install -r requirements.txt

# Get the User Pool ARN
export USER_POOL_ARN=$(aws cognito-idp describe-user-pool \
  --user-pool-id $USER_POOL_ID --region us-west-2 \
  --query 'UserPool.Arn' --output text)

cdk deploy \
  --context user_pool_id=$USER_POOL_ID \
  --context user_pool_arn=$USER_POOL_ARN \
  --require-approval never
```

### Save the API URL

```bash
export LOG_API_URL=$(aws cloudformation describe-stacks \
  --stack-name BankLogGeneratorStack --region us-west-2 \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' --output text)

echo "LOG_API_URL=$LOG_API_URL"
```

---

## Step 4: Deploy the Frontend

The frontend is a React (Vite) app deployed to Amplify Hosting. There are two ways to deploy it.

### Option A: Automated (recommended)

The `deploy-frontend.py` script automatically fetches stack outputs, generates `aws-exports.json`, builds, and deploys:

```bash
cd fast_agentcore
python scripts/deploy-frontend.py
```

If you used a custom stack name:
```bash
python scripts/deploy-frontend.py my-custom-stack-name
```

The script will print the Amplify URL when done.

### Option B: Manual

If the automated script doesn't work or you need to add custom config (like the log generator API URL):

1. Generate `aws-exports.json`:

```bash
cd fast_agentcore/frontend

# Get all required values from stack outputs
export COGNITO_CLIENT_ID=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME --region us-west-2 \
  --query 'Stacks[0].Outputs[?OutputKey==`CognitoClientId`].OutputValue' --output text)

export RUNTIME_ARN=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME --region us-west-2 \
  --query 'Stacks[0].Outputs[?OutputKey==`RuntimeArn`].OutputValue' --output text)

export AMPLIFY_URL=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME --region us-west-2 \
  --query 'Stacks[0].Outputs[?OutputKey==`AmplifyUrl`].OutputValue' --output text)

export FEEDBACK_API_URL=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME --region us-west-2 \
  --query 'Stacks[0].Outputs[?OutputKey==`FeedbackApiUrl`].OutputValue' --output text)

cat > public/aws-exports.json << EOF
{
  "authority": "https://cognito-idp.us-west-2.amazonaws.com/$USER_POOL_ID",
  "client_id": "$COGNITO_CLIENT_ID",
  "redirect_uri": "$AMPLIFY_URL",
  "post_logout_redirect_uri": "$AMPLIFY_URL",
  "response_type": "code",
  "scope": "email openid profile",
  "automaticSilentRenew": true,
  "region": "us-west-2",
  "awsRegion": "us-west-2",
  "userPoolId": "$USER_POOL_ID",
  "userPoolWebClientId": "$COGNITO_CLIENT_ID",
  "agentRuntimeArn": "$RUNTIME_ARN",
  "agentPattern": "langgraph-single-agent",
  "feedbackApiUrl": "$FEEDBACK_API_URL",
  "logGeneratorApiUrl": "$LOG_API_URL"
}
EOF
```

Note: `agentPattern` must be `langgraph-single-agent` — this tells the frontend parser how to interpret the streaming response format. The SRE 4-agent system uses LangGraph internally.

2. Build and deploy:

```bash
npm install
npm run build

# Zip from inside the build directory
cd build
zip -r ../frontend-build.zip .
cd ..

# Upload to S3 staging bucket
aws s3 cp frontend-build.zip s3://$STAGING_BUCKET/frontend-build.zip

# Trigger Amplify deployment
aws amplify start-deployment \
  --app-id $AMPLIFY_APP_ID \
  --branch-name main \
  --source-url s3://$STAGING_BUCKET/frontend-build.zip
```

---

## Step 5: Create a Cognito User

If you set `admin_user_email` in config.yaml, a user was auto-created and credentials emailed. Otherwise, create one manually:

```bash
aws cognito-idp admin-create-user \
  --user-pool-id $USER_POOL_ID \
  --username your-username \
  --user-attributes Name=email,Value=your-email@example.com Name=email_verified,Value=true \
  --temporary-password TempPass123! \
  --region us-west-2
```

On first login you'll be prompted to set a permanent password.

---

## Step 6: Test the System

1. Open the Amplify URL (printed during deployment, or find it in the Amplify console)
2. Sign in with your Cognito credentials
3. Click "Generate Logs" to create test banking logs in CloudWatch
4. Click "Test System" to trigger the full 4-agent workflow
5. Watch the agents analyze the logs in real-time:
   - Analyst detects anomalies
   - RCA identifies root causes
   - Impact calculates business impact
   - Mitigation generates action plans with communication templates

You can also type questions in the chat input to ask about the system's capabilities.

---

## Cleanup

To tear down all resources:

```bash
# 1. Delete Lambda Log Generator
cd lambda_log_generator/cdk
cdk destroy --force

# 2. Delete AgentCore Backend (includes Cognito, Amplify, Gateway, Runtime, Memory)
cd ../../fast_agentcore/infra-cdk
cdk destroy --force

# 3. Delete Knowledge Base (includes OpenSearch Serverless, S3 bucket)
cd ../../bedrock_kb
cdk destroy --force
```

---

## Redeploying After Code Changes

### Agent code changes (sre_agent.py, agent logic)

```bash
cd fast_agentcore/infra-cdk
cdk deploy --require-approval never
```

This rebuilds the Docker image and updates the AgentCore Runtime.

### Frontend changes

```bash
cd fast_agentcore
python scripts/deploy-frontend.py
```

Or manually: build, zip, upload to S3, trigger Amplify deployment (see Step 4B).

### Lambda changes

```bash
cd lambda_log_generator/cdk
cdk deploy --require-approval never
```

### Knowledge Base document changes

```bash
# Upload updated docs
aws s3 cp ../docs/policies/ s3://$KB_BUCKET/policies/ --recursive --exclude "README.md"

# Re-sync
aws bedrock-agent start-ingestion-job \
  --knowledge-base-id $KB_ID \
  --data-source-id $DATA_SOURCE_ID \
  --region us-west-2
```
