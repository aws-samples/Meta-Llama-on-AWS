# Bedrock Knowledge Base with OpenSearch Serverless

Simple TypeScript CDK stack for deploying a Bedrock Knowledge Base with OpenSearch Serverless vector store.

## Architecture

- **S3 Bucket**: Storage for documents
- **OpenSearch Serverless**: Vector database for embeddings
- **Bedrock Knowledge Base**: RAG-enabled knowledge base
- **Lambda Function**: Creates OpenSearch index with proper configuration

## Prerequisites

- AWS CLI configured
- Node.js 20+
- Docker running (for Lambda bundling)
- AWS CDK CLI: `npm install -g aws-cdk`

## Deployment

```bash
# Install dependencies
npm install

# Deploy
cdk deploy

# Outputs:
# - KnowledgeBaseId: Use this in your applications
# - PrimaryS3BucketName: Upload documents here
```

## Configuration

Edit `bin/app.ts` to customize:

```typescript
new SimpleBedrockStack(app, 'SimpleBedrockStack', {
  prefix: 'sre-kb',                    // Resource name prefix
  embeddingModelArn: '...',            // Embedding model
  vectorDimensions: 1024,              // Vector dimensions
  enableMetadataExtraction: false,     // Enable JSON metadata extraction
});
```

## Upload Documents

```bash
# Get bucket name from deployment output
BUCKET_NAME="<from-output>"

# Upload documents
aws s3 cp docs/ s3://$BUCKET_NAME/docs/ --recursive

# Trigger sync (optional - auto-syncs every 5 minutes)
aws bedrock-agent start-ingestion-job \
  --knowledge-base-id <KB_ID> \
  --data-source-id <DATA_SOURCE_ID>
```

## Cleanup

```bash
cdk destroy
```

## Features

- ✅ Automatic OpenSearch index creation
- ✅ Proper IAM permissions
- ✅ 30s wait for policy propagation
- ✅ Support for Titan v2 embeddings (1024 dimensions)
- ✅ Optional metadata extraction from JSON files

## Deployed Resources

- Knowledge Base ID: `YOUR_KB_ID`
- S3 Bucket: `simplebedrockstack-primaryknowledgebasebucket93ad8-fn0en55pantw`
