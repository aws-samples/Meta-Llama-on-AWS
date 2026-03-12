#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { SimpleBedrockStack } from '../lib/simple-bedrock-stack';

const app = new cdk.App();

new SimpleBedrockStack(app, 'SimpleBedrockStack', {
  prefix: 'sre-kb',
  embeddingModelArn: 'arn:aws:bedrock:us-west-2::foundation-model/amazon.titan-embed-text-v2:0',
  vectorDimensions: 1024,
  enableMetadataExtraction: false,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'us-west-2',
  },
});
