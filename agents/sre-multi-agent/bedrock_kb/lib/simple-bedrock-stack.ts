import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as opensearchserverless from 'aws-cdk-lib/aws-opensearchserverless';
import * as bedrock from 'aws-cdk-lib/aws-bedrock';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as cr from 'aws-cdk-lib/custom-resources';
import { Construct } from 'constructs';

export interface SimpleBedrockStackProps extends cdk.StackProps {
  /** Prefix for resource names (e.g., 'titan-v2-1024', 'cohere-v3') */
  prefix?: string;
  /** Embedding model ARN */
  embeddingModelArn?: string;
  /** Vector dimensions for the embedding model */
  vectorDimensions?: number;
  /** Enable metadata extraction from JSON files */
  enableMetadataExtraction?: boolean;
}

export class SimpleBedrockStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: SimpleBedrockStackProps) {
    super(scope, id, props);

    // Use provided values or defaults
    const PREFIX = props?.prefix || 'eb-semantic';
    const EMBEDDING_MODEL_ARN = props?.embeddingModelArn || 
      'arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0';
    const EMBEDDING_MODEL_VECTOR_DIMENSIONS = props?.vectorDimensions || 1024;
    const ENABLE_METADATA_EXTRACTION = props?.enableMetadataExtraction || false;

    // Primary S3 Bucket for Knowledge Base Data
    const primaryKnowledgeBaseBucket = new s3.Bucket(this, 'PrimaryKnowledgeBaseBucket', {
      // Let CDK auto-generate bucket name to avoid conflicts
    });
 

    // OpenSearch Serverless Collection
    const vectorCollection = new opensearchserverless.CfnCollection(this, 'VectorCollection', {
      name: `${PREFIX}-kb-collection`,
      type: 'VECTORSEARCH',
    });

    // Security Policy for OpenSearch Serverless
    const securityPolicy = new opensearchserverless.CfnSecurityPolicy(this, 'SecurityPolicy', {
      name: `${PREFIX}-kb-security-policy`,
      type: 'encryption',
      policy: JSON.stringify({
        Rules: [
          {
            ResourceType: 'collection',
            Resource: [`collection/${vectorCollection.name}`],
          },
        ],
        AWSOwnedKey: true,
      }),
    });

    // Network Policy for OpenSearch Serverless
    const networkPolicy = new opensearchserverless.CfnSecurityPolicy(this, 'NetworkPolicy', {
      name: `${PREFIX}-kb-network-policy`,
      type: 'network',
      policy: JSON.stringify([
        {
          Rules: [
            {
              ResourceType: 'collection',
              Resource: [`collection/${vectorCollection.name}`],
            },
            {
              ResourceType: 'dashboard',
              Resource: [`collection/${vectorCollection.name}`],
            },
          ],
          AllowFromPublic: true,
        },
      ]),
    });

    // IAM Role for Bedrock Knowledge Base
    const bedrockExecutionRole = new iam.Role(this, 'BedrockExecutionRole', {
      assumedBy: new iam.ServicePrincipal('bedrock.amazonaws.com'),
    });

    // S3 Policy for Bedrock - Full Access to Primary Bucket
    const s3FullAccessPolicy = new iam.Policy(this, 'BedrockS3FullAccessPolicy', {
      statements: [
        new iam.PolicyStatement({
          actions: [
            's3:ListBucket',
            's3:GetObject',
            's3:PutObject',
            's3:DeleteObject',
          ],
          resources: [
            primaryKnowledgeBaseBucket.bucketArn,
            `${primaryKnowledgeBaseBucket.bucketArn}/*`,
          ],
        }),
      ],
    }); 

    // Foundation Model Policy for Bedrock
    const foundationModelPolicy = new iam.Policy(this, 'BedrockFoundationModelPolicy', {
      statements: [
        new iam.PolicyStatement({
          actions: [
            'bedrock:InvokeModel',
            'bedrock:InvokeModelWithResponseStream',
            'bedrock:GetFoundationModel',
            'bedrock:ListFoundationModels',
          ],
          resources: ['*'],
        }),
      ],
    });

    // Async Foundation Model Policy for Bedrock
    const asyncFoundationModelPolicy = new iam.Policy(this, 'BedrockAsyncFoundationModelPolicy', {
      statements: [
        new iam.PolicyStatement({
          actions: [
            'bedrock:InvokeModelAsync',
            'bedrock:GetModelInvocationJob',
            'bedrock:ListModelInvocationJobs',
          ],
          resources: ['*'],
        }),
      ],
    });

    // OpenSearch Serverless Policy for Bedrock
    const ossPolicy = new iam.Policy(this, 'BedrockOSSPolicy', {
      statements: [
        new iam.PolicyStatement({
          actions: ['aoss:APIAccessAll'],
          resources: [vectorCollection.attrArn],
        }),
      ],
    });

    // Attach policies to the role
    bedrockExecutionRole.attachInlinePolicy(s3FullAccessPolicy); 
    bedrockExecutionRole.attachInlinePolicy(foundationModelPolicy);
    bedrockExecutionRole.attachInlinePolicy(asyncFoundationModelPolicy);
    bedrockExecutionRole.attachInlinePolicy(ossPolicy);

    // Dependencies for OpenSearch Serverless
    vectorCollection.addDependency(securityPolicy);
    vectorCollection.addDependency(networkPolicy);

    // Lambda function to create OpenSearch index
    const createIndexLambda = new lambda.Function(this, 'CreateIndexFunction', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('lambda/create-index', {
        bundling: {
          image: lambda.Runtime.PYTHON_3_12.bundlingImage,
          command: [
            'bash', '-c',
            'pip install -r requirements.txt -t /asset-output && cp -au . /asset-output'
          ],
        },
      }),
      timeout: cdk.Duration.minutes(5),
      environment: {
        COLLECTION_ENDPOINT: vectorCollection.attrCollectionEndpoint,
        INDEX_NAME: `${PREFIX}-kb-default-index`,
        VECTOR_FIELD: 'bedrock-knowledge-base-default-vector',
        VECTOR_DIMENSIONS: EMBEDDING_MODEL_VECTOR_DIMENSIONS.toString(),
      },
    });

    // Grant Lambda permissions to access OpenSearch
    createIndexLambda.addToRolePolicy(new iam.PolicyStatement({
      actions: ['aoss:APIAccessAll'],
      resources: [vectorCollection.attrArn],
    }));

    // Add Lambda role to access policy
    const lambdaPrincipal = createIndexLambda.role!.roleArn;
    
    // Update access policy to include Lambda
    const accessPolicy = new opensearchserverless.CfnAccessPolicy(this, 'AccessPolicy', {
      name: `${PREFIX}-kb-access-policy`,
      type: 'data',
      policy: JSON.stringify([
        {
          Rules: [
            {
              ResourceType: 'collection',
              Resource: [`collection/${vectorCollection.name}`],
              Permission: [
                'aoss:CreateCollectionItems',
                'aoss:DeleteCollectionItems',
                'aoss:UpdateCollectionItems',
                'aoss:DescribeCollectionItems',
              ],
            },
            {
              ResourceType: 'index',
              Resource: [`index/${vectorCollection.name}/*`],
              Permission: [
                'aoss:CreateIndex',
                'aoss:DeleteIndex',
                'aoss:UpdateIndex',
                'aoss:DescribeIndex',
                'aoss:ReadDocument',
                'aoss:WriteDocument',
              ],
            },
          ],
          Principal: [
            bedrockExecutionRole.roleArn,
            lambdaPrincipal
          ],
        },
      ]),
    });

    accessPolicy.addDependency(vectorCollection);

    // Custom Resource to trigger index creation
    const indexProvider = new cr.Provider(this, 'IndexProvider', {
      onEventHandler: createIndexLambda,
    });

    const indexResource = new cdk.CustomResource(this, 'IndexResource', {
      serviceToken: indexProvider.serviceToken,
    });

    indexResource.node.addDependency(accessPolicy);

    // Bedrock Knowledge Base
    const knowledgeBase = new bedrock.CfnKnowledgeBase(this, 'KnowledgeBase', {
      name: `${PREFIX}-knowledge-base`,
      roleArn: bedrockExecutionRole.roleArn,
      knowledgeBaseConfiguration: {
        type: 'VECTOR',
        vectorKnowledgeBaseConfiguration: {
          embeddingModelArn: EMBEDDING_MODEL_ARN,
          // Only include dimension configuration for models that support it (Titan v2)
          ...(EMBEDDING_MODEL_ARN.includes('titan-embed-text-v2') && {
            embeddingModelConfiguration: {
              bedrockEmbeddingModelConfiguration: {
                dimensions: EMBEDDING_MODEL_VECTOR_DIMENSIONS,
              },
            },
          }),
        },
      },
      storageConfiguration: {
        type: 'OPENSEARCH_SERVERLESS',
        opensearchServerlessConfiguration: {
          collectionArn: vectorCollection.attrArn,
          vectorIndexName: `${PREFIX}-kb-default-index`,
          fieldMapping: {
            vectorField: 'bedrock-knowledge-base-default-vector',
            textField: 'AMAZON_BEDROCK_TEXT',
            metadataField: 'AMAZON_BEDROCK_METADATA',
          },
        },
      },
    });

    knowledgeBase.addDependency(accessPolicy);
    knowledgeBase.node.addDependency(indexResource);

    // Data Source for the Knowledge Base (using primary bucket)
    const dataSourceConfig: bedrock.CfnDataSource.DataSourceConfigurationProperty = {
      type: 'S3',
      s3Configuration: {
        bucketArn: primaryKnowledgeBaseBucket.bucketArn,
      },
    };

    // Build data source props with optional metadata extraction
    const dataSourceProps: bedrock.CfnDataSourceProps = {
      knowledgeBaseId: knowledgeBase.attrKnowledgeBaseId,
      name: 'sre-kb-data-source',
      dataSourceConfiguration: dataSourceConfig,
      ...(ENABLE_METADATA_EXTRACTION && {
        vectorIngestionConfiguration: {
          chunkingConfiguration: {
            chunkingStrategy: 'FIXED_SIZE',
            fixedSizeChunkingConfiguration: {
              maxTokens: 300,
              overlapPercentage: 20,
            },
          },
          parsingConfiguration: {
            parsingStrategy: 'BEDROCK_FOUNDATION_MODEL',
            bedrockFoundationModelConfiguration: {
              modelArn: 'arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0',
              parsingPrompt: {
                parsingPromptText: 'Extract all metadata fields from this JSON document, including is_child, is_parent, is_singleton, online_event, is_paid, age_restriction, duration_in_days, event_sales_status, and any other relevant fields.',
              },
            },
          },
        },
      }),
    };

    new bedrock.CfnDataSource(this, 'DataSource', dataSourceProps);

    // Outputs
    new cdk.CfnOutput(this, 'KnowledgeBaseId', {
      value: knowledgeBase.attrKnowledgeBaseId,
    });

    new cdk.CfnOutput(this, 'PrimaryS3BucketName', {
      value: primaryKnowledgeBaseBucket.bucketName,
    });
 
  }
}
