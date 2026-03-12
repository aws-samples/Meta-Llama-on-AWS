from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import os
import boto3
import json
import logging
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    logger.info(f"Event: {json.dumps(event)}")
    
    host = os.environ['COLLECTION_ENDPOINT'].replace('https://', '')
    index_name = os.environ['INDEX_NAME']
    vector_field = os.environ['VECTOR_FIELD']
    vector_dimensions = int(os.environ.get('VECTOR_DIMENSIONS', '1024'))
    region = os.environ['AWS_REGION']
    
    session = boto3.Session()
    credentials = session.get_credentials()
    
    auth = AWSV4SignerAuth(credentials, region, 'aoss')
    
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )
    
    try:
        if event['RequestType'] == 'Create':
            logger.info(f"Creating index: {index_name}")
            
            # Wait for access policy to propagate
            logger.info("Waiting 30 seconds for access policy to propagate...")
            time.sleep(30)
            
            index_body = {
                'settings': {
                    'index.knn': True,
                },
                'mappings': {
                    'properties': {
                        vector_field: {
                            'type': 'knn_vector',
                            'dimension': vector_dimensions,
                            'method': {
                                'space_type': 'l2',
                                'engine': 'faiss',
                                'name': 'hnsw',
                            },
                        },
                        'AMAZON_BEDROCK_METADATA': {'type': 'text', 'index': False},
                        'AMAZON_BEDROCK_TEXT': {'type': 'text'},
                    }
                },
            }
            
            response = client.indices.create(index_name, body=index_body)
            logger.info(f"Index created: {response}")
            
            # Wait longer for index to be fully available in OpenSearch Serverless
            logger.info("Waiting 60 seconds for index to be fully available...")
            time.sleep(60)
            
        elif event['RequestType'] == 'Delete':
            logger.info(f"Deleting index: {index_name}")
            if client.indices.exists(index_name):
                client.indices.delete(index_name)
                
        return {
            'PhysicalResourceId': index_name,
            'Data': {'IndexName': index_name}
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
