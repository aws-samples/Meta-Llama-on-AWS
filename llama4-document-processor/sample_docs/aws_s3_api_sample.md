# AWS S3 API Documentation

## Base URL
```
https://s3.amazonaws.com
```

## Authentication
All requests must be signed using AWS Signature Version 4. Include your AWS credentials:
- Access Key ID
- Secret Access Key
- Region (e.g., us-east-1)

## Create Bucket

Create a new S3 bucket.

**Endpoint:** `PUT /{bucket-name}`

**Headers:**
- `Authorization: AWS4-HMAC-SHA256 Credential=...`
- `x-amz-content-sha256: UNSIGNED-PAYLOAD`
- `x-amz-date: 20231201T120000Z`

**Example Request:**
```bash
curl -X PUT https://my-bucket.s3.amazonaws.com/ \
  -H "Authorization: AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20231201/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-date, Signature=..." \
  -H "x-amz-date: 20231201T120000Z"
```

## Upload Object

Upload a file to S3 bucket.

**Endpoint:** `PUT /{bucket-name}/{object-key}`

**Parameters:**
- `bucket-name` (required): Name of the S3 bucket
- `object-key` (required): Key/path for the object

**Example Request:**
```bash
curl -X PUT https://my-bucket.s3.amazonaws.com/my-file.txt \
  -H "Authorization: AWS4-HMAC-SHA256 ..." \
  -H "Content-Type: text/plain" \
  --data-binary @my-file.txt
```

## Download Object

Retrieve an object from S3.

**Endpoint:** `GET /{bucket-name}/{object-key}`

**Example Request:**
```bash
curl https://my-bucket.s3.amazonaws.com/my-file.txt \
  -H "Authorization: AWS4-HMAC-SHA256 ..."
```

## List Objects

List objects in a bucket.

**Endpoint:** `GET /{bucket-name}?list-type=2`

**Query Parameters:**
- `list-type=2` (required): Use version 2 of list API
- `prefix` (optional): Filter objects by prefix
- `max-keys` (optional): Maximum number of objects to return

**Example Request:**
```bash
curl "https://my-bucket.s3.amazonaws.com/?list-type=2&prefix=photos/" \
  -H "Authorization: AWS4-HMAC-SHA256 ..."
```

**Example Response:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult>
    <Name>my-bucket</Name>
    <Prefix>photos/</Prefix>
    <KeyCount>2</KeyCount>
    <Contents>
        <Key>photos/image1.jpg</Key>
        <Size>1024</Size>
        <LastModified>2023-12-01T12:00:00.000Z</LastModified>
    </Contents>
</ListBucketResult>
```

## Delete Object

Delete an object from S3.

**Endpoint:** `DELETE /{bucket-name}/{object-key}`

**Example Request:**
```bash
curl -X DELETE https://my-bucket.s3.amazonaws.com/my-file.txt \
  -H "Authorization: AWS4-HMAC-SHA256 ..."
```

## Error Handling

Common HTTP status codes:
- `200` - Success
- `403` - Forbidden (check credentials)
- `404` - Not Found (bucket or object doesn't exist)
- `409` - Conflict (bucket already exists)

**Error Response Format:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<Error>
    <Code>NoSuchBucket</Code>
    <Message>The specified bucket does not exist</Message>
    <BucketName>my-bucket</BucketName>
</Error>
```

## SDK Usage

AWS provides SDKs for multiple languages:
- Python: boto3
- JavaScript: AWS SDK for JavaScript
- Java: AWS SDK for Java
- .NET: AWS SDK for .NET