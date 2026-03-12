# Troubleshooting

## Deployment Issues

### CDK bootstrap fails

```bash
# Verify credentials
aws sts get-caller-identity

# Bootstrap explicitly with account and region
cdk bootstrap aws://$(aws sts get-caller-identity --query Account --output text)/us-west-2
```

### Docker build fails during `cdk deploy`

Make sure Docker is running:
```bash
docker ps
```

If you get ARM64/platform errors, the Dockerfile targets `linux/arm64`. On Intel Macs or Linux x86, Docker Desktop handles cross-platform builds automatically. If using Finch or another runtime, ensure QEMU emulation is configured.

### Knowledge Base stack takes too long

OpenSearch Serverless collection creation can take 10-15 minutes. This is normal. If it fails, check:
```bash
aws cloudformation describe-stack-events \
  --stack-name SimpleBedrockStack --region us-west-2 \
  --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`].[LogicalResourceId,ResourceStatusReason]' \
  --output table
```

### AgentCore stack fails with "Runtime already exists"

If you're redeploying after a partial failure:
```bash
# Check if runtime exists
aws bedrock-agentcore list-runtimes --region us-west-2

# If stuck, destroy and redeploy
cd fast_agentcore/infra-cdk
cdk destroy --force
cdk deploy --require-approval never
```

---

## Runtime Issues

### "LLM response was not valid JSON"

The agent's LLM response was truncated. The max token limit may be too low. Check `fast_agentcore/patterns/sre-four-agent/src/orchestration/four_agent/settings.py`:

```python
BEDROCK_DEFAULT_MAX_TOKENS = 4096  # Increase if needed
```

After changing, redeploy:
```bash
cd fast_agentcore/infra-cdk
cdk deploy --require-approval never
```

### Agents return empty or generic responses

1. Check that logs exist in CloudWatch:
```bash
aws logs filter-log-events \
  --log-group-name /aws/banking/system-logs \
  --start-time $(($(date +%s) - 3600))000 \
  --region us-west-2 --limit 5
```

If empty, click "Generate Logs" in the frontend first.

2. Check that the Knowledge Base has documents:
```bash
aws bedrock-agent get-knowledge-base \
  --knowledge-base-id YOUR_KB_ID --region us-west-2
```

3. Check AgentCore runtime logs:
```bash
aws logs tail /aws/bedrock-agentcore/runtimes/ --since 10m --region us-west-2
```

### "Access denied" errors in agent logs

The AgentCore execution role may be missing permissions. Common ones:

- `bedrock:Retrieve` — needed for Knowledge Base queries. Check `agentcore-role.ts` includes this.
- `logs:FilterLogEvents` — needed for CloudWatch log reading. Must include the `/aws/banking/system-logs` log group.

After updating `agentcore-role.ts`, redeploy:
```bash
cd fast_agentcore/infra-cdk
cdk deploy --require-approval never
```

---

## Frontend Issues

### Login page shows but can't sign in

1. Verify the Cognito user exists:
```bash
aws cognito-idp admin-get-user \
  --user-pool-id YOUR_USER_POOL_ID \
  --username your-username \
  --region us-west-2
```

2. If user status is `FORCE_CHANGE_PASSWORD`, sign in with the temporary password — you'll be prompted to set a new one.

3. Check that `aws-exports.json` has the correct `authority`, `client_id`, and `redirect_uri` matching your Amplify URL.

### Chat messages are unreadable (dark text on dark background)

The frontend uses a dark theme. If assistant messages appear dark, check `ChatMessage.tsx` — assistant messages should use `text-gray-100`, not `text-gray-800`.

### "Generate Logs" button doesn't work

1. Check that `logGeneratorApiUrl` is set in `aws-exports.json`
2. The API Gateway uses a Cognito authorizer — the frontend must send an `Authorization: Bearer <access_token>` header (not `id_token`)
3. Check the Lambda logs:
```bash
aws logs tail /aws/lambda/BankLogGeneratorStack-BankLogGenerator --since 10m --region us-west-2
```

### Frontend deployment fails

If `deploy-frontend.py` fails, deploy manually:
```bash
cd fast_agentcore/frontend
npm install
npm run build
cd build
zip -r ../frontend-build.zip .
cd ..
aws s3 cp frontend-build.zip s3://YOUR_STAGING_BUCKET/frontend-build.zip
aws amplify start-deployment \
  --app-id YOUR_AMPLIFY_APP_ID \
  --branch-name main \
  --source-url s3://YOUR_STAGING_BUCKET/frontend-build.zip
```

---

## Checking Logs

### AgentCore Runtime Logs (agent execution)

```bash
# Tail recent logs
aws logs tail /aws/bedrock-agentcore/runtimes/ --since 30m --region us-west-2

# Filter for errors
aws logs filter-log-events \
  --log-group-name /aws/bedrock-agentcore/runtimes/YOUR_RUNTIME_ID-DEFAULT \
  --filter-pattern "ERROR" \
  --start-time $(($(date +%s) - 1800))000 \
  --region us-west-2
```

### Banking Logs (generated test data)

```bash
aws logs tail /aws/banking/system-logs --since 1h --region us-west-2
```

### Lambda Logs

```bash
# Log generator
aws logs tail /aws/lambda/BankLogGeneratorStack-BankLogGenerator --since 30m --region us-west-2

# Feedback API
aws logs tail /aws/lambda/meta-sre-agent-feedback --since 30m --region us-west-2
```

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Setting `agentPattern` to `sre-four-agent` in aws-exports.json | Must be `langgraph-single-agent` — this controls the frontend streaming parser |
| Forgetting to upload KB documents after deploying the KB stack | Run `aws s3 cp docs/policies/ s3://$KB_BUCKET/policies/ --recursive` |
| Forgetting to trigger KB sync after uploading documents | Run `aws bedrock-agent start-ingestion-job ...` |
| Using `deployment_type: zip` in config.yaml | Must be `docker` for the sre-four-agent pattern |
| Not having Docker running during `cdk deploy` | Start Docker Desktop before deploying |
| Zipping the `build/` folder itself instead of its contents | `cd build && zip -r ../out.zip .` (zip from inside the directory) |
