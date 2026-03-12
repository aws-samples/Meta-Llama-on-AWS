# Communication Templates

## Internal Incident Notifications

### SEV-1 Critical Incident Alert
```
🚨 SEV-1 INCIDENT ALERT 🚨

Service: [SERVICE_NAME]
Issue: [BRIEF_DESCRIPTION]
Impact: [AFFECTED_USERS/FUNCTIONALITY]
Started: [TIMESTAMP]
Status: [INVESTIGATING/MITIGATING/MONITORING]

Incident Commander: @[USERNAME]
War Room: [SLACK_CHANNEL]
Status Page: [URL]

Next Update: [TIME]
```

### SEV-2 High Impact Incident
```
⚠️ SEV-2 INCIDENT ⚠️

Service: [SERVICE_NAME]
Issue: [BRIEF_DESCRIPTION]
Impact: [AFFECTED_USERS/FUNCTIONALITY]
Started: [TIMESTAMP]
Status: [INVESTIGATING/MITIGATING/MONITORING]

Owner: @[USERNAME]
Channel: [SLACK_CHANNEL]

ETA: [ESTIMATED_RESOLUTION]
```

### Incident Resolution
```
✅ INCIDENT RESOLVED

Service: [SERVICE_NAME]
Issue: [BRIEF_DESCRIPTION]
Duration: [START_TIME] - [END_TIME]
Root Cause: [SUMMARY]

Resolution: [ACTIONS_TAKEN]
Follow-up: [POST_MORTEM_SCHEDULE]

Thanks to everyone who helped resolve this incident!
```

## Customer Communications

### Status Page - Initial Report
```
We are currently investigating reports of [ISSUE_DESCRIPTION] affecting [AFFECTED_SERVICES].

Our engineering team has been notified and is actively working to identify the root cause.

Users may experience [SPECIFIC_SYMPTOMS].

We will provide updates as more information becomes available.

Posted: [TIMESTAMP]
Next Update: [TIME]
```

### Status Page - Progress Update
```
Update: We have identified the issue as [ROOT_CAUSE_BRIEF].

Our team is implementing a fix and we expect service to be restored within [TIMEFRAME].

Affected users may continue to experience [SYMPTOMS] until the fix is deployed.

Workaround: [TEMPORARY_SOLUTION if applicable]

Posted: [TIMESTAMP]
Next Update: [TIME]
```

### Status Page - Resolution
```
✅ The issue has been resolved.

Root Cause: [DETAILED_EXPLANATION]
Resolution: [ACTIONS_TAKEN]
Duration: [TOTAL_TIME]

Service is now operating normally. We sincerely apologize for any inconvenience this may have caused.

If you continue to experience issues, please contact our support team.

Posted: [TIMESTAMP]
```

## Executive Communications

### Executive Summary - Critical Incident
```
Subject: URGENT - SEV-1 Incident affecting [SERVICE]

Executive Summary:
- Issue: [BRIEF_DESCRIPTION]
- Impact: [BUSINESS_IMPACT]
- Users Affected: [NUMBER/PERCENTAGE]
- Duration: [TIME_SINCE_START]
- Current Status: [INVESTIGATING/MITIGATING/RESOLVED]

Business Impact:
- Revenue Impact: [ESTIMATE if known]
- Customer Impact: [DESCRIPTION]
- Reputation Risk: [ASSESSMENT]

Response:
- Incident Commander: [NAME]
- Teams Involved: [LIST]
- ETA to Resolution: [ESTIMATE]

Next update in [TIMEFRAME].

[NAME]
[TITLE]
```

### Post-Incident Executive Brief
```
Subject: Post-Incident Brief - [SERVICE] Outage [DATE]

Incident Summary:
- Service: [SERVICE_NAME]
- Duration: [TOTAL_TIME]
- Root Cause: [EXPLANATION]
- Users Affected: [NUMBER/PERCENTAGE]

Impact Assessment:
- Revenue Impact: [AMOUNT if applicable]
- Customer Complaints: [NUMBER]
- SLA Breaches: [LIST if any]

Resolution Actions:
- Immediate Fix: [DESCRIPTION]
- Monitoring: [ENHANCED_MONITORING]
- Prevention: [LONG_TERM_MEASURES]

Lessons Learned:
- [KEY_INSIGHT_1]
- [KEY_INSIGHT_2]
- [KEY_INSIGHT_3]

Action Items:
1. [ACTION] - Owner: [NAME] - Due: [DATE]
2. [ACTION] - Owner: [NAME] - Due: [DATE]

Post-mortem meeting scheduled for [DATE/TIME].

[NAME]
[TITLE]
```

## Partner and Vendor Communications

### Partner Notification
```
Subject: Service Incident affecting [INTEGRATION/API]

Dear [PARTNER_NAME],

We are currently experiencing an incident that may affect our [API/SERVICE] integration with your platform.

Issue: [DESCRIPTION]
Affected Functionality: [LIST]
Expected Impact: [DESCRIPTION]
ETA to Resolution: [TIMEFRAME]

What this means for you:
- [SPECIFIC_IMPACT_1]
- [SPECIFIC_IMPACT_2]

Recommended Actions:
- [SUGGESTION_1]
- [SUGGESTION_2]

We will send updates as the situation evolves.

Technical Contact: [EMAIL]
Status Page: [URL]

Best regards,
[NAME]
[COMPANY]
```

## Customer Support Scripts

### Support Chat Response
```
We're aware of the current issue with [SERVICE/FEATURE] and our engineering team is actively working on a fix.

Here's what we know:
- Issue: [BRIEF_DESCRIPTION]
- Affected: [SCOPE]
- Status: [CURRENT_STATUS]
- ETA: [TIMEFRAME if available]

For real-time updates, please check our status page: [URL]

Is there anything specific about your experience I can help with while we work on this?
```

### Support Email Template
```
Subject: Re: [ISSUE] - We're working on a fix

Hi [CUSTOMER_NAME],

Thank you for reporting this issue. We're currently experiencing a service incident that is affecting [FUNCTIONALITY].

Current Status: [DESCRIPTION]
Expected Resolution: [TIMEFRAME]
Workaround: [TEMPORARY_SOLUTION if available]

We understand how frustrating this can be and we're working as quickly as possible to resolve it.

You can track our progress on our status page: [URL]

I'll personally follow up with you once this is resolved.

Best regards,
[SUPPORT_AGENT_NAME]
[COMPANY] Support Team
```

## Social Media Templates

### Twitter/X Status Update
```
We're currently investigating reports of issues with [SERVICE]. Our team is working on a fix. Updates: [STATUS_PAGE_URL] #[COMPANY]Status
```

### LinkedIn Professional Update
```
We're experiencing a service incident affecting [SERVICE_NAME]. Our engineering team is actively working on a resolution. We'll keep you updated on our progress via our status page: [URL]

We apologize for any inconvenience and appreciate your patience.
```

## Team Communication Guidelines

### Dos
- Be clear and concise
- Include timestamps in all updates
- Provide specific ETAs when possible
- Acknowledge uncertainty when present
- Thank team members for their contributions

### Don'ts
- Don't assign blame during active incidents
- Don't make promises you can't keep
- Don't use technical jargon in customer communications
- Don't delay communication while gathering perfect information
- Don't forget to update all stakeholders