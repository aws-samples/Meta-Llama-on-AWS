# Amazon Bedrock Guardrails

## Introduction

[**Amazon Bedrock Guardrails**](https://aws.amazon.com/bedrock/guardrails/) enables AI developers to implement safeguards for your generative AI applications based on your responsible AI policies. You can tailor and create multiple guardrails to different use cases and apply them across multiple foundation models (FMs), providing consistent user experience and standardizing the safety and privacy controls across the applications.

You can configure the following policies in a guardrail to avoid undesirable and harmful content and remove sensitive information for privacy protection.
- **Content filters** – Adjust filter strengths to block input prompts or model responses containing harmful content.
- **Denied topics** – Define a set of topics that are undesirable in the context of your application. These topics will be blocked if detected in user queries or model responses.
- **Word filters** – Configure filters to block undesirable words, phrases, and profanity. Such words can include offensive terms, competitor names etc.
- **Sensitive information filters** – Block or mask sensitive information such as personally identifiable information (PII) or custom regex in user inputs and model responses.
- **Contextual grounding check** – Detect and filter hallucinations in model responses based on grounding in a source and relevance to the user query.

## How it works

- You can either use Amazon Bedrock Guardrails as API paramter for both `InvokeModel` and `Converse` APIs from **boto3** client. There are also native integrations available with **Agents for Amazon Bedrock** and **Knowledge Bases for Amazon Bedrock**.
- Alternatively, when you use 3rd party or self-hosted model, you can use `ApplyGuardrail` API as a standalone model evaluation on both input prompts and model response.

## Objectives

In this repository, we will showcase how we can utilize **contextual grounding check** policy from Amazon Bedrock Guardrails to detect hallucinations in model responses, specifically in [**RAG** (Retrieval augmented generation](https://aws.amazon.com/what-is/retrieval-augmented-generation/) application.

**Contextual grounding check** is a capability provided by Amazon Bedrock Guardrails that helps detect and filter out hallucinations in model responses. It evaluates responses based on two criteria;

1. **Grounding**, whether the response is factually accurate and based on the provided reference source, and
2. **Relevance**, whether the response answers the user's query.

This is important for applications like question-answering, summarization, and conversational AI that rely on **external sources** of information. Without grounding and relevance checks, the model could produce responses that are factually incorrect or completely irrelevant to the query, which would diminish the user experience.


## Pricing

Please refer to Amazon Bedrock [website](https://aws.amazon.com/bedrock/pricing/) for detailed pricing. 

