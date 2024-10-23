# Responsible AI

This repository contains the examples to help customers get started with **Responsible AI** by utilizing [Amazon Bedrock Guardrails](https://aws.amazon.com/bedrock/guardrails/) and other open source tools with [Meta Llama on Amazon Bedrock](https://aws.amazon.com/bedrock/llama/).

## Background

**Responsible AI** refers to the practice of developing and deploying artificial intelligence systems in an ethical, transparent, and accountable manner, ensuring they align with societal values and minimize potential harm. In the context of generative AI, which involves models that can generate human-like text, images, or other content, responsible AI practices are crucial due to the significant risks and challenges associated with these powerful technologies.

**Evaluation** plays a vital role in responsible generative AI because it allows for the identification and mitigation of potential issues before deployment. Comprehensive evaluation frameworks, like DeepEval, RAGAS, enable organizations to assess their generative models for risks such as generating biased, toxic, or harmful content, hallucinating false information, or exhibiting other undesirable behaviors. By thoroughly evaluating models across various dimensions, including safety, factual accuracy, robustness, and fairness, organizations can build trust and confidence in their generative AI systems. Responsible evaluation also promotes transparency, as it provides insights into the strengths, limitations, and potential failure modes of these models, enabling informed decision-making and responsible deployment strategies.

For more details on Responsible AI on AWS, please visit [here](https://aws.amazon.com/ai/responsible-ai/) and this [generative AI scoping matrix](https://aws.amazon.com/ai/generative-ai/security/scoping-matrix/).

## Prerequisite

- First, please ensure that you have the access the foundation models on Amazon Bedrock, you can follow this [documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) for details walkthrough.

- Throughout this repository, we will use Amazon Shareholder's letter as our datasources, and use [langchain chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/) as our vector database. Please execute the prerequisite notebook before exploring each evaluation framework.


## Contents
- [**Amazon Bedrock Guardrails**](./BedrockGuardrailsContextualGrounding) - Amazon Bedrock Guardrails support **contextual grounding check**, which can use to detect and filter hallucinations in model responses when a reference source and user query are provided. This is done by checking for relevance for each chunk processed from RAG application. If any one chunk is deemded relevant, the whole response is considered relevant as it has the answer to the user query. Please refer to the [documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-contextual-grounding-check.html) for more details.
- [**DeepEval**](./DeepEval) - DeepEval is an open-source comprehensive evaluation framework by [**ConfidentAI**](https://docs.confident-ai.com/). It designed to assess the safety, reliability, and performance of large language models (LLMs) and [Retrieval augmented generation (RAG) systems](https://aws.amazon.com/what-is/retrieval-augmented-generation/). The key benefit of DeepEval is that it enables organizations to thoroughly evaluate their generative AI models before deployment, ensuring they meet the necessary standards for responsible and trustworthy AI. By identifying and mitigating potential issues early on, DeepEval helps build confidence in the safe and ethical use of generative AI technologies.
- **LlamaIndex Evaluation** - [LlamaIndex Evaluation](https://docs.llamaindex.ai/en/stable/optimizing/evaluation/evaluation/) is a framework within the LlamaIndex library that allows developers to assess and compare the performance of various components in their RAG application. It provides tools to evaluate query engines, retrievers, and other elements against predefined metrics such as relevance, coherence, and factual accuracy. The benefit of LlamaIndex Evaluation is that it enables developers to fine-tune their systems, identify areas for improvement, and ensure the reliability and effectiveness of their AI applications.
- **RAGAS** - Or, **Retrieval-Augmented Generation Automated Scoring** is a framework specifically designed for evaluating the performance of generative AI models in tasks that involve retrieving and synthesizing information from external sources. It addresses the need for automated evaluation metrics that can accurately assess the quality of generated outputs when models have access to external knowledge sources. Please refer to [RAGAS documentation](https://docs.ragas.io/en/stable/) for more details.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
