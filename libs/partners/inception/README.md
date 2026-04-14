# langchain-inception

This package contains the LangChain integration with [Inception](https://inceptionlabs.ai/).

## Installation

```bash
pip install -U langchain-inception
```

## Chat models

`ChatInception` class exposes chat models from Inception.

```python
from langchain_inception import ChatInception

llm = ChatInception(model="mercury-2")
llm.invoke("Hello, how are you?")
```

Set your API key with the `INCEPTION_API_KEY` environment variable, or pass
it directly:

```python
llm = ChatInception(model="mercury-2", api_key="your-api-key")
```
