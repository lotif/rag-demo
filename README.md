# RAG Demo

A simple and quick demo project for Retrieval Augmented Generation (RAG).

## Setup

Create and activate a virtual environment is by using the [virtualenv](https://pypi.org/project/virtualenv/) package
and install the requirements:

```shell
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
Download [LLama2 compiled for CPU](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)
from the link below into a `models/` folder:

https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin

## Run
```ssh
$ python
Python 3.11.3 (main, Apr  7 2023, 19:29:16) [Clang 14.0.0 (clang-1400.0.29.202)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from rag_utils import build_db
>>> build_db()
>>> from main import ask
>>> ask("Which telescope is the biggest NASA has ever made?")
```
