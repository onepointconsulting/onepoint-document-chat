
# Onepoint Document Chat

Allows to chat with PDF documents based on a set of uploaded documents.

## Setup

We suggest to use [Conda](https://docs.conda.io/en/latest/) to manage the virtual environment and then install poetry.

```
conda activate base
conda remove -n onepoint_document_chat --all
conda create -n onepoint_document_chat python=3.11
conda activate onepoint_document_chat
pip install poetry
``````

Also copy the content of the binary of this repository (https://github.com/onepointconsulting/chatbot-ui.git) to a new `ui` folder.

## Running the server

```
python ./onepoint_document_chat/server/websocket_server.py
```