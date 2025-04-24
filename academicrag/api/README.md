# AcademicRAG Server and WebUI

The AcademicRAG Server is designed to provide Web UI and API support. The Web UI facilitates document indexing, knowledge graph exploration, and a simple RAG query interface. AcademicRAG Server also provide an Ollama compatible interfaces, aiming to emulate AcademicRAG as an Ollama chat model. This allows AI chat bot, such as Open WebUI, to access AcademicRAG easily.



## Getting Start

### Installation

* Installation from Source

```bash
# Change to the repository directory
cd academicrag

# create a Python virtual enviroment if neccesary
# Install in editable mode with API support
pip install -e ".[api]"
```

### Before Starting AcademicRAG Server

AcademicRAG necessitates the integration of both an LLM (Large Language Model) and an Embedding Model to effectively execute document indexing and querying operations. Prior to the initial deployment of the AcademicRAG server, it is essential to configure the settings for both the LLM and the Embedding Model. AcademicRAG supports binding to various LLM/Embedding backends:

* ollama
* lollms
* openai or openai compatible
* azure_openai

It is recommended to use environment variables to configure the AcademicRAG Server. There is an example environment variable file named `env.example` in the root directory of the project. Please copy this file to the startup directory and rename it to `.env`. After that, you can modify the parameters related to the LLM and Embedding models in the `.env` file. It is important to note that the AcademicRAG Server will load the environment variables from `.env` into the system environment variables each time it starts. Since the AcademicRAG Server will prioritize the settings in the system environment variables, if you modify the `.env` file after starting the AcademicRAG Server via the command line, you need to execute `source .env` to make the new settings take effect.

Here are some examples of common settings for LLM and Embedding models：

* OpenAI LLM + Ollama Embedding

```
LLM_BINDING=openai
LLM_MODEL=gpt-4o
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key
MAX_TOKENS=32768                # Max tokens send to LLM (less than model context size)

EMBEDDING_BINDING=ollama
EMBEDDING_BINDING_HOST=http://localhost:11434
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
# EMBEDDING_BINDING_API_KEY=your_api_key
```

* Ollama LLM + Ollama Embedding

```
LLM_BINDING=ollama
LLM_MODEL=mistral-nemo:latest
LLM_BINDING_HOST=http://localhost:11434
# LLM_BINDING_API_KEY=your_api_key
MAX_TOKENS=8192                  # Max tokens send to LLM (base on your Ollama Server capacity)

EMBEDDING_BINDING=ollama
EMBEDDING_BINDING_HOST=http://localhost:11434
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
# EMBEDDING_BINDING_API_KEY=your_api_key
```

### Starting AcademicRAG Server

The AcademicRAG Server supports two operational modes:
* The simple and efficient Uvicorn mode

```
academicrag-server
```
* The multiprocess Gunicorn + Uvicorn mode (production mode, not supported on Windows environments)

```
academicrag-gunicorn --workers 4
```
The `.env` file must be placed in the startup directory. Upon launching, the AcademicRAG Server will create a documents directory (default is `./inputs`) and a data directory (default is `./rag_storage`). This allows you to initiate multiple instances of AcademicRAG Server from different directories, with each instance configured to listen on a distinct network port.

Here are some common used startup parameters:

- `--host`: Server listening address (default: 0.0.0.0)
- `--port`: Server listening port (default: 9621)
- `--timeout`: LLM request timeout (default: 150 seconds)
- `--log-level`: Logging level (default: INFO)
- --input-dir: specifying the directory to scan for documents (default: ./input)

### Auto scan on startup

When starting any of the servers with the `--auto-scan-at-startup` parameter, the system will automatically:

1. Scan for new files in the input directory
2. Indexing new documents that aren't already in the database
3. Make all content immediately available for RAG queries

> The `--input-dir` parameter specify the input directory to scan for. You can trigger input diretory scan from webui.

### Multiple workers for Gunicorn + Uvicorn

The AcademicRAG Server can operate in the `Gunicorn + Uvicorn` preload mode. Gunicorn's Multiple Worker (multiprocess) capability prevents document indexing tasks from blocking RAG queries.  Using CPU-exhaustive document extraction tools, such as docling, can lead to the entire system being blocked in pure Uvicorn mode.

Though AcademicRAG Server uses one workers to process the document indexing pipeline, with aysnc task supporting of Uvicorn, multiple files can be processed in parallell. The bottleneck of document indexing speed mainly lies with the LLM. If your LLM supports high concurrency, you can accelerate document indexing by increasing the concurrency level of the LLM. Below are several environment variables related to concurrent processing, along with their default values:

```
WORKERS=2                    # Num of worker processes, not greater then (2 x number_of_cores) + 1
MAX_PARALLEL_INSERT=2        # Num of parallel files to process in one batch
MAX_ASYNC=4                  # Max concurrency requests of LLM
```

### Install Academicrag as a Linux Service

Create a your service file `academicrag.sevice` from the sample file : `academicrag.sevice.example`. Modified the WorkingDirectoryand EexecStart in the service file:

```text
Description=AcademicRAG Ollama Service
WorkingDirectory=<academicrag installed directory>
ExecStart=<academicrag installed directory>/academicrag/api/academicrag-api
```

Modify your service startup script: `academicrag-api`. Change you python virtual environment activation command as needed:

```shell
#!/bin/bash

# your python virtual environment activation
source /home/netman/academicrag-xyj/venv/bin/activate
# start academicrag api server
academicrag-server
```

Install AcademicRAG service. If your system is Ubuntu, the following commands will work:

```shell
sudo cp academicrag.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start academicrag.service
sudo systemctl status academicrag.service
sudo systemctl enable academicrag.service
```

## Ollama Emulation

We provide an Ollama-compatible interfaces for AcademicRAG, aiming to emulate AcademicRAG as an Ollama chat model. This allows AI chat frontends supporting Ollama, such as Open WebUI, to access AcademicRAG easily.

### Connect Open WebUI to AcademicRAG

After starting the academicrag-server, you can add an Ollama-type connection in the Open WebUI admin pannel. And then a model named academicrag:latest will appear in Open WebUI's model management interface. Users can then send queries to AcademicRAG through the chat interface. You'd better install AcademicRAG as service for this use case.

Open WebUI's use LLM to do the session title and session keyword generation task. So the Ollama chat chat completion API detects and forwards OpenWebUI session-related requests directly to underlying LLM. 

### Choose Query mode in chat

A query prefix in the query string can determines which AcademicRAG query mode is used to generate the respond for the query. The supported prefixes include:

```
/subgraph
/global
/hybrid
/naive
/mix
/bypass
```

For example, chat message "/mix 唐僧有几个徒弟" will trigger a mix mode query for LighRAG. A chat message without query prefix will trigger a hybrid mode query by default。

"/bypass" is not a AcademicRAG query mode, it will tell API Server to pass the query directly to the underlying LLM with chat history. So user can use LLM to answer question base on the chat history. If you are using Open WebUI as front end, you can just switch the model to a normal LLM instead of using /bypass prefix.



## API-Key and Authentication

By default, the AcademicRAG Server can be accessed without any authentication. We can configure the server with an API-Key or account credentials to secure it.

* API-KEY

```
ACADEMICRAG_API_KEY=your-secure-api-key-here
WHITELIST_PATHS=/health,/api/*
```

> Health check and Ollama emuluation endpoins is exclude from API-KEY check by default.

* Account credentials (the web UI requires login before access)

AcademicRAG API Server implements JWT-based authentication using HS256 algorithm. To enable secure access control, the following environment variables are required:

```bash
# For jwt auth
AUTH_USERNAME=admin      # login name
AUTH_PASSWORD=admin123   # password
TOKEN_SECRET=your-key    # JWT key
TOKEN_EXPIRE_HOURS=4     # expire duration
```

> Currently, only the configuration of an administrator account and password is supported. A comprehensive account system is yet to be developed and implemented.

If Account credentials are not configured, the web UI will access the system as a Guest. Therefore, even if only API-KEY is configured, all API can still be accessed through the Guest account, which remains insecure. Hence, to safeguard the API, it is necessary to configure both authentication methods simultaneously.



## For Azure OpenAI Backend

Azure OpenAI API can be created using the following commands in Azure CLI (you need to install Azure CLI first from [https://docs.microsoft.com/en-us/cli/azure/install-azure-cli](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)):

```bash
# Change the resource group name, location and OpenAI resource name as needed
RESOURCE_GROUP_NAME=AcademicRAG
LOCATION=swedencentral
RESOURCE_NAME=AcademicRAG-OpenAI

az login
az group create --name $RESOURCE_GROUP_NAME --location $LOCATION
az cognitiveservices account create --name $RESOURCE_NAME --resource-group $RESOURCE_GROUP_NAME  --kind OpenAI --sku S0 --location swedencentral
az cognitiveservices account deployment create --resource-group $RESOURCE_GROUP_NAME  --model-format OpenAI --name $RESOURCE_NAME --deployment-name gpt-4o --model-name gpt-4o --model-version "2024-08-06"  --sku-capacity 100 --sku-name "Standard"
az cognitiveservices account deployment create --resource-group $RESOURCE_GROUP_NAME  --model-format OpenAI --name $RESOURCE_NAME --deployment-name text-embedding-3-large --model-name text-embedding-3-large --model-version "1"  --sku-capacity 80 --sku-name "Standard"
az cognitiveservices account show --name $RESOURCE_NAME --resource-group $RESOURCE_GROUP_NAME --query "properties.endpoint"
az cognitiveservices account keys list --name $RESOURCE_NAME -g $RESOURCE_GROUP_NAME

```

The output of the last command will give you the endpoint and the key for the OpenAI API. You can use these values to set the environment variables in the `.env` file.

```
# Azure OpenAI Configuration in .env
LLM_BINDING=azure_openai
LLM_BINDING_HOST=your-azure-endpoint
LLM_MODEL=your-model-deployment-name
LLM_BINDING_API_KEY=your-azure-api-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview  # optional, defaults to latest version
EMBEDDING_BINDING=azure_openai  # if using Azure OpenAI for embeddings
EMBEDDING_MODEL=your-embedding-deployment-name

```



## AcademicRAG Server Configuration in Detail

API Server can be config in three way (highest priority first):

* Command line arguments
* Enviroment variables or .env file
* Config.ini (Only for storage configuration)

Most of the configurations come with a default settings, check out details in sample file: `.env.example`. Datastorage configuration can be also set by config.ini. A sample file `config.ini.example` is provided for your convenience.

### LLM and Embedding Backend Supported

AcademicRAG supports binding to various LLM/Embedding backends:

* ollama
* lollms
* openai & openai compatible
* azure_openai

Use environment variables  `LLM_BINDING` or CLI argument `--llm-binding` to select LLM backend type. Use environment variables  `EMBEDDING_BINDING` or CLI argument `--embedding-binding` to select LLM backend type.

### Entity Extraction Configuration
* ENABLE_LLM_CACHE_FOR_EXTRACT: Enable LLM cache for entity extraction (default: true)

It's very common to set `ENABLE_LLM_CACHE_FOR_EXTRACT` to true for test environment to reduce the cost of LLM calls.

### Storage Types Supported

AcademicRAG uses 4 types of storage for difference purposes:

* KV_STORAGE：llm response cache, text chunks, document information
* VECTOR_STORAGE：entities vectors, relation vectors, chunks vectors
* GRAPH_STORAGE：entity relation graph
* DOC_STATUS_STORAGE：documents indexing status

Each storage type have servals implementations:

* KV_STORAGE supported implement-name

```
JsonKVStorage    JsonFile(default)
RedisKVStorage   Redis
```

* GRAPH_STORAGE supported implement-name

```
NetworkXStorage      NetworkX(defualt)
Neo4JStorage         Neo4J
```

* VECTOR_STORAGE supported implement-name

```
NanoVectorDBStorage         NanoVector(default)
MilvusVectorDBStorage       Milvus
ChromaVectorDBStorage       Chroma
FaissVectorDBStorage        Faiss
QdrantVectorDBStorage       Qdrant
```

* DOC_STATUS_STORAGE：supported implement-name

```
JsonDocStatusStorage        JsonFile(default)
```

### How Select Storage Implementation

You can select storage implementation by environment variables. Your can set the following environmental variables to a specific storage implement-name before the your first start of the API  Server:

```
ACADEMICRAG_KV_STORAGE=JsonKVStorage
ACADEMICRAG_VECTOR_STORAGE=NanoVector
ACADEMICRAG_GRAPH_STORAGE=NetworkX
ACADEMICRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage
```

You can not change storage implementation selection after you add documents to AcademicRAG. Data migration from one storage implementation to anthor is not supported yet. For further information please read the sample env file or config.ini file.

### AcademicRAG API Server Comand Line Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| --host | 0.0.0.0 | Server host |
| --port | 9621 | Server port |
| --working-dir | ./rag_storage | Working directory for RAG storage |
| --input-dir | ./inputs | Directory containing input documents |
| --max-async | 4 | Maximum async operations |
| --max-tokens | 32768 | Maximum token size |
| --timeout | 150 | Timeout in seconds. None for infinite timeout(not recommended) |
| --log-level | INFO | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| --verbose | - | Verbose debug output (True, Flase) |
| --key | None | API key for authentication. Protects academicrag server against unauthorized access |
| --ssl | False | Enable HTTPS |
| --ssl-certfile | None | Path to SSL certificate file (required if --ssl is enabled) |
| --ssl-keyfile | None | Path to SSL private key file (required if --ssl is enabled) |
| --top-k | 50 | Number of top-k items to retrieve; corresponds to entities in "local" mode and relationships in "global" mode. |
| --cosine-threshold | 0.4 | The cossine threshold for nodes and relations retrieval, works with top-k to control the retrieval of nodes and relations. |
| --llm-binding | ollama | LLM binding type (lollms, ollama, openai, openai-ollama, azure_openai) |
| --embedding-binding | ollama | Embedding binding type (lollms, ollama, openai, azure_openai) |
| auto-scan-at-startup | - | Scan input directory for new files and start indexing |

### Example Usage

#### Running a Academicrag server with ollama default local server as llm and embedding backends

Ollama is the default backend for both llm and embedding, so by default you can run academicrag-server with no parameters and the default ones will be used. Make sure ollama is installed and is running and default models are already installed on ollama.

```bash
# Run academicrag with ollama, mistral-nemo:latest for llm, and bge-m3:latest for embedding
academicrag-server

# Using an authentication key
academicrag-server --key my-key
```

#### Running a Academicrag server with lollms default local server as llm and embedding backends

```bash
# Run academicrag with lollms, mistral-nemo:latest for llm, and bge-m3:latest for embedding
# Configure LLM_BINDING=lollms and EMBEDDING_BINDING=lollms in .env or config.ini
academicrag-server

# Using an authentication key
academicrag-server --key my-key
```

#### Running a Academicrag server with openai server as llm and embedding backends

```bash
# Run academicrag with openai, GPT-4o-mini for llm, and text-embedding-3-small for embedding
# Configure in .env or config.ini:
# LLM_BINDING=openai
# LLM_MODEL=GPT-4o-mini
# EMBEDDING_BINDING=openai
# EMBEDDING_MODEL=text-embedding-3-small
academicrag-server

# Using an authentication key
academicrag-server --key my-key
```

#### Running a Academicrag server with azure openai server as llm and embedding backends

```bash
# Run academicrag with azure_openai
# Configure in .env or config.ini:
# LLM_BINDING=azure_openai
# LLM_MODEL=your-model
# EMBEDDING_BINDING=azure_openai
# EMBEDDING_MODEL=your-embedding-model
academicrag-server

# Using an authentication key
academicrag-server --key my-key
```

**Important Notes:**
- For LoLLMs: Make sure the specified models are installed in your LoLLMs instance
- For Ollama: Make sure the specified models are installed in your Ollama instance
- For OpenAI: Ensure you have set up your OPENAI_API_KEY environment variable
- For Azure OpenAI: Build and configure your server as stated in the Prequisites section

For help on any server, use the --help flag:
```bash
academicrag-server --help
```

## API Endpoints

All servers (LoLLMs, Ollama, OpenAI and Azure OpenAI) provide the same REST API endpoints for RAG functionality. When API Server is running, visit:

- Swagger UI: http://localhost:9621/docs
- ReDoc: http://localhost:9621/redoc

You can test the API endpoints using the provided curl commands or through the Swagger UI interface. Make sure to:

1. Start the appropriate backend service (LoLLMs, Ollama, or OpenAI)
2. Start the RAG server
3. Upload some documents using the document management endpoints
4. Query the system using the query endpoints
5. Trigger document scan if new files is put into inputs directory

### Query Endpoints

#### POST /query
Query the RAG system with options for different search modes.

```bash
curl -X POST "http://localhost:9621/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "Your question here", "mode": "hybrid"}'
```

#### POST /query/stream
Stream responses from the RAG system.

```bash
curl -X POST "http://localhost:9621/query/stream" \
    -H "Content-Type: application/json" \
    -d '{"query": "Your question here", "mode": "hybrid"}'
```

### Document Management Endpoints

#### POST /documents/text
Insert text directly into the RAG system.

```bash
curl -X POST "http://localhost:9621/documents/text" \
    -H "Content-Type: application/json" \
    -d '{"text": "Your text content here", "description": "Optional description"}'
```

#### POST /documents/file
Upload a single file to the RAG system.

```bash
curl -X POST "http://localhost:9621/documents/file" \
    -F "file=@/path/to/your/document.txt" \
    -F "description=Optional description"
```

#### POST /documents/batch
Upload multiple files at once.

```bash
curl -X POST "http://localhost:9621/documents/batch" \
    -F "files=@/path/to/doc1.txt" \
    -F "files=@/path/to/doc2.txt"
```

#### POST /documents/scan

Trigger document scan for new files in the Input directory.

```bash
curl -X POST "http://localhost:9621/documents/scan" --max-time 1800
```

> Ajust max-time according to the estimated index time  for all new files.

#### DELETE /documents

Clear all documents from the RAG system.

```bash
curl -X DELETE "http://localhost:9621/documents"
```

### Ollama Emulation Endpoints

#### GET /api/version

Get Ollama version information.

```bash
curl http://localhost:9621/api/version
```

#### GET /api/tags

Get Ollama available models.

```bash
curl http://localhost:9621/api/tags
```

#### POST /api/chat

Handle chat completion requests. Routes user queries through AcademicRAG by selecting query mode based on query prefix. Detects and forwards OpenWebUI session-related requests (for meta data generation task) directly to underlying LLM.

```shell
curl -N -X POST http://localhost:9621/api/chat -H "Content-Type: application/json" -d \
  '{"model":"academicrag:latest","messages":[{"role":"user","content":"猪八戒是谁"}],"stream":true}'
```

> For more information about Ollama API pls. visit :  [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)

#### POST /api/generate

Handle generate completion requests. For compatibility purpose, the request is not processed by AcademicRAG, and will be handled by underlying LLM model.

### Utility Endpoints

#### GET /health
Check server health and configuration.

```bash
curl "http://localhost:9621/health"
```
