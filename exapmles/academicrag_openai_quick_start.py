import os
import asyncio
from academicrag import AcademicRAG, QueryParam
from academicrag.llm.openai import gpt_4o_mini_complete, openai_embed
from academicrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():


    rag = AcademicRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed
    )

    await rag.initialize_storages()

    await initialize_pipeline_status()

    return rag

rag = asyncio.run(initialize_rag())

with open("./book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform naive search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="subgraph"))
)

# Perform global search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
)
