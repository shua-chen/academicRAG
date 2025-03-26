import os
import asyncio
from academicrag import AcademicRAG, QueryParam
from academicrag.llm.openai import openai_complete_if_cache
from academicrag.llm.siliconcloud import siliconcloud_embedding
from academicrag.utils import EmbeddingFunc
from academicrag.kg.shared_storage import initialize_pipeline_status
import numpy as np

WORKING_DIR = "./test"

os.environ["SILICONFLOW_API_KEY"] = "your_api_key"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url="https://api.siliconflow.cn/v1/",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await siliconcloud_embedding(
        texts,
        model="BAAI/bge-m3",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        max_token_size=8196,
    )



async def initialize_rag():

    rag = AcademicRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=embedding_func)
    )

    await rag.initialize_storages()

    await initialize_pipeline_status()

    return rag

rag = asyncio.run(initialize_rag())

with open("data/machine_learning.txt", encoding="utf-8") as f:
    rag.insert(f.read())

# # Perform naive search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
# )

# # Perform local search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="subgraph"))
# )

# # Perform global search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
# )

# Perform hybrid search
print(
    rag.query("What is the histry of machine learning?", param=QueryParam(mode="hybrid"))
)