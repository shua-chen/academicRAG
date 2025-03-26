import os
import asyncio
from academicrag import AcademicRAG, QueryParam
from academicrag.llm.openai import openai_complete_if_cache, openai_embed
from academicrag.utils import EmbeddingFunc
from academicrag.kg.shared_storage import initialize_pipeline_status
import numpy as np

WORKING_DIR = "./test"

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "model_name",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="model_name",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


# asyncio.run(test_funcs())

async def initialize_rag():

    embedding_dimension = await get_embedding_dim()

    rag = AcademicRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=8192,
                func=embedding_func)
    )

    await rag.initialize_storages()

    await initialize_pipeline_status()

    return rag

def main():
    rag = asyncio.run(initialize_rag())

    # Insert text
    with open("academicRAG\exapmles\data\machine_learning.txt", encoding="utf-8") as f:
        rag.insert(f.read())

    # # Perform naive search
    # print(
    #     rag.aquery(
    #         "What are the top themes in this story?", param=QueryParam(mode="naive")
    #     )
    # )
    #
    # # Perform local search
    # print(
    #     rag.aquery(
    #         "What are the top themes in this story?", param=QueryParam(mode="subgraph")
    #     )
    # )
    #
    # # Perform global search
    # print(
    #     rag.aquery(
    #         "What are the top themes in this story?",
    #         param=QueryParam(mode="global"),
    #     )
    # )

    # Perform hybrid search
    print(
        rag.aquery(
            "What are the top themes in this story?",
            param=QueryParam(mode="hybrid"),
        )
    )


if __name__ == "__main__":
    asyncio.run(main())