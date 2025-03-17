import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status
import traceback

WORKING_DIR = "./dickens"

from dotenv import load_dotenv
load_dotenv()

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "qwen1.5-72b-chat-gptq-int4",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="EMPTY",
        base_url="http://10.2.5.6:27888/v1/",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="bge-m3",
        api_key="EMPTY",
        base_url="http://10.2.45.211:9997/v1/",
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
    # embedding_dimension = await get_embedding_dim()
    embedding_dimension = 1024
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
        graph_storage="Neo4JStorage"
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_rag()

        await rag.ainsert("test text")
        # with open("/workspace/LightRAG/resources/short.md", "r", encoding="utf-8") as f:
        #     await rag.ainsert(f.read(), "@data@")

        # Perform hybrid search
        # print(
        #     await rag.aquery(
        #         "光启公司从2019年至2021年的财务情况如何，可以告诉我具体数据吗？",
        #         param=QueryParam(mode="hybrid"),
        #     )
        # )

        '''
        # Perform naive search
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="naive")
            )
        )

        # Perform local search
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="global"),
            )
        )
        '''

    except Exception as e:
        print(f"An error occurred: {type(e).__name__}:{e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
