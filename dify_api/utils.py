from lightrag.operate import (_merge_nodes_then_upsert, _merge_edges_then_upsert, 
                              _find_most_related_edges_from_entities, _find_most_related_text_unit_from_entities,
                              _find_most_related_entities_from_relationships, _find_related_text_unit_from_relationships,
                              _get_node_data, _get_edge_data,
                              truncate_list_by_token_size)
from fastapi import Body
from lightrag.utils import bulk_upsert, compute_mdhash_id
from lightrag.base import QueryParam
from lightrag.kg.neo4j_impl import Neo4JStorage
from lightrag.kg.milvus_impl import MilvusVectorDBStorage
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.utils import EmbeddingFunc, always_get_an_event_loop
from dify_api.config import global_config, embedding_func, CONFIG
from dify_api.logger import set_logger
import asyncio
import time
import os
import json

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

config_log = CONFIG['log_config']
logger = set_logger(name='knowledge_logger', log_level=config_log['log_level'], log_file=config_log['log_file'])

embeddingFunc=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=embedding_func,
        )

from lightrag.kg.shared_storage import (
            initialize_share_data,
        )

# 初始化存储
initialize_share_data()

knowledge_graph_inst = Neo4JStorage(namespace="chunk_entity_relation", global_config=global_config, embedding_func=embeddingFunc)
entities_vdb = MilvusVectorDBStorage(namespace="entities", global_config=global_config, embedding_func=embeddingFunc, meta_fields={"entity_name", "source_id", "content"})
relationships_vdb = MilvusVectorDBStorage(namespace="relationships", global_config=global_config, embedding_func=embeddingFunc, meta_fields={"src_id", "tgt_id", "source_id", "content"})
chunks_vdb = MilvusVectorDBStorage(namespace="chunks", global_config=global_config, embedding_func=embeddingFunc)
text_chunks_db = JsonKVStorage(namespace="text_chunks", global_config=global_config, embedding_func=embeddingFunc)
node_datas_db = JsonKVStorage(namespace="node_datas", global_config=global_config, embedding_func=embeddingFunc)
edge_datas_db = JsonKVStorage(namespace="edge_datas", global_config=global_config, embedding_func=embeddingFunc)

# 事件循环
def _run_async_safely(async_func, action_name=""):
    """Safely execute an async function, avoiding event loop conflicts."""
    try:
        loop = always_get_an_event_loop()
        if loop.is_running():
            task = loop.create_task(async_func())
            task.add_done_callback(
                lambda t: logger.info(f"{action_name} completed!")
            )
        else:
            loop.run_until_complete(async_func())
    except RuntimeError:
        logger.warning(
            f"No running event loop, creating a new loop for {action_name}."
        )
        loop = asyncio.new_event_loop()
        loop.run_until_complete(async_func())
        loop.close()

async def kv_storage_initialize():
    tasks = []
    tasks.append(text_chunks_db.initialize())
    tasks.append(node_datas_db.initialize())
    tasks.append(edge_datas_db.initialize())
    await asyncio.gather(*tasks)

# 初始化存储
_run_async_safely(kv_storage_initialize, "Storage Initialized")

async def entites_from_kw(
        knowledge_id: str = Body(..., description="知识库标识符"),
        query: str = Body(..., description="关键词列表字符串"),
        retrieval_setting: dict = Body(..., description="参数", examples=[{"top_k": 10, "score_threshold": 0.5, "mode": "Hybrid"}])
):
    top_k = retrieval_setting.get("top_k", 10)
    score_threshold = retrieval_setting.get("score_threshold", 0.5)
    mode = retrieval_setting.get("mode", "Hybrid")

    global_config["vector_db_storage_cls_kwargs"]["score_threshold"] = score_threshold
    # query_param = QueryParam(top_k=top_k, mode=mode)

    ############ 
    logger.info(
        f"Query nodes: {query}, top_k: {top_k}, cosine: {entities_vdb.cosine_better_than_threshold}"
    )
    results = await entities_vdb.query(query, top_k=top_k)
    # print(results)
    if not len(results):
        return "", "", ""
    # get entity information
    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
        ),
        asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
        ),
    )

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d, "score": k["distance"]}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    upsert_data = {}
    for n in node_datas:
        key = compute_mdhash_id(query, prefix="key-")
        ent = compute_mdhash_id(n["entity_name"], prefix="ent-")
        if key not in upsert_data:
            upsert_data[key] = {"query": query}
        upsert_data[key][ent] = n
    # print(upsert_data)

    await node_datas_db.upsert(upsert_data)
    await node_datas_db.index_done_callback()

    records = []
    for n in node_datas:
        source_id = n["source_id"].split("<SEP>")
        source_chunks = await text_chunks_db.get_by_ids(source_id)
        description = ""

        for chunk in source_chunks:
            description += f"chunk order {chunk['chunk_order_index']} from doc {chunk['full_doc_id']}" + "\n"

        records.append({
            "metadata": {
                "path": str(n["source_id"]),
                "description": description,
            },
            "score": n["score"],
            "title": n["entity_name"],
            "content": json.dumps(n, indent=4, ensure_ascii=False)
        })
    return {"records": records}

async def edges_from_kw( 
        knowledge_id: str = Body(..., description="知识库标识符"),
        query: str = Body(..., description="关键词列表字符串"),
        retrieval_setting: dict = Body(..., description="参数", examples=[{"top_k": 10, "score_threshold": 0.5, "mode": "Hybrid"}])
):
    top_k = retrieval_setting.get("top_k", 10)
    score_threshold = retrieval_setting.get("score_threshold", 0.5)
    mode = retrieval_setting.get("mode", "Hybrid")

    global_config["vector_db_storage_cls_kwargs"]["score_threshold"] = score_threshold

    logger.info(
        f"Query edges: {query}, top_k: {top_k}, cosine: {score_threshold}"
    )
    results = await relationships_vdb.query(query, top_k=top_k)

    if not len(results):
        return "", "", ""

    edge_datas, edge_degree = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"])
                for r in results
            ]
        ),
    )

    edge_datas = [
        {
            "src_id": k["src_id"],
            "tgt_id": k["tgt_id"],
            "rank": d,
            "created_at": k.get("__created_at__", None),
            **v,
            "score": k["distance"]
        }
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )

    upsert_data = {}
    for e in edge_datas:
        key = compute_mdhash_id(query, prefix="key-")
        edg = compute_mdhash_id(e["src_id"]+e["tgt_id"], prefix="edg-")
        if key not in upsert_data:
            upsert_data[key] = {"query": query}
        upsert_data[key][edg] = e
    # print(upsert_data)

    await edge_datas_db.upsert(upsert_data)
    await edge_datas_db.index_done_callback()

    records = []
    for e in edge_datas:
        source_id = e["source_id"].split("<SEP>")
        source_chunks = await text_chunks_db.get_by_ids(source_id)
        description = ""

        for chunk in source_chunks:
            description += f"chunk order {chunk['chunk_order_index']} from doc {chunk['full_doc_id']}" + "\n"

        records.append(
            {
                "metadata": {
                    "path": str(source_id),
                    "description": description,
                },
                "score": e["rank"],
                "title": f"{e['src_id']} -> {e['tgt_id']}",
                "content": json.dumps(e, indent=4, ensure_ascii=False)
            }
        )

    return {"records": records}

async def edge_from_entities(
        knowledge_id: str = Body(..., description="知识库标识符"),
        query: str = Body(..., description="关键词列表字符串"),
        retrieval_setting: dict = Body(..., description="参数", examples=[{"top_k": 10, "score_threshold": 0.5, "mode": "Hybrid"}])
):
    top_k = retrieval_setting.get("top_k", 10)
    score_threshold = retrieval_setting.get("score_threshold", 0.5)
    mode = retrieval_setting.get("mode", "Hybrid")

    query_param = QueryParam(top_k=top_k, mode=mode)

    node_datas_ = await node_datas_db.get_by_id(compute_mdhash_id(content=query, prefix="key-"))
    node_datas = []
    for k, v in node_datas_.items():
        if k.startswith("ent-"):
            node_datas.append(v)

    ############ 
    use_relations = await _find_most_related_edges_from_entities(node_datas, query_param, knowledge_graph_inst)
    # print(use_relations)
    records = []

    for e in use_relations:
        source_id = e["source_id"].split("<SEP>")
        source_chunks = await text_chunks_db.get_by_ids(source_id)
        description = ""

        for chunk in source_chunks:
            description += f"chunk order {chunk['chunk_order_index']} from doc {chunk['full_doc_id']}" + "\n"

        records.append(
            {
                "metadata": {
                    "path": e["source_id"],
                    "description": description,
                },
                "score": e["rank"],
                "title": str(e["src_tgt"]),
                "content": json.dumps(e, indent=4, ensure_ascii=False)
            }
        )

    return {"records": records}

async def text_units_from_entities(
        knowledge_id: str = Body(..., description="知识库标识符"),
        query: str = Body(..., description="关键词列表字符串"),
        retrieval_setting: dict = Body(..., description="参数", examples=[{"top_k": 10, "score_threshold": 0.5, "mode": "Hybrid"}])
):
    top_k = retrieval_setting.get("top_k", 10)
    score_threshold = retrieval_setting.get("score_threshold", 0.5)
    mode = retrieval_setting.get("mode", "Hybrid")

    query_param = QueryParam(top_k=top_k, mode=mode)

    ############ 
    node_datas_ = await node_datas_db.get_by_id(compute_mdhash_id(content=query, prefix="key-"))
    node_datas = []
    for k, v in node_datas_.items():
        if k.startswith("ent-"):
            node_datas.append(v)

    use_text_units = await _find_most_related_text_unit_from_entities(node_datas, query_param, text_chunks_db, knowledge_graph_inst)

    records = [
        {
            "metadata": {
                "path": t["full_doc_id"],
                "description": f"chunk order index : {t['chunk_order_index']}"
            },
            "score": node_datas[i]["score"],
            "title": t["titles"],
            "content": t["content"]
        }
        for i, t in enumerate(use_text_units)
    ]

    return {"records": records}
    

async def node_from_relations(
        knowledge_id: str = Body(..., description="知识库标识符"),
        query: str = Body(..., description="关键词列表字符串"),
        retrieval_setting: dict = Body(..., description="参数", examples=[{"top_k": 10, "score_threshold": 0.5, "mode": "Hybrid"}])
):
    top_k = retrieval_setting.get("top_k", 10)
    score_threshold = retrieval_setting.get("score_threshold", 0.5)
    mode = retrieval_setting.get("mode", "Hybrid")

    query_param = QueryParam(top_k=top_k, mode=mode)
    # 这里先截断了？为什么？
    edge_datas_ = await edge_datas_db.get_by_id(compute_mdhash_id(content=query, prefix="key-"))
    edge_datas = []
    for k, v in edge_datas_.items():
        if k.startswith("edg-"):
            edge_datas.append(v)

    use_entities = await _find_most_related_entities_from_relationships(edge_datas, query_param, knowledge_graph_inst)
    # print(use_entities)

    records = []
    for n in use_entities:
        source_id = n["source_id"].split("<SEP>")
        source_chunks = await text_chunks_db.get_by_ids(source_id)
        description = ""

        for chunk in source_chunks:
            description += f"chunk order {chunk['chunk_order_index']} from doc {chunk['full_doc_id']}" + "\n"

        records.append({
            "metadata": {
                "path": n["source_id"],
                "description": description,
            },
            "score": n["rank"],
            "title": n["entity_name"],
            "content": json.dumps(n, indent=4, ensure_ascii=False)
        })
    return {"records": records}


async def text_units_from_relations(
        knowledge_id: str = Body(..., description="知识库标识符"),
        query: str = Body(..., description="关键词列表字符串"),
        retrieval_setting: dict = Body(..., description="参数", examples=[{"top_k": 10, "score_threshold": 0.5, "mode": "Hybrid"}])
):
    top_k = retrieval_setting.get("top_k", 10)
    score_threshold = retrieval_setting.get("score_threshold", 0.5)
    mode = retrieval_setting.get("mode", "Hybrid")

    query_param = QueryParam(top_k=top_k, mode=mode)
    # 这里先截断了？为什么？
    edge_datas_ = await edge_datas_db.get_by_id(compute_mdhash_id(content=query, prefix="key-"))
    edge_datas = []
    for k, v in edge_datas_.items():
        if k.startswith("edg-"):
            edge_datas.append(v)

    use_text_units = await _find_related_text_unit_from_relationships(edge_datas, query_param, text_chunks_db, knowledge_graph_inst)

    records = [
        {
            "metadata": {
                "path": t["full_doc_id"],
                "description": f"chunk order index : {t['chunk_order_index']}"
            },
            "score": edge_datas[i]["score"],
            "title": t["titles"],
            "content": t["content"]
        }
        for i, t in enumerate(use_text_units)
    ]

    return {"records": records}

# 接受所有可能的图节点和图关系，插入到知识图谱以及向量数据库中
async def upsert_entities_and_relations(
        maybe_nodes: list[dict],
        maybe_edges: list[dict],
        concurrency = 100
):
    all_entities_data = await bulk_upsert(
        [(k, v, knowledge_graph_inst, global_config) for k, v in maybe_nodes.items()],
        _merge_nodes_then_upsert,
        concurrency
    )

    all_relationships_data = await bulk_upsert(
        [(k[0], k[1], v, knowledge_graph_inst, global_config) for k, v in maybe_edges.items()],
        _merge_edges_then_upsert,
        concurrency
    )

    data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "keywords": dp["keywords"],
                "content": f"{dp['src_id']}\t{dp['tgt_id']}\n{dp['keywords']}\n{dp['description']}",
                "source_id": dp["source_id"],
                "metadata": {
                    "created_at": dp.get("metadata", {}).get("created_at", time.time())
                },
            }
            for dp in all_relationships_data
        }
    await relationships_vdb.upsert(data_for_vdb)

    data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "entity_name": dp["entity_name"],
                "entity_type": dp["entity_type"],
                "content": f"{dp['entity_name']}\n{dp['description']}",
                "source_id": dp["source_id"],
                "metadata": {
                    "created_at": dp.get("metadata", {}).get("created_at", time.time())
                },
            }
            for dp in all_entities_data
        }
    await entities_vdb.upsert(data_for_vdb)

async def upsert_chunks(
        chunks: dict
):
    text_chunks_db.upsert(chunks)
    chunks_vdb.upsert(chunks)

async def get_data_from_cache(db: JsonKVStorage, query: str, prefix: str):
    data_ = await db.get_by_id(compute_mdhash_id(query, prefix="key-"))
    data = []
    for k, v in data_.items():
        if k.startswith(prefix):
            data.append(v)

    return data