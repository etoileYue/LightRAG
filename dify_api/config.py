from lightrag.llm.openai import openai_complete_if_cache, openai_embed
import numpy as np
import os
import yaml

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

global_config = {
    "llm_model_func": llm_model_func,
    "embedding_func": {
        'embedding_dim': 1024, 
        'max_token_size': 8192, 
        'func': embedding_func,
        }, 
    "tiktoken_model_name": "gpt-4o-mini",
    "entity_summary_to_max_tokens": 500,
    "addon_params": {
        "language": "中文",
    },
    'vector_db_storage_cls_kwargs': {
        'cosine_better_than_threshold': 0.2,
    }, 
    "embedding_batch_num": 32,
    "working_dir": "./dickens"
}

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


# 获取当前脚本的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录
current_directory = os.path.dirname(current_file_path)
# 设定配置文件路径
config_path = f'{current_directory}/config.yml'
# 加载配置文件
CONFIG = load_yaml(config_path)

# knowledge_graph_inst=self.chunk_entity_relation_graph,
#                 entity_vdb=self.entities_vdb,
#                 relationships_vdb=self.relationships_vdb
"""
global_config = {
    'working_dir': './dickens', 
    'kv_storage': 'JsonKVStorage', 
    'vector_storage': 'MilvusVectorDBStorage', 
    'graph_storage': 'Neo4JStorage', 
    'doc_status_storage': 'JsonDocStatusStorage', 
    'log_level': None, 
    'log_file_path': None, 
    'entity_extract_max_gleaning': 1, 
    'entity_summary_to_max_tokens': 500, 
    'chunk_token_size': 1200, 
    'chunk_overlap_token_size': 100, 
    'tiktoken_model_name': 'gpt-4o-mini', 
    'chunking_func': "<function chunking_by_token_size at 0x7f500c89d360>", 
    'node_embedding_algorithm': 'node2vec', 
    'node2vec_params': {
        'dimensions': 1536, 
        'num_walks': 10, 
        'walk_length': 40, 
        'window_size': 2, 
        'iterations': 3, 
        'random_seed': 3
        }, 
    'embedding_func': {
        'embedding_dim': 1024, 
        'max_token_size': 8192, 
        'func': embedding_func,
        }, 
    'embedding_batch_num': 32, 
    'embedding_func_max_async': 16, 
    'embedding_cache_config': {
        'enabled': False, 
        'similarity_threshold': 0.95, 
        'use_llm_check': False
        }, 
    'llm_model_func': llm_model_func, 
    'llm_model_name': 'gpt-4o-mini', 
    'llm_model_max_token_size': 18000, 
    'llm_model_max_async': 4, 
    'llm_model_kwargs': {}, 
    'vector_db_storage_cls_kwargs': {}, 
    'namespace_prefix': '', 
    'enable_llm_cache': True, 
    'enable_llm_cache_for_entity_extract': True, 
    'max_parallel_insert': 20, 
    'addon_params': {'language': '中文'}, 
    'auto_manage_storages_states': True, 
    'convert_response_to_json_func': "<function convert_response_to_json at 0x7f500d428430>", 
    'cosine_better_than_threshold': 0.2, 
    '_storages_status': "<StoragesStatus.NOT_CREATED: 'not_created'>"
    }"
"""