import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dify_api.config import CONFIG
from dify_api.logger import set_logger
from dify_api.utils import(
    edge_from_entities,
    node_from_relations,
    text_units_from_entities,
    text_units_from_relations,
    upsert_entities_and_relations,
    upsert_chunks,
    entites_from_kw,
    edges_from_kw,
)
# log
config_log = CONFIG['log_config']
logger = set_logger(name='knowledge_logger', log_level=config_log['log_level'], log_file=config_log['log_file'])


def create_app():
    app = FastAPI(
        title=CONFIG['app_name'],
        version=CONFIG['app_version'],
    )
    # 跨域
    cross_domain_config = CONFIG['cross_domain']
    if cross_domain_config['open_cross_domain']:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cross_domain_config['allow_origins'],
            allow_credentials=cross_domain_config['allow_credentials'],
            allow_methods=cross_domain_config['allow_methods'],
            allow_headers=cross_domain_config['allow_headers'],
        )
        
    # 路由
    mount_app_routes(app)
    return app


def mount_app_routes(app: FastAPI):
    app.post(
        "/retrieval/keywords/entities",
        summary="知识库检索，通过关键词找实体"
    )(entites_from_kw)

    app.post(
        "/retrieval/keywords/edges",
        summary="知识库检索，通过关键词找关系"
    )(edges_from_kw)

    app.post(
        "/retrieval/node/edge",
        summary="知识库检索，通过节点找边",
    )(edge_from_entities)

    app.post(
        "/retrieval/node/text_unit",
        summary="知识库检索，通过节点找相关文本",
    )(text_units_from_entities)

    app.post( 
        "/retrieval/edge/node",
        summary="知识库检索，通过边找节点",
    )(node_from_relations)

    app.post(
        "/retrieval/edge/text_unit",
        summary="知识库检索，通过边找相关文本",
    )(text_units_from_relations)

    app.post(
        "/upsert/entites_and_relations",
        summary="知识库插入实体和关系"
    )(upsert_entities_and_relations)

    app.post(
        "/upsert/chunks",
        summary="知识库插入文本块"
    )(upsert_chunks)

def run_api(app):
    api_config = CONFIG['api_server']
    if api_config.get("ssl_keyfile") and api_config.get("ssl_certfile"):
        uvicorn.run(app,
                    host=api_config['host'],
                    port=api_config['port'],
                    ssl_keyfile=api_config.get("ssl_keyfile"),
                    ssl_certfile=api_config.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=api_config['host'], port=api_config['port'])
        

if __name__ == '__main__':
    app = create_app()
    run_api(app)