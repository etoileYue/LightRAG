import logging

def set_logger(name, log_level=logging.DEBUG, log_file="dify_api.log"):
    # 创建 Logger 对象
    logger = logging.getLogger(name)
    logger.setLevel(log_level)  # 设置最低级别

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # 控制台只输出 WARNING 及以上

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别

    # 定义日志格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到 Logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
