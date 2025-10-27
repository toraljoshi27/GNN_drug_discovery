def get_logger(name: str):
    import logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
    return logging.getLogger(name)
