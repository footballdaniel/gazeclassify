import logging



def setup_logger(level: str) -> None:
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger().setLevel(level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    return logging.info