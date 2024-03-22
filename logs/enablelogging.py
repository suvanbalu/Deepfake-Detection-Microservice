import os
import datetime
import logging

def setup_logging(log_dir,log_name=None):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    now = datetime.datetime.now()
    if log_name==None:
        log_name = f"log-{now.strftime('%Y-%m-%d_%H-%M')}.log"
    log_file_path = os.path.join(log_dir, log_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
        ]
    )

def close_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()