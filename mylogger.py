#Logger setup

import logging

logger = logging.getLogger(__name__)

LOG_INTERVAL = 25

FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
LogLevel = logging.DEBUG
logging.basicConfig(format=FORMAT, level=LogLevel)

websockets_logger = logging.getLogger('websockets')
websockets_logger.setLevel(LogLevel)
websockets_logger.addHandler(logging.StreamHandler())