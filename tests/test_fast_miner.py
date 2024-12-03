from fiber.logging_utils import get_logger
from fiber.miner import server
from miner.endpoint import factory_router as subnet_router
from fiber.miner.middleware import configure_extra_logging_middleware
import logging
from miner.config import get_subnet_config
logger = get_logger(__name__)

app = server.factory_app(debug=True)

app.include_router(subnet_router())

if __name__ == "__main__":
    subnet_config = get_subnet_config()

    import uvicorn
    
    logger.info("Running miner")
    
    # you should update port that is available in your machine
    uvicorn.run(app, host="0.0.0.0", port=51685)

