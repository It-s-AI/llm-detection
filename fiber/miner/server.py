import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from fiber.logging_utils import get_logger
from fiber.miner.core import configuration
from fiber.miner.endpoints.handshake import factory_router as handshake_factory_router

logger = get_logger(__name__)


def factory_app(debug: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = configuration.factory_config()
        metagraph = config.metagraph
        sync_thread = None
        if metagraph.substrate is not None:
            sync_thread = threading.Thread(target=metagraph.periodically_sync_nodes, daemon=True)
            sync_thread.start()

        yield

        logger.info("Shutting down...")

        config.encryption_keys_handler.close()
        metagraph.shutdown()
        if metagraph.substrate is not None and sync_thread is not None:
            sync_thread.join()

    app = FastAPI(lifespan=lifespan, debug=debug)

    handshake_router = handshake_factory_router()
    app.include_router(handshake_router)

    return app
