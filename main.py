import logging

from app.brain.brain import think
from app.brain.location.server import start_location_server_if_enabled, stop_location_server
from app.brain.logging_setup import initialize_logging
from config.settings import APP_NAME, VERSION

# Importing app.brain.voice.server registers RFC-009's /voice/turn and /voice/client routes
# on the shared remote server (see app.brain.location.server's module docstring) before it
# starts listening below -- it has no lifecycle functions of its own to call.
import app.brain.voice.server  # noqa: E402,F401


def start_jarvis() -> None:
    initialize_logging()
    logger = logging.getLogger(__name__)
    logger.info("Application startup")
    start_location_server_if_enabled()

    print("=" * 50)
    print(APP_NAME)
    print(f"Version: {VERSION}")
    print("Status: ONLINE")
    print("=" * 50)

    try:
        while True:
            command = input("\nYou: ").strip()

            if not command:
                print("JARVIS: Please enter a command.")
                continue

            response = think(command)

            if response == "shutdown":
                print("JARVIS: Shutdown...")
                break

            print(f"JARVIS: {response}")
    except KeyboardInterrupt:
        print("\nJARVIS: Shutdown...")
    finally:
        stop_location_server()
        logging_shutdown()


def logging_shutdown() -> None:
    logger = logging.getLogger(__name__)
    logger.info("Application shutdown")


if __name__ == "__main__":
    start_jarvis()