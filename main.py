import logging

from app.brain.brain import think
from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.location.server import start_location_server_if_enabled, stop_location_server
from app.brain.logging_setup import initialize_logging
from config.settings import APP_NAME, VERSION

# Importing app.brain.voice.server registers RFC-009's /voice/turn and /voice/client routes
# on the shared remote server (see app.brain.location.server's module docstring) before it
# starts listening below -- it has no lifecycle functions of its own to call.
import app.brain.voice.server  # noqa: E402,F401


def _maybe_start_voice_listening_on_launch() -> bool:
    """RFC-009 follow-up, owner's explicit request: JARVIS should be able to hear and act
    on commands as soon as it starts, with no 'ai on'/'voice on'/'voice talk' typed first.
    `ai_enabled`/`voice_enabled` now default to True (see config/settings.py) so those two
    are already on without any command; this covers the third one by entering the same
    continuous-listening loop 'voice listen' triggers, right at launch, whenever
    `voice_listen_on_startup` (also on by default) and `voice_enabled` both say to.

    Returns True if `start_jarvis` should go on to the normal typed prompt afterwards
    (every case except the owner asking to exit while this startup listening session was
    still running -- including voice being disabled, the mic being unavailable, or the
    listening session ending normally via a stop phrase/Ctrl+C, all of which just fall
    back to typing instead of ending the program)."""
    from app.brain.router import _voice_listen_loop

    config = get_effective_runtime_config()
    if not (config.get("voice_enabled", False) and config.get("voice_listen_on_startup", False)):
        return True
    print(
        "\nJARVIS: Starting with continuous voice listening (voice_listen_on_startup) -- "
        "just talk. Say a stop phrase (e.g. 'stop listening') or press Ctrl+C to switch to typing."
    )
    response = _voice_listen_loop()
    if response == "shutdown":
        print("JARVIS: Shutdown...")
        return False
    print(f"JARVIS: {response}")
    return True


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
        if _maybe_start_voice_listening_on_launch():
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