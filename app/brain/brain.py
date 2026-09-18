from app.brain.router import route_command


def think(command: str) -> str:
    return route_command(command)