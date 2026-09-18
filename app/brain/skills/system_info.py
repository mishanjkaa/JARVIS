from datetime import datetime


def get_local_time() -> str:
    return datetime.now().strftime("%H:%M")


def get_local_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")
