import os
import webbrowser


def open_notepad() -> str:
    try:
        os.startfile("notepad.exe")
        return "Opened Notepad."
    except OSError as exc:
        return f"Could not open Notepad: {exc}"


def open_calculator() -> str:
    try:
        os.startfile("calc.exe")
        return "Opened Calculator."
    except OSError as exc:
        return f"Could not open Calculator: {exc}"


def open_browser() -> str:
    try:
        if webbrowser.open("https://www.google.com"):
            return "Opened the default web browser."
        return "The browser could not be opened."
    except Exception as exc:
        return f"Could not open the browser: {exc}"
