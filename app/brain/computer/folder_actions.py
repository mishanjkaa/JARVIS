import ctypes
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KNOWN_FOLDERS = {
    "desktop": "Desktop",
    "downloads": "Downloads",
    "documents": "Documents",
    "pictures": "Pictures",
    "music": "Music",
    "videos": "Videos",
}

KNOWN_FOLDER_GUIDS = {
    "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
    "pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
}


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _safe_status(category: str) -> str:
    return category


def _resolve_known_folder_path(folder_name: str) -> Optional[Path]:
    if os.name != "nt":
        return None

    try:
        shell32 = ctypes.windll.shell32
        ole32 = ctypes.windll.ole32
        if not hasattr(shell32, "SHGetKnownFolderPath") or not hasattr(ole32, "CoTaskMemFree") or not hasattr(ole32, "CLSIDFromString"):
            return None

        shell32.SHGetKnownFolderPath.argtypes = [ctypes.POINTER(_GUID), ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p)]
        shell32.SHGetKnownFolderPath.restype = ctypes.c_long
        ole32.CLSIDFromString.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(_GUID)]
        ole32.CLSIDFromString.restype = ctypes.c_long
        ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        ole32.CoTaskMemFree.restype = None

        guid_value = KNOWN_FOLDER_GUIDS[folder_name]
        guid_struct = _GUID()
        clsid_result = ole32.CLSIDFromString(guid_value, ctypes.byref(guid_struct))
        if clsid_result != 0:
            return None

        path_ptr = ctypes.c_wchar_p()
        result = shell32.SHGetKnownFolderPath(ctypes.byref(guid_struct), 0, None, ctypes.byref(path_ptr))
        if result != 0 or not path_ptr.value:
            if path_ptr.value:
                ole32.CoTaskMemFree(path_ptr)
            return None

        resolved_path = Path(path_ptr.value)
        ole32.CoTaskMemFree(path_ptr)
        if resolved_path.exists() and resolved_path.is_dir():
            return resolved_path
        return None
    except Exception:
        return None


def _get_known_folder_status(folder_name: str) -> str:
    if os.name != "nt":
        return "unavailable"

    try:
        if _resolve_known_folder_path(folder_name) is not None:
            return "api_available"
        return "api_failed"
    except Exception:
        return "api_failed"


def _diagnose_known_folder_status(folder_name: str) -> str:
    return _get_known_folder_status(folder_name)


def open_known_folder(folder_name: str) -> str:
    normalized_name = folder_name.strip().lower()
    if normalized_name not in KNOWN_FOLDERS:
        return "I can only open known folders."

    try:
        logger.info("Known folder open requested: %s", normalized_name)
        if os.name != "nt":
            return "This feature is only available on Windows."

        target_folder = _resolve_known_folder_path(normalized_name)
        fallback_candidates = []
        if target_folder is not None:
            fallback_candidates.append(target_folder)

        home_path = Path.home()
        fallback_candidates.append(home_path / KNOWN_FOLDERS[normalized_name])

        if normalized_name == "desktop":
            fallback_candidates.extend(
                [
                    home_path / "OneDrive" / "Desktop",
                    home_path / "OneDrive - Personal" / "Desktop",
                    home_path / "Desktop",
                ]
            )

        for candidate in fallback_candidates:
            try:
                if candidate.exists() and candidate.is_dir():
                    os.startfile(candidate)
                    return f"Opened {KNOWN_FOLDERS[normalized_name]}."
            except Exception:
                continue

        return f"The {KNOWN_FOLDERS[normalized_name]} folder is not available."
    except Exception:
        logger.exception("Known folder open failed")
        return "I could not open that folder right now."
