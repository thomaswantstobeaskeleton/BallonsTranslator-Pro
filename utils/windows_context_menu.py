"""
Windows Explorer context menu: "Open in BallonsTranslator" for .json files and folders.
Uses HKEY_CURRENT_USER so no administrator rights are required and .json files work.
"""

from __future__ import annotations

import os
import sys

MENU_LABEL = "Open in BallonsTranslator"


def _get_repo_root() -> str:
    """BallonsTranslator repo root (parent of utils/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_command(repo_root: str | None = None) -> str:
    """Build the shell command for the context menu. Uses pythonw + launch.py."""
    if repo_root is None:
        repo_root = _get_repo_root()
    launch_py = os.path.join(repo_root, "launch.py")
    if not os.path.isfile(launch_py):
        raise FileNotFoundError(f"launch.py not found at {launch_py}")
    exe = sys.executable
    if exe.lower().endswith("python.exe"):
        pw = os.path.join(os.path.dirname(exe), "pythonw.exe")
        if os.path.isfile(pw):
            exe = pw
    return f'"{exe}" "{launch_py}" "--proj-dir" "%1"'


def install(repo_root: str | None = None) -> tuple[bool, str]:
    """
    Add "Open in BallonsTranslator" to the context menu for .json and folders.
    Uses HKEY_CURRENT_USER so no admin required. Returns (success, message).
    """
    if sys.platform != "win32":
        return False, "Windows only."
    try:
        import winreg
    except ImportError:
        return False, "winreg not available."
    if repo_root is None:
        repo_root = _get_repo_root()
    cmd = get_command(repo_root)
    hkcu = winreg.HKEY_CURRENT_USER
    try:
        # .json: HKCU\Software\Classes\.json\shell\OpenInBallonsTranslator
        # This makes the menu show for .json files without needing admin.
        with winreg.CreateKey(hkcu, r"Software\Classes\.json\shell\OpenInBallonsTranslator") as k:
            winreg.SetValue(k, None, winreg.REG_SZ, MENU_LABEL)
        with winreg.CreateKey(hkcu, r"Software\Classes\.json\shell\OpenInBallonsTranslator\command") as k:
            winreg.SetValue(k, None, winreg.REG_SZ, cmd)

        # Directory: HKCU\Software\Classes\Directory\shell\OpenInBallonsTranslator
        with winreg.CreateKey(hkcu, r"Software\Classes\Directory\shell\OpenInBallonsTranslator") as k:
            winreg.SetValue(k, None, winreg.REG_SZ, MENU_LABEL)
        with winreg.CreateKey(hkcu, r"Software\Classes\Directory\shell\OpenInBallonsTranslator\command") as k:
            winreg.SetValue(k, None, winreg.REG_SZ, cmd)

        return True, "Context menu installed for .json files and folders."
    except OSError as e:
        if getattr(e, "winerror", None) == 5:
            return False, "Access denied. Try running as Administrator."
        return False, str(e)


def uninstall() -> tuple[bool, str]:
    """Remove the context menu entries. Returns (success, message)."""
    if sys.platform != "win32":
        return False, "Windows only."
    try:
        import winreg
    except ImportError:
        return False, "winreg not available."
    hkcu = winreg.HKEY_CURRENT_USER
    keys_to_remove = [
        r"Software\Classes\.json\shell\OpenInBallonsTranslator\command",
        r"Software\Classes\.json\shell\OpenInBallonsTranslator",
        r"Software\Classes\Directory\shell\OpenInBallonsTranslator\command",
        r"Software\Classes\Directory\shell\OpenInBallonsTranslator",
    ]
    try:
        for key_path in keys_to_remove:
            try:
                winreg.DeleteKey(hkcu, key_path)
            except FileNotFoundError:
                pass
        return True, "Context menu removed."
    except OSError as e:
        if getattr(e, "winerror", None) == 5:
            return False, "Access denied. Try running as Administrator."
        return False, str(e)
