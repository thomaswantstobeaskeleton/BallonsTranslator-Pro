"""
Add "Open in BallonsTranslator" to the Windows Explorer context menu for:
  - .json files (e.g. imgtrans_chapter_002.json)  -> opens that project in the app
  - Folders (directories)                          -> opens that folder as project

Uses HKEY_CURRENT_USER so no administrator rights are required and .json files work.

Run this script once (e.g. "python scripts/install_windows_context_menu.py") from the
BallonsTranslator root. To remove the context menu, run with --uninstall.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.windows_context_menu import install as do_install, uninstall as do_uninstall, get_command


def install():
    ok, msg = do_install(REPO_ROOT)
    if ok:
        cmd = get_command(REPO_ROOT)
        print("Context menu installed:")
        print("  - Right-click a .json file -> Open in BallonsTranslator")
        print("  - Right-click a folder     -> Open in BallonsTranslator")
        print("Command:", cmd[:80] + ("..." if len(cmd) > 80 else ""))
    else:
        print(msg)
    return 0 if ok else 1


def uninstall():
    ok, msg = do_uninstall()
    print(msg)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Install or remove 'Open in BallonsTranslator' from Windows Explorer context menu.")
    ap.add_argument("--uninstall", action="store_true", help="Remove the context menu entries")
    args = ap.parse_args()
    if sys.platform != "win32":
        print("This script is for Windows only.")
        return 2
    return uninstall() if args.uninstall else install()


if __name__ == "__main__":
    try:
        code = main()
        if sys.platform == "win32":
            input("\nPress Enter to close this window...")
        sys.exit(code)
    except Exception as e:
        print("Error:", e)
        if sys.platform == "win32":
            input("\nPress Enter to close this window...")
        sys.exit(1)
