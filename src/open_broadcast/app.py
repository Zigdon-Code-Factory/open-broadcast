"""Main application entry point."""

from __future__ import annotations

import argparse
import getpass
import logging
import logging.handlers
import platform
import signal
import sys

from open_broadcast import __version__, paths

log = logging.getLogger("open_broadcast")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    if sys.stderr is not None:  # None in windowed PyInstaller builds
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
    try:
        paths.log_dir().mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            paths.log_dir() / "open-broadcast.log", maxBytes=2_000_000, backupCount=2,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass
    for noisy in ("comtypes", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="open-broadcast", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--minimized", action="store_true", help="start hidden in the system tray")
    p.add_argument("--start", action="store_true", help="start processing immediately")
    p.add_argument("--no-update-check", action="store_true", help="disable update checks")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--self-test-output", metavar="FILE", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        from open_broadcast import selftest

        return selftest.run(args.self_test_output)
    setup_logging(args.verbose)
    log.info("Open Broadcast %s on %s %s (Python %s)", __version__, platform.system(),
             platform.machine(), platform.python_version())

    if platform.system() == "Windows":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("OpenBroadcast.App")
        except Exception:
            pass

    from PySide6.QtNetwork import QLocalServer, QLocalSocket
    from PySide6.QtWidgets import QApplication, QSystemTrayIcon

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Open Broadcast")
    app.setApplicationVersion(__version__)
    app.setQuitOnLastWindowClosed(False)

    # Single instance: if already running, ask it to show its window and exit.
    server_name = f"open-broadcast-{getpass.getuser()}"
    probe = QLocalSocket()
    probe.connectToServer(server_name)
    if probe.waitForConnected(300):
        probe.write(b"show")
        probe.waitForBytesWritten(300)
        log.info("Already running; activated existing instance")
        return 0
    QLocalServer.removeServer(server_name)
    server = QLocalServer()
    server.listen(server_name)

    from open_broadcast import config as config_mod
    from open_broadcast.pipeline.manager import PipelineManager
    from open_broadcast.ui.tray import SystemTrayApp, app_icon

    app.setWindowIcon(app_icon())
    cfg = config_mod.load()
    manager = PipelineManager(cfg)
    tray = SystemTrayApp(manager)
    server.newConnection.connect(lambda: (server.nextPendingConnection(), tray.show_panel()))

    if QSystemTrayIcon.isSystemTrayAvailable():
        tray.show()
    if not args.minimized or not QSystemTrayIcon.isSystemTrayAvailable():
        tray.show_panel()
    if args.start:
        tray.panel.toggle()
    if not args.no_update_check:
        tray.updates.start_background_checks()

    app.aboutToQuit.connect(manager.stop)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
