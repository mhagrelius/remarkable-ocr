"""Main GTK4 application."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gio

from remarkable_ocr.gui.window import MainWindow


class RemarkableOCRApp(Adw.Application):
    """Main application class."""

    def __init__(self) -> None:
        super().__init__(
            application_id="com.github.remarkable-ocr",
            flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
        )

    def do_activate(self) -> None:
        """Handle application activation."""
        win = self.props.active_window
        if not win:
            win = MainWindow(application=self)
        win.present()


def main() -> None:
    """Entry point for the GUI application."""
    app = RemarkableOCRApp()
    app.run(None)
