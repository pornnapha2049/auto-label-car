import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="🚗 Car Segmentation & CVAT Auto-Annotator (YOLO + SAM)"
    )
    parser.add_argument(
        "--web", action="store_true", help="Launch Web GUI in browser (Flask) on port 5000"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port for Web GUI (default: 5000)"
    )
    args = parser.parse_args()

    if args.web:
        print(f"🚀 Starting Web GUI on http://127.0.0.1:{args.port} ...")
        from web_app import app
        app.run(host="0.0.0.0", port=args.port, debug=False)
    else:
        print("🖥️ Starting PyQt5 Desktop GUI...")
        from PyQt5.QtWidgets import QApplication
        from gui import MainWindow
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
