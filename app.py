"""
app.py – TB Chest X-ray Screening Assistant
Entry point for the Flask development server.

⚠️  DISCLAIMER
This application is a research/demo screening aid only.
It is NOT a medical device and must NOT be used for diagnosis,
treatment, or ruling out tuberculosis.
Always consult a qualified healthcare professional.
"""

import os
from flask import Flask, send_from_directory
from flask_cors import CORS

from backend.routes import api


def create_app() -> Flask:
    """Application factory."""
    # Resolve the frontend directory relative to this file
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")

    app = Flask(
        __name__,
        static_folder=frontend_dir,
        static_url_path="",
    )

    # Allow the browser to call the API from the frontend (supports local dev and cloud deployments)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register API blueprint
    app.register_blueprint(api)

    # Serve the SPA index for every non-API route
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        target = os.path.join(frontend_dir, path)
        if path and os.path.exists(target):
            return send_from_directory(frontend_dir, path)
        return send_from_directory(frontend_dir, "index.html")

    return app


# Expose app at module level for WSGI servers like Gunicorn (e.g. gunicorn app:app)
app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    is_cloud = bool(os.environ.get("PORT"))

    print("\n" + "=" * 60)
    print("  TB Chest X-ray Screening Assistant  -  Stage 1 MVP")
    print("=" * 60)
    print("  [!] DISCLAIMER: Research/demo screening aid only.")
    print("  Not a medical device. Not for diagnosis or treatment.")
    print("  Consult a qualified healthcare professional.")
    print("=" * 60)
    print(f"  Running at:  http://{host}:{port}")
    print(f"  API health:  http://{host}:{port}/api/health")
    print("=" * 60 + "\n")
    app.run(host=host, port=port, debug=not is_cloud)
