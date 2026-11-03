import os
import sys
from pathlib import Path

# Make the service package importable when pytest runs from ocr-service/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Settings maps the field `service_token` from SERVICE_TOKEN (no env prefix).
# Set it before any test module imports app.config.
os.environ.setdefault("SERVICE_TOKEN", "test-secret-token")