#!/usr/bin/env python3
"""
IonQ Qiskit provider setup.

This script only initializes the IonQ provider and selects a backend (e.g., ionq_qpu.aria-1).
Replace the placeholder API key with your real key before running.

Usage:
    # Edit API_KEY_PLACEHOLDER below or set IONQ_API_KEY env var, then run:
    .venv/bin/python ionq_setup.py

Notes:
- This does not submit any jobs; it just verifies initialization and prints backend info.
- Requires qiskit, qiskit-ionq, pylatexenc to be installed (added to requirements.txt).
"""
import os
import sys

try:
    from qiskit_ionq import IonQProvider
except Exception as e:
    raise ImportError(
        "qiskit-ionq is required. Install via: pip install qiskit qiskit-ionq pylatexenc"
    ) from e

# Optional: load a local .env if present (never commit .env)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # dotenv is optional; if not present, environment variables are still read
    pass

# Read API key strictly from environment
API_KEY = os.getenv("IONQ_API_KEY")

if not API_KEY:
    print(
        "IonQ setup: No API key found.\n"
        "Provide it via environment or a local .env file (which is gitignored).\n"
        "Examples:\n"
        "  # Shell env var\n"
        "  export IONQ_API_KEY=\"<YOUR_KEY>\"\n\n"
        "  # Or create a .env file next to this script with:\n"
        "  IONQ_API_KEY=<YOUR_KEY>\n\n"
        "Then run: .venv/bin/python ionq_setup.py"
    )
    sys.exit(0)

# Initialize provider
provider = IonQProvider(API_KEY)

# Choose a backend; example: Aria-1 QPU
backend_name = "ionq_qpu.aria-1"
backend = provider.get_backend(backend_name)

# Print some basic info to confirm initialization
print(f"IonQ provider initialized. Selected backend: {backend_name}")
try:
    conf = backend.configuration()
    status = backend.status()
    print("Backend configuration:", conf)
    print("Backend status:", status)
except Exception:
    # Older versions may not expose configuration/status in the same way
    print("Backend object:", backend)

print("Setup complete. You can now construct Qiskit circuits and submit to this backend.")
