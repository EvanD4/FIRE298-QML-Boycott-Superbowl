#!/usr/bin/env python3
"""
Run the AWS BB84 QKD simulation with noise and Golay error correction locally.
This script mirrors the steps in AWS_QKD_Simulation.ipynb but as a single runnable script.

Requirements:
- python >= 3.9
- numpy
- amazon-braket-sdk

Usage:
    python run_bb84.py

Notes:
- On first run, this script will download and unzip the AWS sample repo into
  ./sample-BB84-qkd-on-amazon-braket-main to access the `utils/` modules.
- If amazon-braket-sdk is not installed, you'll get an ImportError with guidance.
"""

import base64
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
AWS_SAMPLE_ZIP_URL = (
    "https://github.com/aws-samples/sample-BB84-qkd-on-amazon-braket/archive/refs/heads/main.zip"
)
REPO_DIR_NAME = "sample-BB84-qkd-on-amazon-braket-main"
THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR / REPO_DIR_NAME

BIT_FLIP_PROBABILITY = 0.15
NUMBER_OF_QUBITS = 12
ERROR_CORRECTION_CHUNK_SIZE = 12

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def ensure_repo_downloaded() -> Path:
    """Ensure the AWS sample repo is downloaded and unzipped locally.

    Returns the path to the repo directory that contains `utils/`.
    """
    if REPO_DIR.exists() and (REPO_DIR / "utils").is_dir():
        return REPO_DIR

    print("Downloading AWS sample BB84 repo …")
    with urllib.request.urlopen(AWS_SAMPLE_ZIP_URL) as resp:
        data = resp.read()

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # Extract into THIS_DIR
        zf.extractall(THIS_DIR)

    if not (REPO_DIR / "utils").is_dir():
        raise RuntimeError(
            f"Failed to prepare repo at {REPO_DIR}. 'utils/' not found after extraction."
        )

    print(f"Repo extracted to: {REPO_DIR}")
    return REPO_DIR


def import_dependencies(repo_dir: Path):
    """Import required runtime dependencies, providing clear guidance on errors."""
    # Make sure we can import the repo's `utils` package
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    try:
        from braket.circuits import noises  # noqa: F401
        from braket.devices import LocalSimulator  # noqa: F401
    except Exception as e:
        raise ImportError(
            "amazon-braket-sdk is required. Install it via:\n"
            "    pip install amazon-braket-sdk amazon-braket-default-simulator numpy\n\n"
            f"Original import error: {e}"
        ) from e

    # Defer returning modules to avoid re-import confusion
    from braket.circuits import noises  # type: ignore
    from braket.devices import LocalSimulator  # type: ignore
    from utils.bb84 import (  # type: ignore
        initialize_protocol,
        encode_qubits,
        measure_qubits,
        filter_qubits,
        array_to_string,
    )
    from utils.golay_code import GolayCode  # type: ignore
    from utils.secret_utils import convert_to_octets  # type: ignore

    return {
        "noises": noises,
        "LocalSimulator": LocalSimulator,
        "initialize_protocol": initialize_protocol,
        "encode_qubits": encode_qubits,
        "measure_qubits": measure_qubits,
        "filter_qubits": filter_qubits,
        "array_to_string": array_to_string,
        "GolayCode": GolayCode,
        "convert_to_octets": convert_to_octets,
    }


# --------------------------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------------------------

def run():
    repo_dir = ensure_repo_downloaded()
    deps = import_dependencies(repo_dir)

    noises = deps["noises"]
    LocalSimulator = deps["LocalSimulator"]
    initialize_protocol = deps["initialize_protocol"]
    encode_qubits = deps["encode_qubits"]
    measure_qubits = deps["measure_qubits"]
    filter_qubits = deps["filter_qubits"]
    array_to_string = deps["array_to_string"]
    GolayCode = deps["GolayCode"]
    convert_to_octets = deps["convert_to_octets"]

    # Raw keys
    alice_raw_key = np.array([])
    bob_raw_key = np.array([])

    # Generate key material until there are 12 bits of raw key
    while len(alice_raw_key) < ERROR_CORRECTION_CHUNK_SIZE:
        # For Alice, the important basis is encoding basis.
        encoding_basis_A, states_A, _ = initialize_protocol(NUMBER_OF_QUBITS)

        # Print the initial state of Alice
        sent_bits = array_to_string(states_A)

        # For Bob, the relevant basis is measurement basis.
        _, _, measurement_basis_B = initialize_protocol(NUMBER_OF_QUBITS)

        # Alice encodes the values of her qubits using according bases from `encoding_bases_A`.
        encoded_qubits_A = encode_qubits(NUMBER_OF_QUBITS, states_A, encoding_basis_A)

        # Transmission of encoded qubits to Bob - might add noise!
        noise = noises.BitFlip(probability=BIT_FLIP_PROBABILITY)
        encoded_qubits_A.apply_gate_noise(noise)

        # Bob performs measurement on the received qubits
        measured_circuit = measure_qubits(encoded_qubits_A, measurement_basis_B)
        device = LocalSimulator("braket_dm")
        result = device.run(measured_circuit, shots=1).result()
        measured_bits = list(result.measurements[0])

        # Key sifting
        alice_raw_key = np.concatenate(
            (alice_raw_key, filter_qubits(sent_bits, encoding_basis_A, measurement_basis_B))
        )
        bob_raw_key = np.concatenate(
            (bob_raw_key, filter_qubits(measured_bits, encoding_basis_A, measurement_basis_B))
        )

    # Truncate to 12
    alice_raw_key = alice_raw_key[:ERROR_CORRECTION_CHUNK_SIZE]
    bob_raw_key = bob_raw_key[:ERROR_CORRECTION_CHUNK_SIZE]
    print(f"Alice raw key (12 bits): {alice_raw_key.astype(int)}")
    print(f"Bob raw key (12 bits):    {bob_raw_key.astype(int)}")

    # Golay error correction
    error_correcting_code = GolayCode()
    generator_matrix = error_correcting_code.get_generator_matrix()
    parity_check = error_correcting_code.get_parity_check_matrix()
    b_matrix = error_correcting_code.get_b_matrix()

    encoded_key_A = np.matmul(generator_matrix, alice_raw_key) % 2
    print(f"Alice encoded key (24 bits, [12 info | 12 parity]): {encoded_key_A}")
    syndrome_A = np.matmul(encoded_key_A, parity_check) % 2
    print(f"Alice syndrome (should be all zeros):                 {syndrome_A}")
    parity_bits = encoded_key_A[ERROR_CORRECTION_CHUNK_SIZE:]
    print(f"Parity bits sent to Bob (12 bits):                   {parity_bits}")

    encoded_key_B = np.concatenate((bob_raw_key, parity_bits))
    print(f"Bob encoded key (24 bits):                            {encoded_key_B}")
    syndrome_B = np.matmul(encoded_key_B, parity_check) % 2
    print(f"Bob syndrome (12 bits):                               {syndrome_B}")
    syndrome_BB = np.matmul(syndrome_B, b_matrix) % 2
    print(f"Bob syndrome mapped via B matrix (12 bits):           {syndrome_BB}")

    if syndrome_BB.sum() < 4:
        correction_mask = np.concatenate((syndrome_BB, np.zeros(ERROR_CORRECTION_CHUNK_SIZE,)))
        print(
            f"Correction mask (24 bits, only first 12 applied):    {correction_mask}"
        )
    else:
        raise RuntimeError("Decoding failed - more than 3 errors")

    corrected_key = np.mod(bob_raw_key + correction_mask[:ERROR_CORRECTION_CHUNK_SIZE], 2).astype(int)
    print(f"Corrected Bob key (12 bits):                          {corrected_key}")
    print(
        f"Keys match (Alice vs Corrected Bob):                  {np.all(corrected_key == alice_raw_key)}"
    )

    # Validate correction
    if not np.all(corrected_key == alice_raw_key):
        raise AssertionError("Error correction failed: corrected_key != alice_raw_key")

    # Derive ASCII key
    ASCII_key = base64.b64encode(convert_to_octets(array_to_string(corrected_key))).decode("ascii")
    print(f"Final ASCII key (base64):                             {ASCII_key}")


if __name__ == "__main__":
    run()
