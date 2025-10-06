#!/usr/bin/env python3
"""
Run the BB84 QKD procedure on an IonQ QPU via Qiskit, with Golay error correction.

What this script does:
- Reads IonQ API key from environment or .env (gitignored).
- Randomly chooses Alice's bits and bases, and Bob's measurement bases.
- Builds a Qiskit circuit to encode Alice's qubits and measure in Bob's bases.
- Submits the circuit to the selected IonQ backend (default: ionq_qpu.aria-1) with 1 shot.
- Repeats rounds until the sifted key length reaches the target (default: 12 bits).
- Applies Golay(24,12) error correction to reconcile Bob's key with Alice's.

WARNING: Submitting to real hardware may take time due to queueing and may incur cost.

Usage:
    .venv/bin/python qiskit_bb84.py --backend ionq_qpu.aria-1 --qubits 12 --sifted 12

Note:
- Real hardware noise is physical and not the synthetic bit-flip used in the Braket demo.
- The --bit-flip option is ignored on hardware and kept only for parity with the notebook signature.
"""
import argparse
import base64
import os
import sys
from pathlib import Path

import numpy as np

# Optional: load .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    from qiskit import QuantumCircuit, transpile
except Exception as e:
    raise ImportError("qiskit is required. Install with: pip install qiskit") from e

try:
    from qiskit_ionq import IonQProvider
    from qiskit_ionq.exceptions import IonQAPIError
except Exception as e:
    raise ImportError(
        "qiskit-ionq is required. Install with: pip install qiskit-ionq"
    ) from e

# Reuse Golay and utils from the AWS sample repo where possible
THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR / "sample-BB84-qkd-on-amazon-braket-main"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

try:
    from utils.golay_code import GolayCode  # type: ignore
    from utils.secret_utils import convert_to_octets  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import Golay utilities from sample repo. Make sure you've run run_bb84.py"
        " once (it downloads the repo), or manually place the repo next to this script."
    ) from e


def initialize_protocol(num_qubits: int):
    """Randomly initialize BB84 choices for one round.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        encoding_basis_A: Alice's basis choices (0=Z, 1=X) for each qubit
        states_A:         Alice's raw bits (0/1) to encode
        measurement_basis_B: Bob's measurement bases (0=Z, 1=X) for each qubit
    """
    encoding_basis_A = np.random.randint(0, 2, size=num_qubits)
    states_A = np.random.randint(0, 2, size=num_qubits)
    measurement_basis_B = np.random.randint(0, 2, size=num_qubits)
    return encoding_basis_A, states_A, measurement_basis_B


def build_bb84_circuit(states_A, encoding_basis_A, measurement_basis_B):
    """Build a Qiskit circuit that implements one BB84 round.

    Steps per qubit i:
    - Alice prepares |0> or |1> depending on states_A[i].
    - If Alice's basis is X (encoding_basis_A[i] == 1), she applies H to rotate to X-basis.
    - Bob measures in his chosen basis: if measurement_basis_B[i] == 1, apply H then measure (X-basis measurement).
    - Measurement result is stored in the corresponding classical bit.
    """
    n = len(states_A)
    qc = QuantumCircuit(n, n)

    # Alice encodes: |0> or |1> then basis H if basis == 1 (X-basis)
    for i in range(n):
        if int(states_A[i]) == 1:
            qc.x(i)
        if int(encoding_basis_A[i]) == 1:
            qc.h(i)

    # Bob measurement: rotate basis by H for X-basis measurement
    for i in range(n):
        if int(measurement_basis_B[i]) == 1:
            qc.h(i)
        qc.measure(i, i)

    return qc


def filter_qubits(bits, basis_A, basis_B):
    """Sift the key by keeping only positions where bases matched.

    Parameters
    ----------
    bits : sequence[int]
        Bit values (Alice's sent bits or Bob's measured bits) for the round.
    basis_A, basis_B : np.ndarray
        Basis choices (0=Z, 1=X) used by Alice and Bob.
    """
    keep = (basis_A == basis_B)
    return np.array([bits[i] for i in range(len(bits)) if keep[i]])


def array_to_string(arr):
    """Convert an array of 0/1 values to a compact string like '0101'."""
    return "".join(str(int(x)) for x in arr)


def run(backend_name: str, num_qubits: int, target_sifted_bits: int):
    """Execute BB84 on the specified IonQ backend until target sifted bits are collected.

    This function performs:
    1) Random selection of Alice/Bob bases and Alice's bits.
    2) Circuit construction and submission to the backend with one shot.
    3) Sifting to keep outcomes where bases match.
    4) Repeats the above until target_sifted_bits are accumulated.
    5) Applies Golay(24,12) error correction to reconcile the keys.
    """
    # 1) Random protocol choices for the first round
    encoding_basis_A, states_A, measurement_basis_B = initialize_protocol(num_qubits)

    # 2) Build and submit circuit for one shot
    qc = build_bb84_circuit(states_A, encoding_basis_A, measurement_basis_B)

    # IonQ provider from env
    api_key = os.getenv("IONQ_API_KEY")
    if not api_key:
        print(
            "No IONQ_API_KEY found. Set it in your environment or .env and re-run."
        )
        sys.exit(1)

    provider = IonQProvider(api_key)
    backend = provider.get_backend(backend_name)

    # IonQ may define a restricted gate set; transpile maps to supported ops
    tqc = transpile(qc, backend)
    try:
        job = backend.run(tqc, shots=1)
        print(f"Submitted job to {backend_name}. Job ID: {job.job_id()}")
        result = job.result()
    except IonQAPIError as ie:
        # Provide clear guidance on insufficient scope / permissions
        msg = str(ie)
        if "Insufficient scope" in msg or "403" in msg:
            print(
                "ERROR: IonQ API returned 'Insufficient scope' (403).\n"
                "Your API key or account likely doesn't have permission to run on this QPU.\n\n"
                "How to fix:\n"
                "  1) Log into the IonQ console and ensure your API key has permissions to run on QPUs.\n"
                "  2) Verify your workspace/subscription includes access to the backend '"
                + backend_name
                + "'.\n"
                "  3) Regenerate an API key with the required scopes, then update IONQ_API_KEY.\n\n"
                "Quick test: try the simulator backend first:\n"
                "  .venv/bin/python qiskit_bb84.py --backend ionq_simulator --qubits 12 --sifted 12\n"
            )
        else:
            print(f"IonQ API Error: {ie}")
        sys.exit(2)

    # Parse one-shot measurement into list of bits (reversed due to Qiskit bit order)
    counts = result.get_counts()
    # Expect one outcome with count 1
    bitstring = next(iter(counts.keys()))
    measured_bits = [int(b) for b in bitstring[::-1]]  # reverse to align q0->c0

    # 3) Sifting: keep only positions where bases matched
    alice_raw_key = filter_qubits(states_A, encoding_basis_A, measurement_basis_B)
    bob_raw_key = filter_qubits(measured_bits, encoding_basis_A, measurement_basis_B)

    # 4) Accumulate more rounds until we have the target sifted key length
    #    (Each round is 1 shot of num_qubits; expected ~50% match rate.)
    while len(alice_raw_key) < target_sifted_bits:
        encoding_basis_A, states_A, measurement_basis_B = initialize_protocol(num_qubits)
        qc = build_bb84_circuit(states_A, encoding_basis_A, measurement_basis_B)
        tqc = transpile(qc, backend)
        try:
            job = backend.run(tqc, shots=1)
            print(f"Submitted job to {backend_name}. Job ID: {job.job_id()}")
            result = job.result()
        except IonQAPIError as ie:
            msg = str(ie)
            if "Insufficient scope" in msg or "403" in msg:
                print(
                    "ERROR: IonQ API returned 'Insufficient scope' (403) during additional round.\n"
                    "Consider switching to the simulator backend until access is granted:\n"
                    "  .venv/bin/python qiskit_bb84.py --backend ionq_simulator --qubits 12 --sifted 12\n"
                )
            else:
                print(f"IonQ API Error: {ie}")
            sys.exit(2)
        counts = result.get_counts()
        bitstring = next(iter(counts.keys()))
        measured_bits = [int(b) for b in bitstring[::-1]]
        alice_raw_key = np.concatenate(
            (alice_raw_key, filter_qubits(states_A, encoding_basis_A, measurement_basis_B))
        )
        bob_raw_key = np.concatenate(
            (bob_raw_key, filter_qubits(measured_bits, encoding_basis_A, measurement_basis_B))
        )

    alice_raw_key = alice_raw_key[:target_sifted_bits]
    bob_raw_key = bob_raw_key[:target_sifted_bits]

    print(f"Alice raw key (12): {alice_raw_key.astype(int)}")
    print(f"Bob raw key (12):    {bob_raw_key.astype(int)}")

    # 5) Golay error correction (reuse AWS utils from the sample repo)
    error_correcting_code = GolayCode()
    G = error_correcting_code.get_generator_matrix()
    H = error_correcting_code.get_parity_check_matrix()
    B = error_correcting_code.get_b_matrix()

    encoded_key_A = np.matmul(G, alice_raw_key) % 2
    print(f"Alice encoded key (24): {encoded_key_A}")
    syndrome_A = np.matmul(encoded_key_A, H) % 2
    print(f"Alice syndrome (should be zeros): {syndrome_A}")
    parity_bits = encoded_key_A[target_sifted_bits:]
    print(f"Parity bits sent to Bob (12): {parity_bits}")

    encoded_key_B = np.concatenate((bob_raw_key, parity_bits))
    print(f"Bob encoded key (24): {encoded_key_B}")
    syndrome_B = np.matmul(encoded_key_B, H) % 2
    print(f"Bob syndrome (12): {syndrome_B}")
    syndrome_BB = np.matmul(syndrome_B, B) % 2
    print(f"Bob syndrome mapped via B (12): {syndrome_BB}")

    if syndrome_BB.sum() < 4:
        correction_mask = np.concatenate((syndrome_BB, np.zeros(target_sifted_bits,)))
        print(f"Correction mask (24, first 12 applied): {correction_mask}")
    else:
        raise RuntimeError("Decoding failed - more than 3 errors")

    corrected_key = np.mod(bob_raw_key + correction_mask[:target_sifted_bits], 2).astype(int)
    print(f"Corrected Bob key (12): {corrected_key}")
    print(f"Keys match: {np.all(corrected_key == alice_raw_key)}")

    # Derive ASCII key
    ASCII_key = base64.b64encode(convert_to_octets(array_to_string(corrected_key))).decode("ascii")
    print(f"Final ASCII key (base64): {ASCII_key}")


def main():
    parser = argparse.ArgumentParser(description="Run BB84 on IonQ via Qiskit")
    parser.add_argument("--backend", default="ionq_qpu.aria-1", help="IonQ backend name")
    parser.add_argument("--qubits", type=int, default=12, help="Number of qubits per round")
    parser.add_argument("--sifted", type=int, default=12, help="Target sifted key length")
    parser.add_argument("--bit-flip", type=float, default=0.15, help="Ignored on hardware; kept for parity")
    args = parser.parse_args()

    run(args.backend, args.qubits, args.sifted)


if __name__ == "__main__":
    main()
