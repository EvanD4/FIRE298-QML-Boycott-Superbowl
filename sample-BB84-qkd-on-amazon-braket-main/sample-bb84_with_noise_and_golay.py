import base64
import argparse
import numpy as np

from braket.circuits import noises
from braket.devices import LocalSimulator

from utils.bb84 import initialize_protocol, encode_qubits, measure_qubits, filter_qubits, array_to_string
from utils.golay_code import GolayCode
from utils.secret_utils import convert_to_octets

# ---

# python sample-bb84_with_noise_and_golay.py - TO RUN

parser = argparse.ArgumentParser(description="BB84 with noise + Golay [24,12] error correction")
parser.add_argument("--num-qubits", type=int, default=24, help="Number of qubits transmitted per BB84 round (default: 12)")
parser.add_argument("--bit-flip", type=float, default=0.25, dest="bit_flip", help="Bit flip probability in the channel [0,1] (default: 0.3)")
parser.add_argument("--blocks", type=int, default=2, help="Number of 12-bit blocks to reconcile with Golay [24,12] (default: 1)")
parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
args, _ = parser.parse_known_args()

if args.seed is not None:
    np.random.seed(args.seed)

BIT_FLIP_PROBABILITY = args.bit_flip          # channel noise
NUMBER_OF_QUBITS = max(1, int(args.num_qubits))  # per-round qubits used in BB84
ERROR_CORRECTION_CHUNK_SIZE = 12                 # Golay [24,12] works on 12-bit messages
NUM_BLOCKS = max(1, int(args.blocks))
TARGET_KEY_BITS = ERROR_CORRECTION_CHUNK_SIZE * NUM_BLOCKS

alice_raw_key = np.array([])
bob_raw_key = np.array([])

# ---
def header(title: str):
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}")

def as_int(a):
    try:
        return np.asarray(a).astype(int)
    except Exception:
        return a

header("CONFIGURATION")
print(f"NUMBER_OF_QUBITS (per round): {NUMBER_OF_QUBITS}")
print(f"BIT_FLIP_PROBABILITY: {BIT_FLIP_PROBABILITY}")
print(f"ERROR_CORRECTION_CHUNK_SIZE (Golay message bits): {ERROR_CORRECTION_CHUNK_SIZE}")
print(f"NUM_BLOCKS: {NUM_BLOCKS}  => TARGET_KEY_BITS: {TARGET_KEY_BITS}")
if args.seed is not None:
    print(f"SEED: {args.seed}")

# ---

# Generate key material until TARGET_KEY_BITS of raw key are accumulated

iteration = 0
while len(alice_raw_key) < TARGET_KEY_BITS:
    iteration += 1
    header(f"BB84 ROUND {iteration}")

    # For Alice, the important basis is encoding basis.
    encoding_basis_A, states_A, _ = initialize_protocol(NUMBER_OF_QUBITS)
    
    # Print the initial state of Alice
    sent_bits = array_to_string(states_A)
    print(f"Alice - encoding_basis: {as_int(encoding_basis_A)}")
    print(f"Alice - states (sent_bits): {as_int(sent_bits)}")

    # For Bob, the relevant basis is measurement basis.
    _, _, measurement_basis_B = initialize_protocol(NUMBER_OF_QUBITS)
    print(f"Bob - measurement_basis: {as_int(measurement_basis_B)}")

    # Alice encodes the values of her qubits using according bases from `encoding_bases_A`.  
    # This is stored as a Qiskit quantum circuit.  
    encoded_qubits_A = encode_qubits(NUMBER_OF_QUBITS, states_A, encoding_basis_A)

    # Transmission of encoded qubits to Bob - emulate bit-flip noise by sampled X gates
    # Using a state-vector simulator avoids density-matrix memory blow-up at higher qubit counts.
    flip_mask = (np.random.rand(NUMBER_OF_QUBITS) < BIT_FLIP_PROBABILITY)
    print(f"Channel - sampled_flip_mask: {as_int(flip_mask)}")
    for qi, do_flip in enumerate(flip_mask):
        if do_flip:
            encoded_qubits_A.x(qi)

    # Bob performs measurement on the received qubits
    measured_circuit = measure_qubits(encoded_qubits_A, measurement_basis_B)
    device = LocalSimulator("braket_sv")
    result = device.run(measured_circuit, shots=1).result()
    measured_bits = list(result.measurements[0])
    print(f"Bob - measured_bits: {as_int(measured_bits)}")

    # After Bob has measured the qubits, he sends his measurement bases to Alice. 
    # She responds to Bob by sending him her encoding bases. Now both parties know both the encoding basis and measurement basis for each qubit.   
    # In the key sifting phase, both sides keep only the qubits for which encoding basis and measurement basis were the same.      
    alice_keep = filter_qubits(sent_bits, encoding_basis_A, measurement_basis_B)
    bob_keep = filter_qubits(measured_bits, encoding_basis_A, measurement_basis_B)
    print(f"Sifting - kept_by_Alice: {as_int(alice_keep)}")
    print(f"Sifting - kept_by_Bob:   {as_int(bob_keep)}")
    alice_raw_key = np.concatenate((alice_raw_key, alice_keep))
    bob_raw_key = np.concatenate((bob_raw_key, bob_keep))
    print(f"Accumulated - alice_raw_key_len: {len(alice_raw_key)} | bob_raw_key_len: {len(bob_raw_key)}")


# ---

header("RAW KEYS (TRUNCATED TO TARGET)")
alice_raw_key = alice_raw_key[:TARGET_KEY_BITS]
bob_raw_key = bob_raw_key[:TARGET_KEY_BITS]
print(f"Alice - raw_key[{TARGET_KEY_BITS}]: {as_int(alice_raw_key)}")
print(f"Bob   - raw_key[{TARGET_KEY_BITS}]: {as_int(bob_raw_key)}")

# ---

error_correcting_code = GolayCode()
generator_matrix = error_correcting_code.get_generator_matrix()
parity_check = error_correcting_code.get_parity_check_matrix()
b_matrix = error_correcting_code.get_b_matrix()

overall_matches = []
final_ascii_keys = []

for block in range(NUM_BLOCKS):
    header(f"GOLAY [24,12] BLOCK {block+1}/{NUM_BLOCKS}")
    a_block = alice_raw_key[block*12:(block+1)*12]
    b_block = bob_raw_key[block*12:(block+1)*12]
    print(f"Alice - block[{block}] key (12): {as_int(a_block)}")
    print(f"Bob   - block[{block}] key (12): {as_int(b_block)}")

    encoded_key_A = np.matmul(generator_matrix, a_block) % 2
    print(f"Alice - encoded_key (24): {as_int(encoded_key_A)}")
    syndrome_A = np.matmul(encoded_key_A, parity_check) % 2
    print(f"Alice - syndrome (should be all zeros): {as_int(syndrome_A)}")
    parity_bits = encoded_key_A[12:]
    print(f"Alice -> Bob - parity_bits (12): {as_int(parity_bits)}")

    header("BOB RECONSTRUCTION & SYNDROME")
    encoded_key_B = np.concatenate((b_block, parity_bits))
    print(f"Bob - concatenated (key || parity): {as_int(encoded_key_B)}")
    syndrome_B = np.matmul(encoded_key_B, parity_check) % 2
    print(f"Bob - syndrome: {as_int(syndrome_B)}")
    syndrome_bb = np.matmul(syndrome_B, b_matrix) % 2
    print(f"Bob - syndrome * B (error locator): {as_int(syndrome_bb)}")

    # Pre-initialize a zero correction mask and a failure flag
    correction_mask = np.zeros(24, dtype=int)
    decode_failed = False

    if syndrome_bb.sum() < 4:
        correction_mask = np.concatenate((syndrome_bb, np.zeros(12, dtype=int)))
        print(f"Bob - correction_mask (24): {as_int(correction_mask)}")
    else:
        decode_failed = True
        print("[ERROR] Decoding failed - more than 3 errors")
        print("[WARNING] Using zero correction mask; keys will likely not match.")

    # Final corrected key for this block
    corrected_key = np.mod(b_block + correction_mask[:12], 2).astype(int)
    header("FINAL KEY & VERIFICATION (BLOCK)")
    print(f"Bob - corrected_key: {as_int(corrected_key)}")
    keys_match = np.array_equal(corrected_key, a_block.astype(int))
    if decode_failed:
        print(f"Keys match (Bob == Alice): {keys_match} (decode_failed=True)")
    else:
        print(f"Keys match (Bob == Alice): {keys_match}")
    overall_matches.append(keys_match)

    ascii_key = base64.b64encode(convert_to_octets(array_to_string(corrected_key))).decode('ascii')
    print(f"Final Base64 key (ASCII, block {block+1}): {ascii_key}")
    final_ascii_keys.append(ascii_key)

# Summary across blocks
header("SUMMARY ACROSS BLOCKS")
print(f"Blocks: {NUM_BLOCKS}")
print(f"Per-block match results: {overall_matches}")
print(f"All blocks matched: {all(overall_matches)}")
print(f"Base64 keys per block: {final_ascii_keys}")