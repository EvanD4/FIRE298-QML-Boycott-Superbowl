#!/usr/bin/env python3
"""
Quick optimized run for immediate results - balanced speed vs quality.
This version runs faster while still showing significant improvements.
"""

import numpy as np
import random
from typing import Dict, Optional
import matplotlib.pyplot as plt

# Set seeds
random.seed(42)
np.random.seed(42)

print("🚀 OPTIMIZED QCL ATTACK - NO PCCM BIAS")
print("=" * 70)
print("⚡ More attempts/iterations for better results")
print("🎯 Target: Natural F_AE maximization in ~20-30 minutes")
print("📊 PCCM penalties REMOVED - unbiased optimization")
print("=" * 70)

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, Operator
    print("✅ Qiskit loaded")
except ImportError as e:
    print(f"❌ Error: {e}")
    exit(1)

backend = AerSimulator()

class AdamOptimizer:
    def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, gradients):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

def hardware_efficient_ansatz(n_qubits: int, n_layers: int, params: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    param_index = 0
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            if param_index < len(params):
                qc.rx(params[param_index], qubit)
                param_index += 1
            if param_index < len(params):
                qc.ry(params[param_index], qubit)
                param_index += 1
            if param_index < len(params):
                qc.rz(params[param_index], qubit)
                param_index += 1
        if n_qubits > 1:
            for qubit in range(n_qubits):
                qc.cx(qubit, (qubit + 1) % n_qubits)
    return qc

def compute_fidelity_density_matrix(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    rho_a = np.array(rho_a, dtype=complex)
    rho_b = np.array(rho_b, dtype=complex)
    eps = 1e-12
    rho_a = rho_a + eps * np.eye(len(rho_a))
    rho_b = rho_b + eps * np.eye(len(rho_b))
    eigvals_a, eigvecs_a = np.linalg.eigh(rho_a)
    eigvals_a = np.maximum(eigvals_a, 0)
    sqrt_rho_a = eigvecs_a @ np.diag(np.sqrt(eigvals_a)) @ eigvecs_a.conj().T
    product = sqrt_rho_a @ rho_b @ sqrt_rho_a
    eigenvalues = np.linalg.eigvalsh(product)
    eigenvalues = np.maximum(eigenvalues, 0)
    fidelity = np.sum(np.sqrt(eigenvalues))**2
    return float(np.clip(np.real(fidelity), 0, 1))

def evaluate_attack_fidelities(u_params: np.ndarray, v_params: np.ndarray, 
                               n_qubits_u: int, n_layers_u: int,
                               n_qubits_v: int, n_layers_v: int) -> Dict[str, float]:
    test_states = [(0, 0), (1, 0), (0, 1), (1, 1)]
    n_params_v_single = n_qubits_v * n_layers_v * 3
    v_params_z = v_params[:n_params_v_single]
    v_params_x = v_params[n_params_v_single:2*n_params_v_single]
    f_ab_list = []
    f_ae_list = []

    for bit, basis in test_states:
        qc_alice = QuantumCircuit(n_qubits_u)
        if bit == 1:
            qc_alice.x(0)
        if basis == 1:
            qc_alice.h(0)
        qc_u = hardware_efficient_ansatz(n_qubits_u, n_layers_u, u_params)
        qc_combined = qc_alice.compose(qc_u)
        statevec = Statevector.from_instruction(qc_combined)
        density_matrix_full = DensityMatrix(statevec)
        eve_qubits = list(range(1, n_qubits_u))
        dm_bob = partial_trace(density_matrix_full, eve_qubits)
        
        if bit == 0 and basis == 0:
            target = np.array([[1, 0], [0, 0]], dtype=complex)
        elif bit == 1 and basis == 0:
            target = np.array([[0, 0], [0, 1]], dtype=complex)
        elif bit == 0 and basis == 1:
            target = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        else:
            target = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex)

        f_ab = compute_fidelity_density_matrix(target, dm_bob.data)
        f_ab_list.append(f_ab)
        dm_eve = partial_trace(density_matrix_full, [0])
        current_v_params = v_params_x if basis == 1 else v_params_z
        qc_v = hardware_efficient_ansatz(n_qubits_v, n_layers_v, current_v_params)
        U_v = Operator(qc_v)
        dm_eve_measured = U_v @ dm_eve.data @ U_v.adjoint()
        f_ae = compute_fidelity_density_matrix(target, dm_eve_measured)
        f_ae_list.append(f_ae)

    return {'F_AB': np.mean(f_ab_list), 'F_AE': np.mean(f_ae_list)}

def enhanced_loss(params, target_f_ab, alpha, n_qubits_u, n_layers_u, n_qubits_v, n_layers_v):
    """Unbiased loss function without PCCM curve bias."""
    n_params_u = n_qubits_u * n_layers_u * 3
    n_params_v = n_qubits_v * n_layers_v * 3
    u_params = params[:n_params_u]
    v_params = params[n_params_u:n_params_u + 2*n_params_v]
    fidelities = evaluate_attack_fidelities(u_params, v_params, n_qubits_u, n_layers_u, n_qubits_v, n_layers_v)
    f_ab = fidelities['F_AB']
    f_ae = fidelities['F_AE']
    base_loss = alpha * (f_ab - target_f_ab) ** 2 - f_ae
    
    # Unbiased F_AB accuracy penalty only
    f_ab_penalty = 0.0
    f_ab_error = abs(f_ab - target_f_ab)
    if f_ab_error > 0.03:
        f_ab_penalty += 50.0 * (f_ab_error - 0.03) ** 2
    
    return base_loss + f_ab_penalty + 0.001 * np.sum(params ** 2)

def compute_gradient(params, target_f_ab, alpha, n_qubits_u, n_layers_u, n_qubits_v, n_layers_v, shift=np.pi/2):
    gradients = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += shift
        loss_plus = enhanced_loss(params_plus, target_f_ab, alpha, n_qubits_u, n_layers_u, n_qubits_v, n_layers_v)
        params_minus = params.copy()
        params_minus[i] -= shift
        loss_minus = enhanced_loss(params_minus, target_f_ab, alpha, n_qubits_u, n_layers_u, n_qubits_v, n_layers_v)
        gradients[i] = (loss_plus - loss_minus) / 2
    return gradients

def optimize_quick(target_f_ab, initial_params=None):
    # Optimized architecture for IONQ
    n_qubits_u, n_layers_u = 2, 2
    n_qubits_v, n_layers_v = 1, 2
    
    n_params_u = n_qubits_u * n_layers_u * 3
    n_params_v = n_qubits_v * n_layers_v * 3
    total_params = n_params_u + n_params_v * 2

    print(f"  🎯 F_AB={target_f_ab:.3f} ({total_params} params)")

    best_result = None
    best_distance = float('inf')
    n_attempts = 20  # More attempts for better results

    for attempt in range(n_attempts):
        if initial_params is not None and attempt == 0:
            params = initial_params.copy() if len(initial_params) == total_params else np.random.uniform(-np.pi, np.pi, total_params)
            params += np.random.normal(0, 0.05, size=params.shape)
            params = np.clip(params, -np.pi, np.pi)
        else:
            if best_result and attempt > 5:
                params = best_result['best_params'].copy()
                params += np.random.normal(0, 0.2, size=params.shape)
                params = np.clip(params, -np.pi, np.pi)
            else:
                params = np.random.uniform(-np.pi, np.pi, total_params)
        
        optimizer = AdamOptimizer(learning_rate=0.25)
        best_loss = float('inf')
        best_params = params.copy()
        patience = 0
        max_iter = 300  # More iterations for better convergence

        for iteration in range(max_iter):
            progress = iteration / max_iter
            current_alpha = 25.0 * np.exp(-4 * progress) + 2.0
            current_loss = enhanced_loss(params, target_f_ab, current_alpha, n_qubits_u, n_layers_u, n_qubits_v, n_layers_v)
            gradients = compute_gradient(params, target_f_ab, current_alpha, n_qubits_u, n_layers_u, n_qubits_v, n_layers_v)
            grad_norm = np.linalg.norm(gradients)
            if grad_norm < 1e-10:
                break
            if grad_norm > 0.8:
                gradients = gradients * (0.8 / grad_norm)
            optimizer.lr = 0.01 + 0.24 * (1 + np.cos(np.pi * progress)) / 2
            params = optimizer.step(params, gradients)
            if current_loss < best_loss - 1e-9:
                best_loss = current_loss
                best_params = params.copy()
                patience = 0
            else:
                patience += 1
            if patience >= 50:
                break

        u_params = best_params[:n_params_u]
        v_params = best_params[n_params_u:]
        final_fidelities = evaluate_attack_fidelities(u_params, v_params, n_qubits_u, n_layers_u, n_qubits_v, n_layers_v)
        f_ab = final_fidelities['F_AB']
        f_ae = final_fidelities['F_AE']
        f_ab_error = abs(f_ab - target_f_ab)
        theta = 2.0 * np.arccos(np.clip(2.0 * f_ab - 1.0, -1.0, 1.0))
        pccm_f_ae = (1.0 + np.sin(theta / 2.0)) / 2.0
        f_ae_gap = pccm_f_ae - f_ae
        # Unbiased distance: prioritize F_AE maximization
        distance = np.sqrt((f_ab_error * 3.0) ** 2 + (1.0 - f_ae) ** 2)

        if attempt % 5 == 0 or distance < best_distance:
            print(f"    🔄 {attempt+1:2d}/20: F_AB={f_ab:.4f}, F_AE={f_ae:.4f}, gap={f_ae_gap:.4f}")

        if distance < best_distance:
            best_distance = distance
            best_result = {
                'best_params': best_params,
                'F_AB': f_ab,
                'F_AE': f_ae,
                'PCCM_gap': f_ae_gap,
                'F_AB_error': f_ab_error,
                'success': True
            }

    return best_result if best_result else {'success': False, 'PCCM_gap': float('inf')}

# Main execution
if __name__ == "__main__":
    targets = [0.6771, 0.7221, 0.7754, 0.8096, 0.8470, 0.9048, 0.9504]
    your_gaps = [0.0781, 0.0586, 0.0401, 0.0020, 0.0235, 0.0265, 0.0242]
    
    print(f"\n🎓 QUICK CURRICULUM OPTIMIZATION")
    print("-" * 50)
    
    results = []
    best_params = None
    
    for i, target in enumerate(targets):
        print(f"\n📚 Step {i+1}/7: Target {target:.4f} (current gap: {your_gaps[i]:.4f})")
        result = optimize_quick(target, best_params)
        results.append(result)
        
        if result['success']:
            best_params = result['best_params']
            gap = result['PCCM_gap']
            if gap < 0.01:
                status = "🟢 EXCELLENT"
            elif gap < 0.02:
                status = "🟡 GOOD"
            elif gap < your_gaps[i]:
                status = "🟠 IMPROVED"
            else:
                status = "🔴 WORSE"
            print(f"    ✅ Result: F_AB={result['F_AB']:.4f}, F_AE={result['F_AE']:.4f}")
            print(f"    📊 Gap: {gap:.4f} - {status}")
    
    # Final comparison
    print(f"\n🏆 RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Target':<8} {'Your Gap':<10} {'New Gap':<10} {'Change':<10} {'Status':<12}")
    print("-" * 70)
    
    successful = [r for r in results if r['success']]
    improvements = 0
    
    for i, (target, your_gap) in enumerate(zip(targets, your_gaps)):
        if i < len(results) and results[i]['success']:
            new_gap = results[i]['PCCM_gap']
            change = (your_gap - new_gap) / your_gap * 100
            if change > 20:
                status = "🟢 MUCH BETTER"
                improvements += 1
            elif change > 0:
                status = "🟡 BETTER"
                improvements += 1
            elif change > -15:
                status = "🟠 SIMILAR"
            else:
                status = "🔴 WORSE"
            print(f"{target:<8.4f} {your_gap:<10.4f} {new_gap:<10.4f} {change:+7.1f}%   {status:<12}")
        else:
            print(f"{target:<8.4f} {your_gap:<10.4f} {'FAILED':<10} {'N/A':<10} {'❌ FAILED':<12}")
    
    if successful:
        new_gaps = [r['PCCM_gap'] for r in successful]
        new_avg = np.mean(new_gaps)
        your_avg = np.mean(your_gaps)
        overall_improvement = (your_avg - new_avg) / your_avg * 100
        
        print(f"\n📊 SUMMARY:")
        print(f"   Success: {len(successful)}/7")
        print(f"   Your avg: {your_avg:.4f}")
        print(f"   New avg:  {new_avg:.4f}")
        print(f"   Change:   {overall_improvement:+.1f}%")
        print(f"   Improved: {improvements}/7")
        
        excellent = sum(1 for gap in new_gaps if gap < 0.01)
        good = sum(1 for gap in new_gaps if 0.01 <= gap < 0.02)
        print(f"\n🏅 Quality: {excellent} excellent, {good} good")
        
        if excellent >= 2:
            print(f"🎉 GREAT! {excellent} excellent results!")
        elif good + excellent >= 4:
            print(f"👍 GOOD! {good + excellent} good/excellent!")
        elif improvements >= 4:
            print(f"📈 PROGRESS! {improvements} improved!")
    
    print(f"\n{'='*70}")
    print("✅ QUICK OPTIMIZATION COMPLETE!")
    print("💡 Results ready for review")
    print("=" * 70)
