#!/usr/bin/env python3
"""
Ultra-optimized QCL attack to beat current results.
Target: PCCM gaps < 0.02 consistently
"""

import sys
import os
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

print("🚀 OPTIMIZED QCL ATTACK - NO PCCM BIAS")
print("=" * 80)
print("🎯 Goal: Natural F_AE maximization without artificial bias")
print("⚡ Strategy: More attempts/iterations + curriculum learning")
print("📊 PCCM penalties REMOVED - unbiased optimization")
print("🔧 Target: True optimal QCL attack in ~30-45 minutes")
print("=" * 80)

# Import required libraries
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, Operator
    print("✅ Qiskit imported successfully")
except ImportError as e:
    print(f"❌ Qiskit import failed: {e}")
    print("Please install: pip install qiskit qiskit-aer")
    sys.exit(1)

# Check for IONQ
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    from qiskit_ionq import IonQProvider
    ionq_token = os.getenv('IONQ_API_KEY')
    if ionq_token:
        provider = IonQProvider(token=ionq_token)
        backend_ionq = provider.get_backend('ionq_forte')
        print("✅ IONQ FORTE connected!")
        ionq_available = True
    else:
        print("⚠️  IONQ_API_KEY not found")
        ionq_available = False
except ImportError:
    print("⚠️  qiskit-ionq not available")
    ionq_available = False

# Initialize backend
backend = AerSimulator()

class AdamOptimizer:
    """Enhanced Adam optimizer."""
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
        
        params_new = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params_new

def hardware_efficient_ansatz(n_qubits: int, n_layers: int, params: np.ndarray) -> QuantumCircuit:
    """Standard ansatz for simulator."""
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
    """Compute quantum fidelity between density matrices."""
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
    fidelity = np.clip(np.real(fidelity), 0, 1)

    return float(fidelity)

def evaluate_attack_fidelities(u_params: np.ndarray, v_params: np.ndarray, 
                               n_qubits_u: int, n_layers_u: int,
                               n_qubits_v: int, n_layers_v: int) -> Dict[str, float]:
    """Evaluate attack fidelities."""
    test_states = [(0, 0), (1, 0), (0, 1), (1, 1)]
    
    n_params_v_single = n_qubits_v * n_layers_v * 3
    v_params_z = v_params[:n_params_v_single]
    v_params_x = v_params[n_params_v_single:2*n_params_v_single]

    f_ab_list = []
    f_ae_list = []

    for bit, basis in test_states:
        # Prepare Alice's state
        qc_alice = QuantumCircuit(n_qubits_u)
        if bit == 1:
            qc_alice.x(0)
        if basis == 1:
            qc_alice.h(0)

        # Apply Eve's attack circuit U
        qc_u = hardware_efficient_ansatz(n_qubits_u, n_layers_u, u_params)
        qc_combined = qc_alice.compose(qc_u)

        # Get density matrices
        statevec = Statevector.from_instruction(qc_combined)
        density_matrix_full = DensityMatrix(statevec)

        # Bob's reduced density matrix
        eve_qubits = list(range(1, n_qubits_u))
        dm_bob = partial_trace(density_matrix_full, eve_qubits)

        # Target state
        if bit == 0 and basis == 0:
            target = np.array([[1, 0], [0, 0]], dtype=complex)
        elif bit == 1 and basis == 0:
            target = np.array([[0, 0], [0, 1]], dtype=complex)
        elif bit == 0 and basis == 1:
            target = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        else:
            target = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex)

        # Compute Alice-Bob fidelity
        f_ab = compute_fidelity_density_matrix(target, dm_bob.data)
        f_ab_list.append(f_ab)

        # Eve's reduced density matrix
        dm_eve = partial_trace(density_matrix_full, [0])

        # Apply Eve's measurement circuit V
        current_v_params = v_params_x if basis == 1 else v_params_z
        qc_v = hardware_efficient_ansatz(n_qubits_v, n_layers_v, current_v_params)
        U_v = Operator(qc_v)
        dm_eve_measured = U_v @ dm_eve.data @ U_v.adjoint()

        # Compute Alice-Eve fidelity
        f_ae = compute_fidelity_density_matrix(target, dm_eve_measured)
        f_ae_list.append(f_ae)

    return {
        'F_AB': np.mean(f_ab_list),
        'F_AE': np.mean(f_ae_list)
    }

def ultra_loss_function(params: np.ndarray, target_f_ab: float, alpha: float,
                        n_qubits_u: int, n_layers_u: int,
                        n_qubits_v: int, n_layers_v: int) -> float:
    """Unbiased loss function for optimal QCL attack without PCCM curve bias."""
    
    n_params_u = n_qubits_u * n_layers_u * 3
    n_params_v = n_qubits_v * n_layers_v * 3

    u_params = params[:n_params_u]
    v_params = params[n_params_u:n_params_u + 2*n_params_v]

    fidelities = evaluate_attack_fidelities(u_params, v_params, n_qubits_u,
                                           n_layers_u, n_qubits_v, n_layers_v)

    f_ab = fidelities['F_AB']
    f_ae = fidelities['F_AE']

    # Base loss: balance F_AB accuracy with F_AE maximization
    base_loss = alpha * (f_ab - target_f_ab) ** 2 - f_ae

    # Unbiased F_AB accuracy penalty (no PCCM curve bias)
    f_ab_penalty = 0.0
    f_ab_error = abs(f_ab - target_f_ab)
    
    # Moderate penalty for F_AB accuracy only
    if f_ab_error > 0.03:
        f_ab_penalty += 50.0 * (f_ab_error - 0.03) ** 2
    
    # L2 regularization for hardware efficiency
    l2_penalty = 0.001 * np.sum(params ** 2)
    
    total_loss = base_loss + f_ab_penalty + l2_penalty
    return total_loss

def compute_gradient_ultra(params, target_f_ab, alpha, n_qubits_u, n_layers_u,
                          n_qubits_v, n_layers_v, shift=np.pi/2):
    """Ultra-enhanced gradient computation."""
    gradients = np.zeros_like(params)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += shift
        loss_plus = ultra_loss_function(params_plus, target_f_ab, alpha,
                                       n_qubits_u, n_layers_u,
                                       n_qubits_v, n_layers_v)
        
        params_minus = params.copy()
        params_minus[i] -= shift
        loss_minus = ultra_loss_function(params_minus, target_f_ab, alpha,
                                        n_qubits_u, n_layers_u,
                                        n_qubits_v, n_layers_v)
        
        gradients[i] = (loss_plus - loss_minus) / 2
    
    return gradients

def optimize_single_target_ultra(target_f_ab: float = 0.85,
                                initial_params: Optional[np.ndarray] = None,
                                verbose: bool = True) -> Dict:
    """Ultra-optimized single target optimization."""
    
    n_qubits_u = 2
    n_layers_u = 2  # Optimized for IONQ
    n_qubits_v = 1
    n_layers_v = 2  # Optimized for IONQ
    
    n_params_u = n_qubits_u * n_layers_u * 3
    n_params_v = n_qubits_v * n_layers_v * 3
    total_params = n_params_u + n_params_v * 2

    if verbose:
        print(f"  🎯 Target F_AB={target_f_ab:.3f}, {total_params} parameters")

    best_result = None
    best_distance = float('inf')
    n_attempts = 30  # More attempts for better results
    
    # Enhanced parameters
    lr_max = 0.20
    lr_min = 0.01
    alpha_start = 25.0  # Strong F_AB control
    alpha_end = 2.0     # Strong F_AE emphasis

    for attempt in range(n_attempts):
        # Smart initialization
        if initial_params is not None and attempt == 0:
            if len(initial_params) == total_params:
                params = initial_params.copy()
                # Add exploration noise
                params += np.random.normal(0, 0.05, size=params.shape)
                params = np.clip(params, -np.pi, np.pi)
            else:
                params = np.random.uniform(-np.pi, np.pi, total_params)
        else:
            # Smart restart near best solution
            if best_result and attempt > 8:
                params = best_result['best_params'].copy()
                params += np.random.normal(0, 0.2, size=params.shape)
                params = np.clip(params, -np.pi, np.pi)
            else:
                params = np.random.uniform(-np.pi, np.pi, total_params)
        
        optimizer = AdamOptimizer(learning_rate=lr_max)
        
        best_loss = float('inf')
        best_params = params.copy()
        patience = 0
        max_patience = 60
        max_iter = 500  # More iterations for better convergence

        for iteration in range(max_iter):
            # Ultra-adaptive alpha with exponential decay
            progress = iteration / max_iter
            current_alpha = alpha_start * np.exp(-4 * progress) + alpha_end
            
            current_loss = ultra_loss_function(params, target_f_ab, current_alpha,
                                              n_qubits_u, n_layers_u, 
                                              n_qubits_v, n_layers_v)

            gradients = compute_gradient_ultra(params, target_f_ab, current_alpha,
                                              n_qubits_u, n_layers_u,
                                              n_qubits_v, n_layers_v)

            grad_norm = np.linalg.norm(gradients)
            if grad_norm < 1e-10:
                break

            # Moderate gradient clipping
            max_grad_norm = 1.0
            if grad_norm > max_grad_norm:
                gradients = gradients * (max_grad_norm / grad_norm)

            # Cosine annealing learning rate
            current_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))
            
            optimizer.lr = current_lr
            params = optimizer.step(params, gradients)

            if current_loss < best_loss - 1e-9:
                best_loss = current_loss
                best_params = params.copy()
                patience = 0
            else:
                patience += 1

            if patience >= max_patience:
                break

        # Final evaluation
        u_params = best_params[:n_params_u]
        v_params = best_params[n_params_u:]
        final_fidelities = evaluate_attack_fidelities(u_params, v_params,
                                                      n_qubits_u, n_layers_u,
                                                      n_qubits_v, n_layers_v)

        f_ab = final_fidelities['F_AB']
        f_ae = final_fidelities['F_AE']
        f_ab_error = abs(f_ab - target_f_ab)

        # PCCM calculation for reference only
        theta = 2.0 * np.arccos(np.clip(2.0 * f_ab - 1.0, -1.0, 1.0))
        pccm_f_ae = (1.0 + np.sin(theta / 2.0)) / 2.0
        f_ae_gap = pccm_f_ae - f_ae

        # Unbiased distance metric: prioritize F_AE maximization
        distance = np.sqrt((f_ab_error * 3.0) ** 2 + (1.0 - f_ae) ** 2)

        if verbose:
            print(f"    🔄 {attempt+1:2d}/{n_attempts}: F_AB={f_ab:.4f}±{f_ab_error:.4f}, "
                  f"F_AE={f_ae:.4f}, gap={f_ae_gap:.4f}")

        if distance < best_distance:
            best_distance = distance
            best_result = {
                'best_params': best_params,
                'F_AB': f_ab,
                'F_AE': f_ae,
                'PCCM_gap': f_ae_gap,
                'F_AB_error': f_ab_error,
                'success': True,
                'distance': distance
            }

    return best_result if best_result else {
        'F_AB': 0.0, 'F_AE': 0.0, 'PCCM_gap': float('inf'),
        'F_AB_error': float('inf'), 'success': False
    }

def run_ultra_curriculum():
    """Run ultra-optimized curriculum learning."""
    
    # Slightly adjusted targets for better coverage
    targets = [0.67, 0.72, 0.77, 0.82, 0.87, 0.92, 0.97]
    results = []
    best_params = None
    
    print(f"\n🎓 ULTRA-CURRICULUM LEARNING")
    print("-" * 60)
    
    for i, target in enumerate(targets):
        print(f"\n📚 Step {i+1}/{len(targets)}: F_AB = {target:.3f}")
        print("-" * 40)
        
        result = optimize_single_target_ultra(
            target_f_ab=target,
            initial_params=best_params,
            verbose=True
        )
        
        results.append(result)
        
        if result['success']:
            best_params = result['best_params']
            gap = result['PCCM_gap']
            
            # Status indicator
            if gap < 0.01:
                status = "🟢 EXCELLENT"
            elif gap < 0.02:
                status = "🟡 GOOD"
            elif gap < 0.05:
                status = "🟠 FAIR"
            else:
                status = "🔴 POOR"
                
            print(f"    ✅ F_AB={result['F_AB']:.4f}, F_AE={result['F_AE']:.4f}")
            print(f"    📊 Gap={gap:.4f} - {status}")
        else:
            print(f"    ❌ FAILED")
    
    return results, targets

# Run the optimization
if __name__ == "__main__":
    print(f"\n🚀 STARTING ULTRA-OPTIMIZATION")
    
    try:
        results, targets = run_ultra_curriculum()
        
        # Analysis
        print(f"\n🏆 ULTRA-OPTIMIZATION RESULTS")
        print("=" * 60)
        
        successful = [r for r in results if r['success']]
        if successful:
            gaps = [r['PCCM_gap'] for r in successful]
            avg_gap = np.mean(gaps)
            min_gap = min(gaps)
            max_gap = max(gaps)
            
            # Your current results for comparison
            your_gaps = [0.0781, 0.0586, 0.0401, 0.0020, 0.0235, 0.0265, 0.0242]
            your_avg = np.mean(your_gaps)
            
            improvement = (your_avg - avg_gap) / your_avg * 100
            
            print(f"📊 Success Rate: {len(successful)}/{len(targets)} ({len(successful)/len(targets)*100:.1f}%)")
            print(f"📉 Average Gap: {avg_gap:.4f} (range: {min_gap:.4f} - {max_gap:.4f})")
            print(f"📈 Your Average: {your_avg:.4f}")
            print(f"🎯 Improvement: {improvement:+.1f}%")
            
            # Detailed comparison
            print(f"\n📋 DETAILED COMPARISON")
            print("-" * 60)
            print(f"{'Target':<8} {'Your Gap':<10} {'New Gap':<10} {'Change':<10} {'Status':<12}")
            print("-" * 60)
            
            for i, (target, your_gap) in enumerate(zip(targets, your_gaps)):
                if i < len(results) and results[i]['success']:
                    new_gap = results[i]['PCCM_gap']
                    change = (your_gap - new_gap) / your_gap * 100
                    
                    if change > 20:
                        status = "🟢 MUCH BETTER"
                    elif change > 0:
                        status = "🟡 BETTER"
                    elif change > -10:
                        status = "🟠 SIMILAR"
                    else:
                        status = "🔴 WORSE"
                    
                    print(f"{target:<8.2f} {your_gap:<10.4f} {new_gap:<10.4f} {change:+7.1f}%   {status:<12}")
                else:
                    print(f"{target:<8.2f} {your_gap:<10.4f} {'FAILED':<10} {'N/A':<10} {'❌ FAILED':<12}")
            
            # Count excellent results
            excellent = sum(1 for r in successful if r['PCCM_gap'] < 0.01)
            good = sum(1 for r in successful if 0.01 <= r['PCCM_gap'] < 0.02)
            
            print(f"\n🏅 QUALITY BREAKDOWN:")
            print(f"   🟢 Excellent (< 0.01): {excellent}/{len(targets)}")
            print(f"   🟡 Good (< 0.02):      {good}/{len(targets)}")
            
            if excellent >= 4:
                print(f"\n🎉 OUTSTANDING! Achieved excellent results for {excellent} targets!")
            elif good + excellent >= 5:
                print(f"\n👍 SUCCESS! Achieved good/excellent results for {good + excellent} targets!")
            else:
                print(f"\n📈 PROGRESS! Some improvements achieved, consider further tuning.")
        
        else:
            print("❌ No successful optimizations. Check parameters.")
            
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("🚀 ULTRA-OPTIMIZATION COMPLETE!")
    print("💡 Results are ready for your research paper")
    print("📊 Use these optimized parameters with your professor")
    print("=" * 80)
