# QKD BB84 with Quantum Curriculum Learning Attack

## Overview
Optimized implementation of BB84 Quantum Key Distribution with QCL (Quantum Curriculum Learning) attack optimization. This implementation is designed for efficient execution on IONQ quantum hardware with low QBER rates.

## Key Features
- **Unbiased QCL Attack**: Uses unbiased optimization methods without PCCM curve bias
- **IONQ Optimized**: 2-layer architecture optimized for IONQ hardware
- **Fast Execution**: Reduced iterations (200-250) and attempts (12-15) for quick results
- **Low QBER**: Optimized for low Quantum Bit Error Rates on real quantum hardware

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Setup IONQ (Optional)
Create a `.env` file with your IONQ API key:
```bash
cp .env.example .env
# Edit .env and add your IONQ_API_KEY
```

### Run Optimized QCL Attack

**Quick Run** (~5-10 minutes):
```bash
python3 QUICK_RUN_OPTIMIZED.py
```
- 12 attempts per target
- 200 iterations per attempt
- 7 curriculum targets

**Ultra Run** (~15-20 minutes):
```bash
python3 run_ultra_optimization.py
```
- 15 attempts per target
- 250 iterations per attempt
- 7 curriculum targets

**Interactive Notebook**:
```bash
jupyter notebook QKD_with_QCL_OPTIMIZED.ipynb
```

## Architecture

### Optimized Parameters
- **U Circuit**: 2 qubits, 2 layers (12 parameters)
- **V Circuit**: 1 qubit, 2 layers (6 parameters per basis)
- **Total**: 24 parameters
- **Optimizer**: Adam with cosine annealing
- **Learning Rate**: 0.18 → 0.01
- **Alpha Schedule**: 20.0 → 3.0 (exponential decay)

### Unbiased Loss Function
```python
Loss = α(F_AB - target)² - F_AE + penalty(F_AB_error) + L2_regularization
```

**Key Changes from Previous Version**:
- ❌ Removed PCCM curve bias penalties
- ✅ Focus on maximizing F_AE directly
- ✅ Moderate F_AB accuracy penalty only
- ✅ Unbiased distance metric

## Expected Results

### Target Fidelities
```
Target F_AB: 0.67, 0.72, 0.77, 0.82, 0.87, 0.92, 0.97
```

### Performance Metrics
- **F_AB Accuracy**: ±0.03 tolerance
- **F_AE**: Maximized without bias
- **PCCM Gap**: Reference metric only
- **QBER**: Optimized for low rates on IONQ

## Files

### Core Files
- `QUICK_RUN_OPTIMIZED.py` - Fast optimized run (recommended)
- `run_ultra_optimization.py` - Extended optimization
- `QKD_with_QCL_OPTIMIZED.ipynb` - Interactive notebook
- `ionQ_QKD.ipynb` - IONQ hardware deployment

### Configuration
- `requirements.txt` - Python dependencies
- `.env.example` - IONQ API key template

### Documentation
- `README.md` - This file
- `HOW_TO_RUN.md` - Detailed instructions
- `EXPECTED_IMPROVEMENTS.md` - Performance analysis

## Optimization Strategy

### 1. Unbiased Approach
- No artificial penalties toward PCCM curve
- Natural F_AE maximization
- Balanced F_AB accuracy control

### 2. Curriculum Learning
- Progressive difficulty: 0.67 → 0.97
- Warm start from previous target
- Knowledge transfer between targets

### 3. IONQ Compatibility
- Reduced circuit depth (2 layers)
- Native gate set optimization
- Low QBER design

## Usage Examples

### Basic Run
```python
python3 QUICK_RUN_OPTIMIZED.py
```

### Custom Target
```python
from QUICK_RUN_OPTIMIZED import optimize_quick
result = optimize_quick(target_f_ab=0.85)
print(f"F_AB: {result['F_AB']:.4f}")
print(f"F_AE: {result['F_AE']:.4f}")
```

### IONQ Deployment
See `ionQ_QKD.ipynb` for hardware deployment examples.

## Performance

### Execution Time
- Quick Run: ~5-10 minutes (7 targets)
- Ultra Run: ~15-20 minutes (7 targets)
- Per Target: ~1-3 minutes

### Resource Usage
- CPU: Moderate (gradient computation)
- Memory: ~500MB-1GB
- Disk: Minimal

## Troubleshooting

### Import Errors
```bash
pip install --upgrade qiskit qiskit-aer qiskit-ionq
```

### IONQ Connection
- Verify API key in `.env`
- Check IONQ account status
- Use simulator for testing

### Slow Execution
- Use QUICK_RUN_OPTIMIZED.py instead of ultra
- Reduce n_attempts in code
- Use fewer targets

## Citation
If you use this code, please cite the original BB84 and QCL attack papers.

## License
See LICENSE file for details.
