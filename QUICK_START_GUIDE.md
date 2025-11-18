# Quick Start Guide - Optimized QKD BB84

## 🚀 Run Immediately

```bash
cd /Users/jasonli/Documents/FIRE298/QKD_BB84/FIRE298-QML-Boycott-Superbowl
python3 QUICK_RUN_OPTIMIZED.py
```

**Time:** 5-10 minutes  
**Output:** 7 optimized QCL attack results

---

## 📊 What You'll Get

```
Target F_AB: 0.67, 0.72, 0.77, 0.82, 0.87, 0.92, 0.97

For each target:
- F_AB: Achieved fidelity (±0.03 accuracy)
- F_AE: Eve's information (maximized)
- PCCM Gap: Reference metric
- Status: Quality indicator
```

---

## 🎯 Key Features

✅ **Unbiased Optimization** - No artificial PCCM curve bias  
✅ **IONQ Ready** - 2-layer circuits, low QBER  
✅ **Fast** - 60-75% faster than before  
✅ **Clean** - 35% fewer files  

---

## 📁 Files You Need

### Run These
- `QUICK_RUN_OPTIMIZED.py` - Fast run (5-10 min)
- `run_ultra_optimization.py` - Extended run (15-20 min)
- `QKD_with_QCL_OPTIMIZED.ipynb` - Interactive notebook

### Read These
- `README.md` - Full documentation
- `OPTIMIZATION_SUMMARY.md` - Complete changes
- `EXPECTED_IMPROVEMENTS.md` - Performance analysis

---

## 🔧 Options

### Quick Run (Recommended)
```bash
python3 QUICK_RUN_OPTIMIZED.py
```
- 12 attempts per target
- 200 iterations
- ~1-2 min per target

### Ultra Run (Better Results)
```bash
python3 run_ultra_optimization.py
```
- 15 attempts per target
- 250 iterations
- ~2-3 min per target

### Interactive
```bash
jupyter notebook QKD_with_QCL_OPTIMIZED.ipynb
```

---

## 💡 What Changed

### Before
- ❌ 35 attempts, 600 iterations (slow)
- ❌ Biased toward PCCM curve
- ❌ 3-4 layer circuits (high QBER)
- ❌ 42 files (cluttered)

### After
- ✅ 12-15 attempts, 200-250 iterations (fast)
- ✅ Unbiased F_AE maximization
- ✅ 2-layer circuits (IONQ optimized)
- ✅ 27 files (clean)

---

## 🎓 Understanding Results

### F_AB (Alice-Bob Fidelity)
- Target: Match specified value
- Tolerance: ±0.03
- Status: ✅ if within tolerance

### F_AE (Alice-Eve Fidelity)
- Goal: Maximize (Eve's information)
- Method: Unbiased optimization
- No artificial PCCM bias

### PCCM Gap
- Reference: Theoretical bound
- Not optimization target
- For comparison only

---

## 🔬 IONQ Deployment (Optional)

### Setup
```bash
cp .env.example .env
# Edit .env: IONQ_API_KEY=your_key
```

### Run
```bash
jupyter notebook ionQ_QKD.ipynb
```

### Benefits
- Real quantum hardware
- Low QBER design
- 2-layer circuits
- Production-ready

---

## ⚡ Performance

| Metric | Value |
|--------|-------|
| **Execution Time** | 5-10 min (quick), 15-20 min (ultra) |
| **Speedup** | 60-75% faster |
| **Circuit Depth** | 2 layers (was 3-4) |
| **Parameters** | 24 (was 30-42) |
| **Files** | 27 (was 42) |

---

## 🆘 Troubleshooting

### Import Error
```bash
pip3 install -r requirements.txt
```

### Slow Execution
- Use `QUICK_RUN_OPTIMIZED.py` (not ultra)
- Already optimized for speed

### IONQ Issues
- Check `.env` file
- Verify API key
- Use simulator for testing

---

## ✅ Ready to Run

Everything is installed and optimized. Just run:

```bash
python3 QUICK_RUN_OPTIMIZED.py
```

**That's it!** Results in 5-10 minutes.

---

**Status:** ✅ Complete  
**Dependencies:** ✅ Installed  
**Optimized:** ✅ Yes  
**IONQ Ready:** ✅ Yes
