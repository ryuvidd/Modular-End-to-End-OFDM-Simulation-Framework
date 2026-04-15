# End-to-End OFDM Simulation Framework

## 📌 Overview

This project implements a complete end-to-end Orthogonal Frequency Division Multiplexing (OFDM) communication system in Python, covering the full signal processing pipeline from data generation to channel estimation and equalization.

The system is designed with a modular **object-oriented architecture**, where each communication block is encapsulated as an independent component. This enables scalable experimentation, algorithm comparison, and reproducible evaluation.

---

## 🚀 Key Features

* 📡 Full OFDM baseband simulation pipeline
* 🔢 QAM modulation (e.g., QPSK Gray mapping)
* 📶 OFDM modulation/demodulation (IFFT/FFT + cyclic prefix)
* 🌊 Channel modeling (Rayleigh fading, configurable taps)
* 📊 Channel estimation:

  * Least Squares (LS)
  * Linear Minimum Mean Square Error (LMMSE)
* 📈 Channel interpolation (e.g., linear)
* ⚖️ Equalization (Zero-Forcing)
* 📉 Evaluation metrics:

  * Bit Error Rate (BER)
  * Normalized Mean Square Error (NMSE)
* 🔁 SNR sweep simulation for performance analysis
* 📊 Automated plotting (BER/NMSE vs SNR)

---

## 🧠 System Architecture

### High-Level Pipeline

```
Bits → QAM → OFDM Mod → Channel → Noise
     → OFDM Demod → Channel Estimation → Interpolation
     → Equalization → Demodulation → BER/NMSE
```

---

### 🧩 Modular Design

```
OFDMSystem
│
├── Transmitter
│   ├── BlocksGenerator (Pilot/Data Grid)
│   ├── QAM Modulator
│   └── OFDM Modulator
│
├── Channel
│   ├── Channel Model (Rayleigh)
│   └── Noise Mixer (AWGN)
│
├── Receiver
│   ├── OFDM Demodulator
│   ├── Channel Estimator (LS / LMMSE)
│   ├── Channel Interpolator
│   ├── Equalizer (ZF)
│   └── Demodulator
│
├── Evaluator
│   ├── BER
│   └── NMSE
│
└── Plotter
    ├── BER vs SNR
    └── NMSE vs SNR
```

---

## ⚙️ Configuration Design

The system uses a structured configuration object (`OFDMConfig`) to control:

* Number of subcarriers
* Number of OFDM blocks
* Pilot allocation
* Channel model parameters
* Estimator selection (LS / LMMSE)
* SNR range

This allows flexible experimentation without modifying core logic.

---

## 🔄 Data Flow & Metadata Handling

To ensure modularity and avoid hidden dependencies:

* **Configuration (`Config`)** → static system parameters
* **Metadata (`Meta`)** → runtime information (e.g., pilot indices, noise variance)

This separation improves:

* reproducibility
* maintainability
* debugging clarity

---

## 📊 Performance Evaluation

The system evaluates performance across an SNR range:

### Metrics

* **BER (Bit Error Rate)**
  Measures end-to-end detection performance

* **NMSE (Normalized Mean Square Error)**
  Evaluates channel estimation accuracy

---

### Example Output

* BER vs SNR (log scale)
* NMSE vs SNR (dB scale)
* Comparison between LS and LMMSE estimators

---

## 🧪 Example Usage

```python
configs = [
    OFDMConfig(Estimator=CHANNEL_ESTIMATOR.LS),
    OFDMConfig(Estimator=CHANNEL_ESTIMATOR.LMMSE)
]

for config in configs:
    system = OFDMSystem(config)
    result = system.run()
```

---

## 📈 Sample Results

* LMMSE consistently outperforms LS in NMSE
* Improved channel estimation leads to lower BER
* Performance gap increases at low SNR

---

## 🏗️ Engineering Highlights

* Designed a **modular transceiver architecture** aligned with communication system theory
* Implemented mathematically grounded algorithms (LS, LMMSE)
* Built a **scalable experiment pipeline** with SNR sweep support
* Ensured **clean separation between configuration and runtime metadata**
* Structured code for **extensibility and reproducibility**

---

## 🔮 Future Work

* MMSE equalizer implementation
* Time-varying channel models
* Deep learning-based channel estimation
* MIMO extension
* Computational complexity analysis

---

## 📚 Technical Stack

* Python
* NumPy
* Matplotlib
* Object-Oriented Design (OOP)

---

## 💬 Author Notes

This project bridges communication theory and practical implementation by translating signal processing models into a structured, extensible simulation framework.

---
