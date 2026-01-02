# 🎥 Video OCR Benchmark  
**Tesseract · EasyOCR · PaddleOCR**

This project implements a **video-based OCR benchmarking pipeline** that compares three popular OCR engines on extracted video frames.  
The focus is not only text extraction, but also **engine behavior, accuracy heuristics, and stability across frames**.

---

## 🚀 Features

- 🎞️ FPS-aware **video → frame extraction**
- 🔤 OCR with **three engines**
  - **Tesseract** (classical OCR)
  - **EasyOCR** (deep learning-based)
  - **PaddleOCR** (end-to-end document OCR pipeline)
- 🧠 **Engine-aware preprocessing**
- 📊 Heuristic accuracy estimation
- 📈 Comparative frame-level analysis
- 💻 **CPU-only**, low-resource friendly setup

---

## 🧩 Architecture Overview

