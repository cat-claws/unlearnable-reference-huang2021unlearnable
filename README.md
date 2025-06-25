# An implementation for unlearnable examples

This repository is based on a method by [Huang et al.](https://openreview.net/forum?id=iAmZUo0DxC0).

We have **fixed key issues** in the original implementation and **extended support** for CIFAR-10, CIFAR-100, and TinyImageNet datasets — maintaining a clean, modular, and reproducible codebase.

---

## 🔧 Key Fixes & Improvements

### Fixed Dataset Handling
- **TinyImageNet resizing bug fixed**:  
  The original repo resized TinyImageNet to 32×32, which defeats the purpose of evaluating on its higher-resolution format.  
  We preserve the correct **64×64 resolution**.

### Functional Argument Parsing
- **Broken or unused arguments removed**:  
  The original code included parameters like `--poison_rate` that were never used.  
  We've cleaned up the argument parser so all parameters are now meaningful and functional.

### Modular Script Conversion
- The only reliable part of the original repo was the `QuickStart.ipynb` notebook.  
  We've:
  - Converted it into modular `.py` scripts.
  - Added clean support for **CIFAR-10**, **CIFAR-100**, and **TinyImageNet**.

---

Run the following scripts to generate unlearnable examples:

```bash
# CIFAR-10
python cifar10.py

# CIFAR-100
python cifar100.py

# TinyImageNet (64×64)
python tinyimagenet.py
```


