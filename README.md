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


To **review the specific changes**, refer to the commit history:  
[**View commit differences here**](https://github.com/HanxunH/Unlearnable-Examples/commit/45554b3e9cb1a3daa70f81db5aa5019f2b55f4e4)  

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

A handy tool can be used (at the end of python script) to upload the unlearnable examples, e.g., CIFAR-10, to Huggingface datasets as follows
```
pip install git+https://github.com/cestwc/sharpen/
```

```python
from sharpen import push_images

push_images(
    X = unlearnable_train_dataset.data,
    dataset_repo = "your-repo",
    token = 'your_token',
    config_name = "some-cifar10",
    label_source = clean_train_dataset.targets, # or some source
    class_names='cifar10', # ['cifar10', 'cifar100', 'tinyimagenet'] or some repo name
)
```
