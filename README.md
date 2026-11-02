# Frequency-Decoupled Dual-Branch Dehazing Network with Mamba2 and Learnable Gabor Convolutions

**Paper:** Frequency-Decoupled Dual-Branch Dehazing Network with Mamba2 and Learnable Gabor Convolutions  
**Authors:** Ye Li, Hongyu Xiao 
**Institution:** School of Optical-Electrical and Computer Engineering, University of Shanghai for Science and Technology, Shanghai,China  

## Code Release Status

| Component | Status |
|-----------|--------|
| Model architecture | ✅ Available |
| Training code | ✅ Available |
| Validation & evaluation code | 🔒 Coming upon acceptance |

## Model Architecture

## Requirements
```
torch>=2.0.0
torchvision>=0.15.0
numpy
Pillow
pyyaml
einops>=0.4.0
tensorboard
lpips
# Install AFTER torch+CUDA (pip install mamba-ssm causal-conv1d)
mamba-ssm>=2.2.0
causal-conv1d>=1.4.0
```
Dependencies: pip install -r requirements.txt
For Train:python train.py

