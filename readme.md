# Simultaneous Synthesis and Verification of Neural Control Barrier Functions

Implementation of the Branch-and-Bound Verification-in-the-Loop Training (BBVT) scheme for Neural Control Barrier Functions, as presented in our ECC 2024 paper titled "Simultaneous Synthesis and Verification of Neural Control Barrier Functions through Branch-and-Bound Verification-in-the-Loop Training"

[![](assets/paper_teaser.jpg)](https://arxiv.org/pdf/2311.10438.pdf)

Control Barrier Functions (CBFs) that provide formal safety guarantees have been widely used for safetycritical systems. However, it is non-trivial to design a CBF. Utilizing neural networks as CBFs has shown great success,
but it necessitates their certification as CBFs. In this work, we leverage bound propagation techniques and the Branchand-Bound scheme to efficiently verify that a neural network satisfies the conditions to be a CBF over the continuous state space. To accelerate training, we further present a framework that embeds the verification scheme into the training loop to synthesize and verify a neural CBF simultaneously. In particular, we employ the verification scheme to identify partitions of the state space that are not guaranteed to satisfy the CBF conditions and expand the training dataset by incorporating additional data from these partitions. The neural network is then optimized using the augmented dataset to meet the CBF conditions. We show that for a non-linear control-affine system, our framework can efficiently certify a neural network as a CBF and render a larger safe set than state-of-the-art neural CBF works. We further employ our learned neural CBF to derive a safe controller to illustrate the practical use of our framework.

<div style="text-align:center;">
<img src="assets/schematic_overview.jpg" alt="BBVT Scheme" width="400">
</div>


## Main entrance

```bash
python3 safe_rl_cbf/main/train_model.py --config_file inverted_pendulum_pretrained.json
```
