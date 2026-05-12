# lr_hybrid — data2 训练结果

## 性能指标

| 指标 | 值 |
|---|---|
| AUC | 0.6420 |
| F1 | 0.6118 |
| Accuracy | 0.6018 |
| ECE | 0.0115 |

## 超参数

- **C**: 0.1
- **cv_auc**: 0.639978232
- **n_features**: 27
- **n_physics_features**: 6
- **baseline_auc**: 0.556099445

## 特征重要性 (Top 15)

| 排名 | 特征 | 重要性 |
|---|---|---|
| 1 | cfa_periodicity | 7.6116 |
| 2 | channel_noise_corr | 5.8160 |
| 3 | branch_frequency | 5.3009 |
| 4 | fft_mid_ratio | 4.9973 |
| 5 | texture_energy | 4.5721 |
| 6 | directional_anisotropy | 4.5019 |
| 7 | jpeg_blockiness | 4.1266 |
| 8 | edge_density | 1.9518 |
| 9 | fft_low_ratio | 1.9377 |
| 10 | branch_smoothness | 1.9233 |
| 11 | smoothness_ai_risk | 1.9233 |
| 12 | branch_physical | 1.5700 |
| 13 | saturation_std | 1.4289 |
| 14 | residual_entropy | 1.4155 |
| 15 | spectral_slope | 1.1048 |
