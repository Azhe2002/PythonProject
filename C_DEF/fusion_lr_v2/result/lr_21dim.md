# lr_21dim — data2 训练结果

## 性能指标

| 指标 | 值 |
|---|---|
| AUC | 0.6382 |
| F1 | 0.6104 |
| Accuracy | 0.5992 |
| ECE | 0.0105 |

## 超参数

- **C**: 10.0
- **cv_auc**: 0.635734138
- **n_features**: 21
- **baseline_auc**: 0.556099445

## 特征重要性 (Top 15)

| 排名 | 特征 | 重要性 |
|---|---|---|
| 1 | texture_energy | 20.0220 |
| 2 | cfa_periodicity | 7.1417 |
| 3 | directional_anisotropy | 5.6996 |
| 4 | edge_density | 5.2916 |
| 5 | jpeg_blockiness | 4.3120 |
| 6 | channel_noise_corr | 4.2640 |
| 7 | fft_high_ratio | 2.8377 |
| 8 | fft_mid_ratio | 1.9741 |
| 9 | fft_low_ratio | 1.3184 |
| 10 | color_channel_corr | 1.1843 |
| 11 | residual_abs_mean | 1.0655 |
| 12 | saturation_std | 1.0079 |
| 13 | spectral_slope | 0.7634 |
| 14 | noise_illum_corr | 0.6429 |
| 15 | residual_std | 0.5957 |
