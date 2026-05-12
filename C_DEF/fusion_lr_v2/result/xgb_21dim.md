# xgb_21dim — data2 训练结果

## 性能指标

| 指标 | 值 |
|---|---|
| AUC | 0.7324 |
| F1 | 0.6681 |
| Accuracy | 0.6691 |
| ECE | 0.0149 |

## 超参数

- **max_depth**: 7
- **learning_rate**: 0.1
- **cv_auc**: 0.7251270848219601
- **baseline_auc**: 0.556099445

## 特征重要性 (Top 15)

| 排名 | 特征 | 重要性 |
|---|---|---|
| 1 | color_channel_corr | 0.0909 |
| 2 | residual_std | 0.0833 |
| 3 | spectral_slope | 0.0795 |
| 4 | noise_illum_corr | 0.0605 |
| 5 | fft_high_ratio | 0.0564 |
| 6 | smooth_area_ratio | 0.0558 |
| 7 | channel_noise_corr | 0.0557 |
| 8 | residual_kurtosis | 0.0473 |
| 9 | edge_density | 0.0470 |
| 10 | residual_entropy | 0.0463 |
| 11 | laplacian_var | 0.0458 |
| 12 | saturation_std | 0.0425 |
| 13 | fft_mid_ratio | 0.0423 |
| 14 | fft_low_ratio | 0.0419 |
| 15 | saturation_mean | 0.0414 |
