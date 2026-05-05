# Privilege-Free Proxy Validation Memo

Generated: `2026-05-04T19:21:23.028626+00:00`

## Overall Correlations

| proxy metric | corr with selection_regret | p(selection_regret) | corr with Rpool(C_model) | p(Rpool) | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| pool_Cmodel_std | 0.254 | 0.011 | 0.267 | 0.007 | 100 |
| top30_Cmodel_std | 0.314 | 0.001 | 0.132 | 0.191 | 100 |
| C_model_dynamic_range | 0.223 | 0.026 | 0.294 | 0.003 | 100 |
| elite_compression_ratio | -0.009 | 0.929 | -0.282 | 0.004 | 100 |

## Per-Subset Correlations

| subset | proxy metric | corr with selection_regret | p(selection_regret) | corr with Rpool(C_model) | p(Rpool) | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| invisible_quadrant | pool_Cmodel_std | 0.588 | 0.017 | 0.147 | 0.587 | 16 |
| invisible_quadrant | top30_Cmodel_std | 0.159 | 0.557 | 0.350 | 0.184 | 16 |
| invisible_quadrant | C_model_dynamic_range | 0.576 | 0.019 | 0.076 | 0.778 | 16 |
| invisible_quadrant | elite_compression_ratio | -0.615 | 0.011 | 0.276 | 0.300 | 16 |
| sign_reversal | pool_Cmodel_std | 0.269 | 0.239 | 0.108 | 0.642 | 21 |
| sign_reversal | top30_Cmodel_std | 0.236 | 0.302 | 0.157 | 0.496 | 21 |
| sign_reversal | C_model_dynamic_range | 0.338 | 0.134 | 0.047 | 0.841 | 21 |
| sign_reversal | elite_compression_ratio | 0.016 | 0.947 | 0.049 | 0.832 | 21 |
| latent_favorable | pool_Cmodel_std | 0.315 | 0.319 | 0.455 | 0.138 | 12 |
| latent_favorable | top30_Cmodel_std | 0.566 | 0.055 | 0.315 | 0.319 | 12 |
| latent_favorable | C_model_dynamic_range | 0.294 | 0.354 | 0.503 | 0.095 | 12 |
| latent_favorable | elite_compression_ratio | 0.028 | 0.931 | -0.343 | 0.276 | 12 |
| v1_favorable | pool_Cmodel_std | 0.434 | 0.138 | 0.099 | 0.748 | 13 |
| v1_favorable | top30_Cmodel_std | 0.082 | 0.789 | 0.187 | 0.541 | 13 |
| v1_favorable | C_model_dynamic_range | 0.429 | 0.144 | 0.176 | 0.566 | 13 |
| v1_favorable | elite_compression_ratio | -0.495 | 0.086 | 0.198 | 0.517 | 13 |
| ordinary | pool_Cmodel_std | 0.209 | 0.159 | 0.249 | 0.092 | 47 |
| ordinary | top30_Cmodel_std | 0.328 | 0.025 | 0.051 | 0.731 | 47 |
| ordinary | C_model_dynamic_range | 0.175 | 0.240 | 0.279 | 0.057 | 47 |
| ordinary | elite_compression_ratio | 0.131 | 0.379 | -0.437 | 0.002 | 47 |

## Cross-Check

`corr(pool_Cmodel_std, pool_Creal_std)` = `0.482` with p-value `<0.001` over n=`100` pairs.

## Conclusion

A subset-specific privilege-free monitoring signal exists, but no global 100-pair proxy crossed the threshold. The strongest proxy for `selection_regret` was `top30_Cmodel_std` with Spearman `0.314` (p=`0.001`, n=`100`). The strongest subset-specific proxy was `pool_Cmodel_std` in `invisible_quadrant` with Spearman `0.588` (p=`0.017`, n=`16`). The decision threshold was Spearman > `0.4`.
