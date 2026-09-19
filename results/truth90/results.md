# truth90_validation: complete experimental tables

90 systems; 90 cases; 1 model paths per case. Equal systems within each tier.
Paired comparisons resample systems. A lower difference is favorable only for error/severity metrics; valid fractions favor higher values.

## T1

| Metric | QField-Dyn | Linear | NeuralMD |
|---|---:|---:|---:|
| Geo_RMSD_A | 1.73966 | 2.48628 | 1.35093 |
| Geo_coordinate_ES_A | 1.80236 | 2.66759 | 1.4148 |
| Phys_v1_valid | 0.24 | 0.0166667 | NA |
| Phys_v2_valid | 0.933333 | 0.0433333 | NA |
| Phys_bond_count | 9.56 | 36.2 | NA |
| Phys_angle_count | 7.50667 | 28.4533 | NA |
| Phys_environment_count | 0.216667 | 0.623333 | NA |
| Phys_common_heavy_valid | 0.28 | 0.0633333 | 0.846667 |
| Phys_common_heavy_environment_count | 0.216667 | 0.623333 | 0 |
| Phys_v2_common_heavy_valid | 0.936667 | 0.14 | 0.966667 |
| Phys_v2_common_heavy_environment_valid | 0.953333 | 0.763333 | 1 |
| Dyn_RMSF_MAE_A | 0.371699 | 0.342285 | 0.818038 |
| Dyn_predicted_RMSF_A | 0.726401 | 0.762998 | 0.0286066 |
| Dyn_contact_Brier | 0.156211 | 0.183568 | 0.107791 |
| Stab_late_RMSD_A | 2.00203 | 3.54059 | 1.55855 |
| Feature_ES | 0.436121 | 0.63476 | 0.418868 |

## T2

| Metric | QField-Dyn | Linear | NeuralMD |
|---|---:|---:|---:|
| Geo_RMSD_A | 2.47503 | 4.40378 | 2.06994 |
| Geo_coordinate_ES_A | 2.62429 | 4.83494 | 2.20372 |
| Phys_v1_valid | 0.311667 | 0.01 | NA |
| Phys_v2_valid | 0.921667 | 0.0233333 | NA |
| Phys_bond_count | 6.51833 | 36.7 | NA |
| Phys_angle_count | 5.21167 | 30.535 | NA |
| Phys_environment_count | 0.0833333 | 2.02667 | NA |
| Phys_common_heavy_valid | 0.378333 | 0.0416667 | 0.708333 |
| Phys_common_heavy_environment_count | 0.0833333 | 2.02667 | 0 |
| Phys_v2_common_heavy_valid | 0.93 | 0.0733333 | 0.923333 |
| Phys_v2_common_heavy_environment_valid | 0.965 | 0.495 | 1 |
| Dyn_RMSF_MAE_A | 0.849524 | 1.06506 | 1.3087 |
| Dyn_predicted_RMSF_A | 0.845401 | 1.62636 | 0.0642034 |
| Dyn_contact_Brier | 0.185595 | 0.250663 | 0.158318 |
| Stab_late_RMSD_A | 3.14926 | 6.79346 | 2.73329 |
| Feature_ES | 0.685971 | 1.19832 | 0.663845 |

## T3

| Metric | QField-Dyn | Linear | NeuralMD |
|---|---:|---:|---:|
| Geo_RMSD_A | 2.58947 | 15.4382 | 2.49482 |
| Geo_coordinate_ES_A | 2.75726 | 17.5283 | 2.6418 |
| Phys_v1_valid | 0.396667 | 0.000833333 | NA |
| Phys_v2_valid | 0.90875 | 0.00333333 | NA |
| Phys_bond_count | 8.77333 | 40.9267 | NA |
| Phys_angle_count | 8.23667 | 45.71 | NA |
| Phys_environment_count | 0.255833 | 5.31833 | NA |
| Phys_common_heavy_valid | 0.413333 | 0.00666667 | 0.1675 |
| Phys_common_heavy_environment_count | 0.255833 | 5.31833 | 0 |
| Phys_v2_common_heavy_valid | 0.9275 | 0.015 | 0.380833 |
| Phys_v2_common_heavy_environment_valid | 0.942083 | 0.241667 | 1 |
| Dyn_RMSF_MAE_A | 0.732433 | 6.09149 | 1.22154 |
| Dyn_predicted_RMSF_A | 1.05815 | 7.55373 | 0.243895 |
| Dyn_contact_Brier | 0.209473 | 0.41199 | 0.178978 |
| Stab_late_RMSD_A | 3.19157 | 25.1154 | 3.09453 |
| Feature_ES | 0.723382 | 5.22531 | 0.658341 |

## all

| Metric | QField-Dyn | Linear | NeuralMD |
|---|---:|---:|---:|
| Geo_RMSD_A | 2.26805 | 7.44277 | 1.9719 |
| Geo_coordinate_ES_A | 2.39464 | 8.3436 | 2.08677 |
| Phys_v1_valid | 0.316111 | 0.00916667 | NA |
| Phys_v2_valid | 0.92125 | 0.0233333 | NA |
| Phys_bond_count | 8.28389 | 37.9422 | NA |
| Phys_angle_count | 6.985 | 34.8994 | NA |
| Phys_environment_count | 0.185278 | 2.65611 | NA |
| Phys_common_heavy_valid | 0.357222 | 0.0372222 | 0.574167 |
| Phys_common_heavy_environment_count | 0.185278 | 2.65611 | 0 |
| Phys_v2_common_heavy_valid | 0.931389 | 0.0761111 | 0.756944 |
| Phys_v2_common_heavy_environment_valid | 0.953472 | 0.5 | 1 |
| Dyn_RMSF_MAE_A | 0.651219 | 2.49961 | 1.11609 |
| Dyn_predicted_RMSF_A | 0.876652 | 3.31436 | 0.112235 |
| Dyn_contact_Brier | 0.183759 | 0.282074 | 0.148362 |
| Stab_late_RMSD_A | 2.78096 | 11.8165 | 2.46212 |
| Feature_ES | 0.615158 | 2.3528 | 0.580351 |
