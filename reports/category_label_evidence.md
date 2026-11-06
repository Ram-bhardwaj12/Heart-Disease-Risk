# Category Label Evidence

Inferred, not documented by the dataset provider.

Evidence derived from `data/processed/heart_deduplicated.csv` (302 unique patients) with `disease = 1 - target` (where `disease = 1` indicates heart disease present).

### Sex (`sex`)
| Code | Proposed Label | Patient Count | Disease Rate |
| ---: | :--- | ---: | ---: |
| 0 | Female | 96 | 0.2500 (25.0%) |
| 1 | Male | 206 | 0.5534 (55.3%) |

### Chest Pain Type (`cp`)
| Code | Proposed Label | Patient Count | Disease Rate |
| ---: | :--- | ---: | ---: |
| 0 | Asymptomatic (no chest pain) | 143 | 0.7273 (72.7%) |
| 1 | Atypical angina | 50 | 0.1800 (18.0%) |
| 2 | Non-anginal pain | 86 | 0.2093 (20.9%) |
| 3 | Typical angina | 23 | 0.3043 (30.4%) |

### Resting ECG (`restecg`)
| Code | Proposed Label | Patient Count | Disease Rate |
| ---: | :--- | ---: | ---: |
| 0 | Left ventricular hypertrophy | 147 | 0.5374 (53.7%) |
| 1 | Normal | 151 | 0.3709 (37.1%) |
| 2 | ST-T wave abnormality | 4 | 0.7500 (75.0%) |

### ST Slope (`slope`)
| Code | Proposed Label | Patient Count | Disease Rate |
| ---: | :--- | ---: | ---: |
| 0 | Downsloping | 21 | 0.5714 (57.1%) |
| 1 | Flat | 140 | 0.6500 (65.0%) |
| 2 | Upsloping | 141 | 0.2482 (24.8%) |

### Thallium Stress Test Result (`thal`)
| Code | Proposed Label | Patient Count | Disease Rate |
| ---: | :--- | ---: | ---: |
| 1 | Fixed defect | 18 | 0.6667 (66.7%) |
| 2 | Normal | 165 | 0.2182 (21.8%) |
| 3 | Reversible defect | 117 | 0.7607 (76.1%) |

*Note: 2 records in the deduplicated dataset have missing values for `thal` (treated as missing).*

