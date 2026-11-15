# SWE015 Customer Support Ticket Router

<!-- DATA-CHECKS-START -->
### Data checks and limitations (written by scripts/data_checks.py)

Number of texts that appear in two splits (compared case-insensitively):

| Splits | Official CLINC, raw | Our splits, raw | Our splits, after preprocessing |
|---|---|---|---|
| train & test | 2 | 0 | 15 |
| train & val | 3 | 0 | 12 |
| val & test | 0 | 0 | 5 |

Our splits = official CLINC splits with the train rows that also appear in val/test removed, plus the 100 appliance rows.

Spelling correction on 18345 CLINC rows of train+val: 2 tokens changed (2 distinct). Words in the CLINC training vocabulary are never changed, so all of them come from validation rows. Each one is a wrong correction (CLINC has no deliberate typos):

- `refrigerated` -> `refrigerator` (1x)
- `ridge` -> `fridge` (1x)
<!-- DATA-CHECKS-END -->
