# RVPI-PDE reuse map

| Component | Source file | Source class/function | Decision | Required adaptation / excluded reason |
|---|---|---|---|---|
| Global neural operator | rvpi/models/fno.py | FNO2d | REUSE_WITH_ADAPTER | Reused through project wrapper; field channels differ from upstream data. |
| Conditional operator | rvpi/models/conditional_fno.py | ConditionalFNO2d | REUSE_WITH_ADAPTER | Candidate for later conditioned histories; not needed for first smoke. |
| Local correction proposal | rvpi/models/fno.py | FNO2d | REUSE_WITH_ADAPTER | Wrapper applies it only to a manually selected patch. |
| Autoregressive rollout pattern | rvpi/pipelines/shallow_water.py | coarse rollout helpers | REUSE_WITH_ADAPTER | Future transition dataset can adopt it after scenarios exist. |
| Selector/RV-PI/policy/budget logic | rvpi/models/set_aware_selector.py, macro_policy.py, rvpi/env | RL components | DO_NOT_REUSE_FOR_CHINESE_EI | Explicitly excluded in this phase. |

Only the non-RL FNO building block is imported in the smoke demo. No selector, policy, reward, or RL training loop is run.
