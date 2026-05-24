# self_similarity

**Public.** Compute the **self-similarity** index from overnight
respiratory recordings — used in the loop-gain / central-apnea analyses
to summarise the consistency of breathing patterns across the night.

## Layout

```
MAIN_similarity.m       entry function: returns centrals/central_hypo/similarity
calcSIM2.m              core similarity computation
fcnCreateEnv.m          envelope helper
fcnDrawBreaths.m        breath-detection helper
```

## Required environment

MATLAB; signals are passed in as `(hdr, s, stage)` triples.

## Status

Public research utility.
