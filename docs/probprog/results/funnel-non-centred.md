# funnel/non-centred

A hierarchical funnel (Rubin's eight schools, centred) against the same model non-centred by `reparam`: divergences and the smallest tau reached.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (draws 400, chains 2), 4.8 s.

**Passed**: the centred model diverges, the non-centred one does not

| parametrisation | divergences | smallest tau | chains x draws |
| --------------- | ----------- | ------------ | -------------- |
| centred         | 3           | 0.77         | 2 x 400        |
| non-centred     | 0           | 0.014        | 2 x 400        |
