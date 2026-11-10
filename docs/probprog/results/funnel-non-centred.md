# funnel/non-centred

A hierarchical funnel (Rubin's eight schools, centred) against the same model non-centred by `reparam`: divergences and the smallest tau reached.

Run on 2026-11-10 at commit 4ce8b576; seed 0, budget scale 3 (draws 1200, chains 2), 10.6 s.

**Passed**: the centred model diverges, the non-centred one does not

| parametrisation | divergences | smallest tau | chains x draws |
| --------------- | ----------- | ------------ | -------------- |
| centred         | 12          | 0.57         | 2 x 1200       |
| non-centred     | 0           | 0.0047       | 2 x 1200       |
