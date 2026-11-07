# Progress

Statuses: planned, in progress, complete, dropped. Times are local, 2026-11-07, taken from commit times for finished
work; "about" means reconstructed from the session. Update a row when its state changes.

"Commit or file" is where the work or its evidence is: the commits of finished work, the file a
finding will be written to, or the section of `findings.md` or `reports/` a planned item comes from.

## Phase 0: scripts and measurements (no model reading)

| id   | item                                                                                               | status   | started     | ended | commit or file                         |
| ---- | -------------------------------------------------------------------------------------------------- | -------- | ----------- | ----- | -------------------------------------- |
| P0.1 | plan; the public-surface digest and `make api-surface-check`                                       | complete | 09:40 about | 09:44 | `b09e46da`                             |
| P0.2 | scans: duplicate names and bodies, helpers, constants, layering, casts, collisions, unused exports | complete | 09:44 about | 09:51 | `4e726d9a`, `d24f7be6` (shared module) |
| P0.3 | scan: type definitions written more than once; base shapes; spellings                              | complete | 09:51 about | 09:58 | `1bed39c4`, `28fc3f5e`                 |
| P0.4 | coverage of the core (engine suite, v8, per domain and file)                                       | complete | 09:47       | 10:03 | `a4a6c8fd`                             |
| P0.5 | public numeric signatures: native arrays against tensors, per domain                               | complete | 10:03       | 10:04 | `a1b37337`                             |
| P0.6 | CPU profile of representative examples: the hot paths                                              | planned  |             |       |                                        |

## Phase 1: targeted reading (read-only agents, findings in `findings/<cluster>.md`)

| id   | cluster                                                                                               | status      | started     | ended | commit or file            |
| ---- | ----------------------------------------------------------------------------------------------------- | ----------- | ----------- | ----- | ------------------------- |
| P1.1 | foundations: base, context, tensor, random, space, trace, autodiff (about 17,600 lines, read in full) | in progress | 10:15 about |       | `findings/foundations.md` |
| P1.2 | numerics, by subfolder, led by the flags                                                              | planned     |             |       |                           |
| P1.3 | optim and losses                                                                                      | planned     |             |       |                           |
| P1.4 | dists, stats, metrics                                                                                 | planned     |             |       |                           |
| P1.5 | kernels, data, exposition, scene                                                                      | planned     |             |       |                           |
| P1.6 | render (one cluster, same rubric)                                                                     | planned     |             |       |                           |

## Phase 2: synthesis

| id   | item                                                          | status  | started | ended | commit or file |
| ---- | ------------------------------------------------------------- | ------- | ------- | ----- | -------------- |
| P2.1 | one ranked list of findings, gone through with the maintainer | planned |         |       |                |

## Phase 3: fixes (small agents, at most two at a time; each merges after the full local gate and a surface comparison)

| id    | item                                                                                                                                                                                                                                             | status                       | started     | ended | commit or file                                                          |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- | ----------- | ----- | ----------------------------------------------------------------------- |
| P3.1  | fixed-size tuple types: `Vec2`, `Vec3`, `Vec4` in `base`; restated and synonym definitions removed; guard in `make lint`                                                                                                                         | complete                     | 10:04 about | 10:17 | `287b6c93`..`1581aa2c`, merged `44f5bd50`                               |
| P3.2  | the pytree leaf error names the path and says to wrap numbers with `tensor(...)`                                                                                                                                                                 | complete                     | 10:10 about | 10:17 | `b1c32597`                                                              |
| P3.3  | duplicate helpers into shared ones: `dot`, `matmul`, `transpose`, `nextAfter`, `softplus`, `logit`, `clip`, `neighbourSets`, `backcastOf`, `blank`, the `rowsOf`/`toRows`/`columnOf`/`matrixOf` coercions, repeated `EPS` constants              | planned                      |             |       | `reports/duplicates-strict.txt`                                         |
| P3.4  | the option-table helper (`option()` copied three times, `knobsOf` in five files) built once                                                                                                                                                      | planned                      |             |       |                                                                         |
| P3.5  | the SVM duals written twice (`optim/smo.ts` and `svm/_dual.ts`)                                                                                                                                                                                  | planned                      |             |       |                                                                         |
| P3.6  | layering: six core files importing a constructed domain; core imports against the order; a rank check in `make layers`                                                                                                                           | planned                      |             |       | `findings.md`, Layering                                                 |
| P3.7  | tests for the core gaps (`exposition/subgroups.ts` 0%, `tensor/ops/indexing.ts`, `tensor/format.ts`, `dists` structure, heavy-tailed, relaxed, `data/archives.ts`, `base/numbers.ts`, `autodiff/custom.ts`; branch coverage), net count reported | planned                      |             |       | `findings.md`, Coverage                                                 |
| P3.8  | native arrays in public numeric signatures to tensors, staged by domain (changes the public surface: the one deliberate break)                                                                                                                   | planned, after P1 sorts them |             |       | `findings.md`, Native arrays; `reports/tensor-migration-candidates.txt` |
| P3.9  | triage of 1,952 core exports no other source file uses (public surface, or dead)                                                                                                                                                                 | planned                      |             |       | `findings.md`, Public surface                                           |
| P3.10 | same-name-different-meaning types: `Matrix`, `Mat`, `Vec`, `ComplexNumber` (`{re, im}`), `Box`                                                                                                                                                   | planned                      |             |       | `log.md` 2026-11-07                                                     |
