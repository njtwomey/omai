# What the ml-concepts notes use of the AIFN engine, and what omai lacks

omai is meant to replace the AIFN engine (`aifn-compute`, `aifn-methods`, `aifn-render`) under the ml-concepts notes.
This folder is a survey of that move: which capabilities the notes actually import, which of them omai has (under the
same name, another name or a differently shaped API), which it does not have, and what the notes are likely to want
next. It was made by a few small scripts, so every number can be regenerated.

| file         | what it holds                                                                                                                             |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `usage.md`   | What the notes use, by area, with the share omai covers                                                                                   |
| `gaps.md`    | The gaps the existing notes hit now, grouped, with a home, a size and a priority; first the design differences that cost the most to port |
| `future.md`  | Likely future gaps: planned notes, the engine requests the note authors wrote, AIFN surface nothing uses yet, render features             |
| `mapping.md` | The renames, the prefix change and the module map, for anyone porting a note                                                              |
| `data/`      | The machine-readable output (JSON, each file under 1 MB) and the generated tables                                                         |

The convolutional sparse coding and dictionary-learning module (and the synthetic ECG beat of the same hand-over) is
being ported by another agent. It is counted as in progress, not as a gap. omai already has the SciPy ECG as
`electrocardiogram` in `omai-engine/datasets`.

## What was compared

- **The notes**: `/Users/nialltwomey/workspace/ml-concepts` at commit `3afae55b`, which pins the AIFN packages at 0.2.10.
  2486 notes (2126 draft, 360 stubs), 4180 `.mdx`/`.tsx`/`.ts` files under `content/notes`, plus `lab/`, `site/src`,
  `scripts/` and `plugins/`.
- **The AIFN engine**: `/Users/nialltwomey/workspace/aifn-engine` at commit `91eeaf5` (the 0.2.10 tag). 301 entry points,
  8421 exported declarations (4830 in compute, 3131 in methods, 460 in render), read from the TypeScript source.
- **omai**: this repository at `fb40b185`, read the way `scripts/api-surface.ts` reads it: 17968 exports at their
  canonical home, with the modules that re-export them.

## The scripts

All in `scripts/aifn-usage/`, TypeScript run with `node` (the repo's type stripping), no network, no writes outside
`docs/aifn/`. ml-concepts and aifn-engine are only read.

| script          | what it does                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | writes                                                                                                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scan-usage.ts` | Reads every `.mdx`, `.tsx` and `.ts` file for static imports and re-exports (named, default, namespace with its `ns.member` uses, side-effect), dynamic `import()` and the worker calls `call('module/export')`. A note is a folder with an `index.mdx`; it uses what its own files import and what the files they import (relatively, or through the site's `@/` alias) import in turn, so code in `_shared/` counts for every note that reaches it. Counts notes, files, lab files and gives a few example notes per symbol | `data/usage.json`                                                                                                                                                                |
| `scan-aifn.ts`  | Lists the AIFN import paths (`./*` of compute and methods onto `src/*/index.ts`, the explicit paths of render) and the symbols each exports, as the TypeScript checker resolves them (`export *` followed). Each declaration is recorded once, at the entry point that holds its file, with where else it is exported                                                                                                                                                                                                         | `data/aifn-surface-{compute,methods,render}.json`                                                                                                                                |
| `scan-omai.ts`  | Reads omai's public surface with the repo's docs extractor (`packages/meta`). A module used by the other scripts; it prints counts when run                                                                                                                                                                                                                                                                                                                                                                                   | nothing                                                                                                                                                                          |
| `scan-wants.ts` | Reads what the notes want next from their non-code content: note frontmatter (stubs), `.scratch/implementations/GAPS.md` (the engine needs of the unwritten notes), the engine requests and API sketches in `.scratch`, comments that say what the engine lacks, and the numerics the notes wrote for themselves                                                                                                                                                                                                              | `data/wants.json`                                                                                                                                                                |
| `match.ts`      | Runs the three scans in memory, gives every AIFN declaration a status against omai, folds the unmatched ones into capabilities (`gap-groups.json`), and writes the tables                                                                                                                                                                                                                                                                                                                                                     | `data/match.json`, `data/gap-groups.json`, `data/unused-unmatched.json`, `data/tables.md`, `data/aliases.md`, and the generated blocks of `usage.md`, `gaps.md` and `mapping.md` |

The two hand-kept inputs are `scripts/aifn-usage/aliases.json` (the alias map, a reason for every entry) and
`scripts/aifn-usage/gap-groups.json` (which unmatched symbols form which capability).

### Rerun

```sh
# once, if node_modules is missing
mise exec -- pnpm install --frozen-lockfile

# everything: the three scans run inside match.ts, which rewrites match.json, the tables and the generated blocks
nice -n 15 mise exec -- node scripts/aifn-usage/match.ts

# the committed per-scan outputs
nice -n 15 mise exec -- node scripts/aifn-usage/scan-usage.ts
nice -n 15 mise exec -- node scripts/aifn-usage/scan-aifn.ts
nice -n 15 mise exec -- node scripts/aifn-usage/scan-wants.ts

# a review aid: the nearest omai names of each unmatched used symbol under a prefix
nice -n 15 mise exec -- node scripts/aifn-usage/match.ts --suggest aifn-compute/nn
```

Each run takes a few seconds. The paths default to `~/workspace/ml-concepts` and `~/workspace/aifn-engine`; set
`ML_CONCEPTS` and `AIFN_ENGINE` to use others. The output is sorted and carries no timestamps, so a rerun on the same
inputs writes the same bytes. The generated blocks in the docs sit between `<!-- generated:name -->` and
`<!-- /generated:name -->` markers; the prose around them is hand-written.

## How a symbol is matched

Each AIFN declaration gets one status, in this order of precedence:
1. `internal`: not a capability to port (a namespace re-export, a helper of the AIFN primitive machinery, a UI kit
   re-export).
2. `none`: the same name exists in omai but is a different thing, so the AIFN symbol has no counterpart.
3. `design`: covered by a differently shaped omai API (the entry names it).
4. `alias`: a hand-kept entry names the omai symbol: renamed, moved, or the same capability under another call shape.
5. `exact`: omai exports the same name. Where `aliases.json` has a module map for the AIFN entry point, the match must be
   in one of the mapped omai modules; otherwise the name must exist in exactly one omai module.
6. `type`: a type or interface with no same-named omai type. A type follows the value it describes; not counted as a gap.
7. `elsewhere`: the name exists only in omai modules the map does not expect. Needs a look; never counted as covered.
8. `unmatched`: nothing found. These are the gap candidates.

## What to trust

- **`exact` is good because both libraries copy sklearn, scipy and torch names.** The exact value matches
  were skimmed side by side (the AIFN summary against the omai summary); two were wrong (`conditionNumber`, which is a
  scalar function's condition in omai, and `breadthFirstSearch`, which is the search-problem version) and are fixed in
  `aliases.json`. The module map caught a dozen more same-name coincidences (`stream` of the data domain, `fromRows`
  of worlds, `Histogram` of stats, `normalCdf` as a bijector and so on).
- **`alias` and `design` entries are judgments.** Each was made by reading the AIFN summary and the omai index or source.
  Where the omai symbol is only close (a different shape, fewer options) the reason says so. A few are looser than
  "same capability": `regression1d` (one generator taking `f` and sigma(x)) is mapped to several fixed omai generators.
- **`unmatched` used symbols were reviewed family by family.** The list was shortened by repeated rounds of: read the
  unmatched symbols of an area, look for the capability in omai by name, by words and in the source, add an alias or a
  design entry, rerun. The symbols used by two notes or more were each looked at. Those used by one note (337 of the
  393) were looked at by family and by a nearest-name search, not each by reading code, so a few may still exist in omai
  under a name nobody tried. A true count of gaps is therefore an upper bound on symbols and closer to exact on
  capabilities.
- **The AIFN symbols no note uses were not reviewed.** `data/unused-unmatched.json` lists 2200 of them with no
  counterpart by name; most are internals (registries such as `*Algorithms`), helpers and renames. It is a place to
  look, not a gap list: a rough reading of a sample of 60 found mostly internals, registries and renames, and a few real
  capabilities.
- **Counts of notes follow the import closure**, not only the files of a note, so a helper in `_shared/` counts for each
  note that reaches it. A note that imports a widget that imports the engine uses the engine.
- **Lab is not the notes.** `lab/` (an explorer of the engine) imports 1113 declarations that no note uses; they are kept
  apart and only feed `future.md`.
- **Types** are matched but not scored: 279 of the 1543 used declarations are types.

## Known limits

- Notes that use no aifn symbol (1212 of 2486) are not "ported": most are prose and static maths; some have figures
  written with local code. `data/usage.json` lists the local code the notes wrote for themselves (`localFiles`).
- A symbol imported through a namespace (`import * as x`) is counted by the members used as `x.name`; two dynamic
  imports (`aifn-methods/data/real/ecg` and `.../fonts`) count the entry point only.
- `render` is a different design in omai (functions that build ECharts options and scenes, drawn by `Chart`, against
  AIFN's React layer components). Those symbols are mapped as `design`, which says they have an equivalent, not that a
  note can swap an import. `gaps.md` starts with what that costs.
- The `.scratch/implementations/GAPS.md` list is from 2026-10-17 and was written against AIFN 0.2.9; some of its needs
  were built in 0.2.10. `future.md` checks the names against 0.2.10 and omai but only by name and by shared words.
