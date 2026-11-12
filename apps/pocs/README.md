# omai pocs

Proofs of concept: single live pages of what the library can do, kept out of the site. `make pocs` serves them at
http://localhost:5310 (the site is `make dev`, port 5300). The shell, colours and controls are the site's own (the app
imports `apps/site/src/shell`), so a page looks the same as it would on the site.

The Bayesian examples' methods (`defineExample`, the per-example knobs, time budgets and views) are `src/examples/`: they compose
library models and inference methods for the pages and are not library code.

Inclusion only: a page exists because someone asked for it, not because the library has the feature. To add one, make
`src/pocs/<dir>/<slug>/poc.ts` (title, summary, tags, date: this is what the index searches) and `src/pocs/<dir>/<slug>/Page.tsx`
(the default export is the page); nothing else needs editing. Use the library's own estimators and slots (an
estimator's `.method(data)` run with `useSession`, its config as `SpaceControls`, panels from `SessionPanels`) so a
page stays a composition; anything with substance belongs in `packages/engine/src/lib`. A page that proves its worth can
be promoted to the site's gallery by moving its recipe there.

Directories: the pages are organised as a tree of real directories under `src/pocs/`, and the app works like a file
browser. A poc is a folder `src/pocs/<dir>/<subdir>/.../<slug>/` with `poc.ts` and `Page.tsx`; the slug is the poc's own
folder name, unique across the whole tree, and its address stays `/poc/<slug>` wherever it sits. A directory has a `dir.ts`
(`export const dir: PocDir = { title, summary, order }`: `order` sorts the directories of a level, ascending) and may hold
subdirectories and pocs to any depth; an empty directory with a `dir.ts` is shown (as a place for pages still to come).
The front page `/` is the root: its directories as tiles (title, summary, how many examples below it, how many folders)
and then its examples; `/dir/<a>/<b>` is a directory (`/group/<slug>` still opens a top-level one); the trail in the top bar
shows the path, each directory a link. A search is global and flat, each result with its directory path. The tree is built by
`buildTree` in `src/poc.ts` from the files `registry.ts` finds (`import.meta.glob('./pocs/**/poc.ts')`), with the pure
functions for listing, the trail and search, all tested in `src/poc.test.ts`. A page left directly under `src/pocs/<slug>/`
with a `group: '<dir>'` in its `poc.ts` (the old way) is still listed, in the top-level directory that names, and a note in
the registry test names it; such pages are moved into the tree with a move script (a history-keeping `mv`, the relative
imports rewritten) that is not part of the repository.

Pages that run several methods at once within a time budget the user chooses (the probabilistic-programming group) share
their plumbing in `src/shared/`: `useAnytimeRuns` drives each run as a session through the library's `runAnytimeAsync`
(slices that hand the thread back, so the page keeps painting; a snapshot of each run about once a second, read by the page's
`derive`), `RunBar` is the budget choice, the progress and a line per run, `CurveChart` draws curves and points in one box,
`useDeferred` computes heavy work after the first paint, and `smoke.ts` (`snapshotOf`) reads a run as the page would, for the
smoke test of a page's `compute.ts`. Any page that computes for longer than a moment (the EP programs, their references and calibrations, the atoms' grids and calibrations, the coverage checks) runs on one small contract, `shared/run.ts`: a task reports `done` of `total` with a stage and a line of detail, hands the thread back or computes in a worker (`shared/runWorker.ts`: `workerTask` on the page's side, `serveRun` in the poc's worker file), and `useRun` turns it into a state (queued, running, converged, done, stopped, failed) that never goes backwards; `RunProgress` is the timer, the bar (after 0.4 s), the status line and a Stop that keeps what was computed. An EP program's `infer` takes optional `EpHooks` (`shared/ep/sweeps.ts`: `onSweep`, `stopped`, `yieldToPage`) and steps its session sweep by sweep with the library's `runAnytimeAsync`, in the shown source. A page that would rather compute first and show afterwards (the probit lab and the small classical pages) runs its jobs in Web Workers through `shared/workerPool.ts` (`runPool` on the page's side, `serveJob` in the poc's `worker.ts`: one status line per job, a stop, results as plain data) and replays what the workers recorded on one playhead; the page computes and draws nothing while the budget is running. Each such page keeps its runs and records in `compute.ts` (with a small test) and its layout
in `Page.tsx`.

A new page's styles arrive with a restart: Tailwind scans for classes when `make pocs` starts, so stop it and start it
again after adding a page whose layout looks unstyled.
