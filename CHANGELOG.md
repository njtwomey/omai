# Changelog

Versions are released in lockstep for mdml-engine, mdml-render and mdml-meta.

## 0.1.2

- mdml-render: a third start policy for sessions, `start: 'first'` (primed): `useSession`, `useSessions`,
  `useLiveRun` and `SessionPlayer` compute the whole run in the background and park the cursor at step 0, paused,
  until play. `start: 'play'` is today's live (the default) and `start: 'end'` today's eager.
- mdml-render: eager no longer blocks the page. An engine `mode: 'eager'` (and `start: 'end'`) is computed in
  animation-frame slices by the hook, and the cursor jumps to the end when the run is done, instead of computing the
  whole run synchronously when the session is made.
- mdml-render: the live handle has `status` (`preparing`, `ready`, `running`), `progress` (0 to 1), `playing`, `play()`,
  `pause()`, `dispose()` and an `onReady` hook option; `dispose` stops the producer, terminates the worker and
  releases the tape, and unmounting calls it. `useLiveRun` returns `phase`, `progress`, `oldest` and `dispose`, takes
  `start` and `onReady`, and keeps its history in a ring buffer.
- mdml-render: the player's clock no longer slows when drawing is slow (it skips steps instead), and
  `SessionPlayer` takes a `playhead` ref that holds the step the clock has reached.
- mdml-render: hand-made handles need `...STILL_HANDLE` for the new members.
- mdml-engine: the linear-programming solver no longer ends on a singular basis for programs where every ratio ties (a
  nearly tied game's correlated equilibrium): a pivot must be large beside its column (`pivotFloor`, default 1e-7),
  `linprog` retries without that floor before the interior point, and a simplex that reaches its default iteration limit
  falls back too. Measured on a nearly-tied generator: 4 failures in 120,000 solves before, 2 in 240,000 after.
- Tests: five flaky property tests fixed (autocorrelation of denormals, the ray cast at a world's boundary, NMS
  thresholds, the spline's finite-difference step, and the text round trip of a negated sum among them); each was a
  false or too-tight claim, not a library bug.

## 0.1.1

- The agent index covers mdml-render as well as mdml-engine: `packages/render/llms.txt` (with a hand-written
  `llms.header.md`), `packages/render/index/` and `index.json`, published with the package. 161 render entries are
  tagged (67 in `react/`, 94 elsewhere), each with a short call to copy, for example `gridSpec(2, 3)`, `bento(...)` or
  `<SessionPlayer live={live} at={at} onScrub={setAt} />`.
- A generated `llms.txt` at the repository root maps the packages (mdml-engine, mdml-render, and mdml-meta as not
  indexed).
- The collector (`mdml-meta` extract, `scripts/agent-index.ts`) indexes several packages: outputs per package, a check
  of stale files per package and of the root file, the same tag vocabulary for all.
- The headers of the engine's and render's `llms.txt` refer to each other.
- mdml-render: removed exports that nothing in the repository used or that the session layer replaced: `EnvView`,
  `EnvViewProps`, `useEnvStepper`, `EnvStepper`, `EpisodePlayer`, `EpisodePlayerProps`, `MethodControls`,
  `MethodControlsProps`, `SessionControls`, `SessionSettings`, `initialSettings`, `logsOf`, `PlaybackSource`,
  `LiveOptions`, `precomputedSource`, `liveSource`, `EpisodeStep`, `episodeSteps`, `stepCaption`, `episodeTimeline`,
  `SunIcon`, `MoonIcon`, `MonitorIcon`, `MenuIcon`, `SceneFigure`, `bestTrialText` and `isProbePending`.
- mdml-render: the player's buttons are grouped (the run, the playback, the frames).
- mdml-engine: the planar flow's inverse works when `w . u_hat` exceeds 1 (the bisection bracket was too narrow).

## 0.1.0

- First release. mdml-engine and mdml-render were released as separate GitHub releases.
