# Changelog

Versions are released in lockstep for mdml-engine, mdml-render and mdml-meta.

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
