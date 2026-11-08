# omai-render

> Figures for omai: ECharts option builders, a figure model, graph layouts, diagrams of structures, views of
> omai-engine exposition records, SVG, PNG, GIF and WebM export, and React bindings including the session layer (the
> hook, the player, the panels). This index lists the main things in `src`; it is short on purpose and grows as things
> are indexed. The source is the reference.

The data comes from omai-engine (a traced run, an exposition record, a structure, a live session): see the
`llms.txt` of omai-engine (`packages/engine/llms.txt` in the repository, `node_modules/omai-engine/llms.txt` once
installed).

## How to use this index

Look here first, then open the file an entry points to, then follow its imports. Each entry is one line: kind, name,
what it is, `path:line` (relative to the package root) and the import path. Entries are not complete and carry no
signatures: open the source for those. The kinds are the engine's (`function`, `structure`, `table`, ...); a React
component or hook is a `function` whose description says so. `index/index.json` has every entry for tools. The domains
below are the folders of `src`; `react` is large and its index is nested by its subfolders.

## Conventions you need

- Headless and React. Everything outside `src/react` is headless: it takes plain data and returns data (an
  `EChartsOption`, a `Scene`, an SVG string, bytes) and runs in Node. React lives only in `omai-render/react`, which
  has `react` and `react-dom` as peer dependencies (version 19); install them yourself. Rasterising SVG (`svgToPng`,
  `svgToPixels`) needs a browser canvas, and `encodeWebm` the browser's video encoder (`canEncodeVideo`).
- Import from an entry point, not a file: `import { figure } from 'omai-render/figure'`, `import { Chart } from
  'omai-render/react'`. Every folder under `src` is an import path (`omai-render/<domain>`); `omai-render` itself
  re-exports the headless domains. Inside the library, relative imports carry the `.ts` extension. Resolving packages to
  TypeScript source needs the `source` condition (`node --conditions=source`).
- A figure is a model plus a builder. `figure({ medium, width, aspect })` (`omai-render/figure`) gives the size,
  padding and type of a figure per medium (`page`, `blog`, `slide`, `thumbnail`); `gridSpec`, `withMarginals` and `bento`
  lay out panels; limit policies fix axes for figures that change as controls move. The builders in `omai-render/charts`
  (line, bar, histogram, heatmap, fields, ...) return an `EChartsOption` or layers that combine on shared axes. Graph
  layouts (`omai-render/layout`) return positions only; `omai-render/structures` draws an engine `Structure` as a scene.
- Export. `chartToSvg(option, { width, height })` and `sceneToSvg(scene)` give SVG text (`omai-render/svg`);
  `composeSvg(width, height, parts)` puts charts and labels into one document; `encodePng(pixels)` gives PNG bytes
  from RGBA pixels and `svgToPng(svg, width, height)` a PNG data URL (browser only); `animatedSvg(timeline)` gives a
  self-playing SVG, and `timelineFrames(timeline)` gives pixel frames for `encodeGif(frames)` and `encodeWebm(frames)`
  (`omai-render/animation`). In the browser, `FigureFrame` offers a menu of exports on a figure (png and svg by
  default; gif and webm for a figure with a `playback`; `mp4` and `json` only through your own `onExport` handler).
- From a trace to a figure. A traced engine `Run` goes to `omai-render/trace` (`metricChart(run, ['loss'])`, tensor
  trajectories, episode returns, optimisation paths). A result with `exposition` records goes to `omai-render/views`:
  `viewFor(kind)` is the registry of standard views, `stillOf(record)` gives the chart option or scene that explains a
  record, and the views of one record kind also give player frames. Records come from `omai-engine/exposition`.
- The session layer (React). `useSession(method, options, deps)` and `useSessions` (variants of one method in step) run
  an engine `Method` live under a producer and give a `LiveSession`; `Player` and `SessionPlayer` are the playback
  controls over it; `SessionPanels` draws a layout of panels over the tape, with `RowsPanel` and the record panels
  (`StripPanel`, `TablePanel`, `ImagesPanel`, ...) beside it; `ExpositionView` picks the standard view of any record. `omai-render/producer` is the headless side: main-thread and Web Worker producers, the
  method registry (`methodRegistry`) and the tape as a file (`tapeOf`, `parseTape`, `replaySessions`); the React hook
  `useTape(file, registry)` replays one.
- Start policies (React). `start` on `useSession`'s hook options (and `useSessions`, `useLiveRun` and `SessionPlayer`)
  says when a session computes and where its cursor starts. One line each, with when to use it:
  - `start: 'play'` (live, the default): nothing is computed until play, then the run is computed ahead of the cursor.
    Use it when exploring, or when the run may be long or endless.
  - `start: 'first'` (primed): the whole run is computed in the background (in slices of each animation frame, or in
    a worker) and the cursor is parked at step 0, paused, until play. Use it for slides and demos where the animation
    should be ready the moment you press play.
  - `start: 'end'` (eager): the whole run is computed in the background and the cursor jumps to its end when it is done.
    Use it for a short run where you want the finished result at once. An engine `mode: 'eager'` in the options means
    `'end'` unless `start` is given.

  The handle shows how far a session is: `live.status` is `'preparing'` (the tape is incomplete, nothing plays),
  `'ready'` (the whole tape exists, nothing plays) or `'running'` (playing; a play pressed before the tape is ready
  plays at the head, as live does), `live.progress` runs from 0 to 1 over the options' `steps`, and the hook option
  `onReady` is called once the whole tape exists. A host can start the player with `live.play()` and stop it with
  `live.pause()`. `live.dispose()` stops the producer, terminates a worker and drops the tape (unmounting does this
  by itself). `useLiveRun(algorithm, { maxSteps, start: 'first', history })` is the same for an algorithm; its
  `history` is a ring buffer of that many states (stepping back stops at `oldest`), and it returns `phase` (the
  status) and `progress`.
- Slides and hidden pages (React). Four things on the handle serve a host that shows and hides sessions:
  - `active` (hook option of `useSession`, `useSessions` and `useLiveRun`, default true): while false nothing computes
    (no priming, the producer stopped, a worker paused, the player's clock off) and the tape and the state are kept; it
    carries on when true again. `live.halt()` and `live.pause()` are the imperative forms; `dispose()` throws the tape
    away, and is safe to call from a frame that was already queued.
  - `live.disposed` is true after `dispose()` until `compute`, `play`, `advance` or `apply` uses the session again. The
    `status` then reads `'preparing'` (a session at step 0), which is not a restart; check `disposed` to tell them apart.
  - `live.madeWith` is `{ config, seed }` of the run on the tape (`live.session.config` and `.seed`). A change of the
    options is applied after 250 ms, or held until play, so until then the options are newer than the run: draw what the
    run was made with from `madeWith`.
  - `live.progress` is the step reached over the options' `steps` while the run goes on, and exactly 1 once it has
    ended, whatever its length (a run that converges after 8 of 1000 steps jumps to 1 on its last step).
- Player and panel settings (React). `SessionPlayer` takes `fps` (the rate; with `onFpsChange` the host owns it),
  `eagerToggle={false}` (hide the Eager button) and `statistics={false}` (hide the line of event counts and slice
  time). Its starting rate and mode come from the session preferences: a `SessionPreferencesProvider` with `defaults`
  (`{ mode, hold, fps }`) and `storage` (`false` for none, or a `Storage` of the host's; the browser's `localStorage`
  under the key `omai-session-preferences` by default) gives a subtree its own, read by `useSessionPreferences` and
  changed by `useSetSessionPreferences`. The size of every panel (width, height, type size, padding) is the figure of a
  `PanelFigureProvider figure={figure({ width: 640, height: 400, fontSize: 14 })}` above them, or the `figure` prop of
  `SessionPanels`; the default is 440 by 320.
- States that an algorithm changes in place (React and engine). What a view draws at a step comes from what was
  recorded at that step (`observe`, or `ctx.log` and `ctx.emit` at emit time), or from `session.stateAt(t)`, which is
  exact through checkpoints; `session.state` is the live object. An algorithm that mutates its state in place gives
  an optional `snapshot(state)`, and `useLiveRun` keeps each frame of its history through it, so every frame shows its
  own step. States need not be immutable.
- Controls and interaction (React). Form controls, controls generated from a `Space` and a dataset picker are in
  `react/controls`; draggable points, brushes and hover overlays are in `react/interaction`; `Chart`, `SceneView`,
  `StructureView` and `FigureGrid` are the base components in `react`.
- Look and feel: `omai-render/theme` has the light and dark tokens, palettes and number formatting; `omai-render/maths`
  typesets TeX through KaTeX.

## Where to look

1. This index: find the domain, then the entry.
2. The file at `path:line`: its opening paragraph says what it is; each export has TSDoc with an `@example`.
3. Its imports, for the pieces it composes. A domain folder's `index.ts` opens with a summary of the folder.
