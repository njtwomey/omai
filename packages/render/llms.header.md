# mdml-render

> Figures for mdml-js: ECharts option builders, a figure model, graph layouts, diagrams of structures, views of
> mdml-engine exposition records, SVG, PNG, GIF and WebM export, and React bindings including the session layer (the
> hook, the player, the panels). This index lists the main things in `src`; it is short on purpose and grows as things
> are indexed. The source is the reference.

The data comes from mdml-engine (a traced run, an exposition record, a structure, a live session): see the
`llms.txt` of mdml-engine (`packages/engine/llms.txt` in the repository, `node_modules/mdml-engine/llms.txt` once
installed).

## How to use this index

Look here first, then open the file an entry points to, then follow its imports. Each entry is one line: kind, name,
what it is, `path:line` (relative to the package root) and the import path. Entries are not complete and carry no
signatures: open the source for those. The kinds are the engine's (`function`, `structure`, `table`, ...); a React
component or hook is a `function` whose description says so. `index/index.json` has every entry for tools. The domains
below are the folders of `src`; `react` is large and its index is nested by its subfolders.

## Conventions you need

- Headless and React. Everything outside `src/react` is headless: it takes plain data and returns data (an
  `EChartsOption`, a `Scene`, an SVG string, bytes) and runs in Node. React lives only in `mdml-render/react`, which
  has `react` and `react-dom` as peer dependencies (version 19); install them yourself. Rasterising SVG (`svgToPng`,
  `svgToPixels`) needs a browser canvas, and `encodeWebm` the browser's video encoder (`canEncodeVideo`).
- Import from an entry point, not a file: `import { figure } from 'mdml-render/figure'`, `import { Chart } from
  'mdml-render/react'`. Every folder under `src` is an import path (`mdml-render/<domain>`); `mdml-render` itself
  re-exports the headless domains. Inside the library, relative imports carry the `.ts` extension. Resolving packages to
  TypeScript source needs the `source` condition (`node --conditions=source`).
- A figure is a model plus a builder. `figure({ medium, width, aspect })` (`mdml-render/figure`) gives the size,
  padding and type of a figure per medium (`page`, `blog`, `slide`, `thumbnail`); `gridSpec`, `withMarginals` and `bento`
  lay out panels; limit policies fix axes for figures that change as controls move. The builders in `mdml-render/charts`
  (line, bar, histogram, heatmap, fields, ...) return an `EChartsOption` or layers that combine on shared axes. Graph
  layouts (`mdml-render/layout`) return positions only; `mdml-render/structures` draws an engine `Structure` as a scene.
- Export. `chartToSvg(option, { width, height })` and `sceneToSvg(scene)` give SVG text (`mdml-render/svg`);
  `composeSvg(width, height, parts)` puts charts and labels into one document; `encodePng(pixels)` gives PNG bytes
  from RGBA pixels and `svgToPng(svg, width, height)` a PNG data URL (browser only); `animatedSvg(timeline)` gives a
  self-playing SVG, and `timelineFrames(timeline)` gives pixel frames for `encodeGif(frames)` and `encodeWebm(frames)`
  (`mdml-render/animation`). In the browser, `FigureFrame` offers a menu of exports on a figure (png and svg by
  default; gif and webm for a figure with a `playback`; `mp4` and `json` only through your own `onExport` handler).
- From a trace to a figure. A traced engine `Run` goes to `mdml-render/trace` (`metricChart(run, ['loss'])`, tensor
  trajectories, episode returns, optimisation paths). A result with `exposition` records goes to `mdml-render/views`:
  `viewFor(kind)` is the registry of standard views, `stillOf(record)` gives the chart option or scene that explains a
  record, and the views of one record kind also give player frames. Records come from `mdml-engine/exposition`.
- The session layer (React). `useSession(method, options, deps)` and `useSessions` (variants of one method in step) run
  an engine `Method` live under a producer and give a `LiveSession`; `Player` and `SessionPlayer` are the playback
  controls over it; `SessionPanels` draws a layout of panels over the tape, with `RowsPanel` and the record panels
  (`StripPanel`, `TablePanel`, `ImagesPanel`, ...) beside it; `ExpositionView` picks the standard view of any record. `mdml-render/producer` is the headless side: main-thread and Web Worker producers, the
  method registry (`methodRegistry`) and the tape as a file (`tapeOf`, `parseTape`, `replaySessions`); the React hook
  `useTape(file, registry)` replays one.
- Controls and interaction (React). Form controls, controls generated from a `Space` and a dataset picker are in
  `react/controls`; draggable points, brushes and hover overlays are in `react/interaction`; `Chart`, `SceneView`,
  `StructureView` and `FigureGrid` are the base components in `react`.
- Look and feel: `mdml-render/theme` has the light and dark tokens, palettes and number formatting; `mdml-render/maths`
  typesets TeX through KaTeX.

## Where to look

1. This index: find the domain, then the entry.
2. The file at `path:line`: its opening paragraph says what it is; each export has TSDoc with an `@example`.
3. Its imports, for the pieces it composes. A domain folder's `index.ts` opens with a summary of the folder.
