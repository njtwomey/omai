# Render review: the session layer, the record views, the controls, the player and the export path

Read-only review of `packages/render/src` (254 non-test files, about 56,000 lines, 13 folders). All paths below are relative to `packages/render/src/`. Probes cited as
"probe" are throwaway node scripts in
`/private/tmp/claude-501/-Users-nialltwomey-workspace-whitebox-js/d4034594-54d3-484e-a26a-331816a1f3f5/scratchpad/review/probe4/`
(`v1.ts`/`v2.ts` malformed and degenerate records through `stillOf`, `p1.ts` the curves option per playhead tick, `p2.ts` and `p3.ts` the decision-surface player frame and its
parts, `prof.mjs` a CPU profile summary, `p4.ts` spread limits, `p5.ts` colour-blind check, `unused.mjs` exports nothing uses), run one at a time at `nice -n 15` with
`node --conditions=source` against the source; nothing in the repository was changed and no test, tsc, dev server or gate was run. Coverage figures are Phase 0's
(`coverage-final.json`, render's own tests, CI mode). Line numbers are from the tree at commit 727d3bcb. About 50 source files were opened (some in part). No browser was
available: what needs one is marked "needs a browser".

## Summary

1. Healthy: **the structure is sound and the rules hold.** Only `react/` imports React (checked over all 254 files); the engine is imported only through public entry points
   (`mdml-engine/exposition` 210 imports, `scene` 84, `trace` 48, ...); the 10 `// allow:` clock opt-outs (7 in `useSession.ts`, 2 in `usePlayback.ts`, 1 in `producer/serve.ts`)
   are each a frame budget and none reaches the tape; no `Math.random`, `console` or `Date.now`. `viewFor` is a mapped type over `ExpositionKind` (61 kinds, 61 components, all
   exist), the record views are thin wrappers on one `_frame.tsx` (`FramesPlayer`, `RecordFigure`), the session core is well covered and written for strict mode and unmount
   (`useSession.ts` 85%, `SessionPanels.tsx` 83%, `producer/client.ts` 89%; disposal after a microtask, bounded probe cache, sequence-numbered worker events, timers cleared),
   the headless builders are cheap (a 12-series, 12,000-point curves option builds in 1.3 ms, `p1`), the interaction handles are keyboard-operable, and `SettingsButton` is a
   correct dialog (focus moved, trapped, restored).
2. Not healthy, 1: **a decision-surface player frame costs 30 ms to build** (64x64 grid, 3 classes, 200x200 image, `p2`/`p3`): three bicubic resamples at 6.2 ms and a PNG
   data URL (4 to 5 ms of JavaScript deflate) that the canvas renderer then throws away. The player cannot reach 60 fps on the site's main classifier animation before ECharts
   has done anything (F1).
3. Not healthy, 2: **records are not validated at the view, and nothing catches the failure.** 59 of 61 kinds throw a bare `TypeError` deep in a builder for a record missing its
   fields, an unknown kind throws `Cannot read properties of undefined (reading 'still')` from `stillOf` and renders nothing from `ExpositionView`, there is no error boundary in
   the package, and `isExposition` checks only `kind`. A tape file chosen from disk reaches the views this way (F2, F4).
4. Not healthy, 3: **the export and tape path has no test and two defects**: `SessionExport` records a GIF/WebM without pausing playback (F3), a tape whose method the page does not
   register crashes the page at render (F4), and downloads are written four times, two of them leaking an object URL (F7).
5. Not healthy, 4: **accessibility is uneven.** The player is fully labelled, but its root `onKeyDown` takes Space, arrows, Home and End from any focused button and from the
   fps popover portal (F6); the chart has no text alternative (the `aria` option is set by 11 of 46 view files and by no chart builder, F9); the expanded-figure dialog and the
   popovers have no focus management (F10); the default palette is not colour-blind safe past four colours and nothing but colour separates variants (F15); there is no
   reduced-motion handling and two views autoplay on mount (F16).
6. Coverage: the `react` layer is the gap and `react/views` is worse than the aggregate: **238 of 981 statements (24%)** over 44 files, `ExpositionView` 14%, `views/registry.ts`
   36 of 64 functions (28 `still` closures never run). A table-driven smoke over one minimal record per kind closes most of it for +3 tests (F5).
7. Performance by reading (needs a browser to measure): the live handle is rebuilt with new function identities on every render and there is no `React.memo` in the package, so
   every panel re-renders on every playhead tick (F12); the curves panel rebuilds and replaces the whole series for the step marker (F14); inner sessions and probes compute
   synchronously in render once per step (F13).
8. Public surface: 1,154 exports in non-test source, 300 used by nothing outside their own file and the tests (158 interfaces, 21 types, 115 functions, 6 constants), 37
   functions and constants by nothing at all. The register's candidates (trace chart builders, animation timelines) are real API: documented, tested, indexed, only without a
   consumer in the repository (F19). `producer/_fake.ts` ships in `dist` (F24).
9. The three highest-value fixes: (a) **F1** (lazy PNG, separable bicubic: 2 to 3x on the main player, behaviour kept); (b) **F2 + F4 + F5** (an unknown-kind message, a small
   `RecordBoundary`, the registry lookup before a tape is accepted, and the table-driven smoke test); (c) **F6 + F9 + F10** (stop the player taking keys from its own buttons, an
   `aria-label`/`role="img"` on `Chart`, and reuse of `SettingsButton`'s dialog logic in `FigureFrame` and `Popover`).
10. Not read or not verified: `charts/` beyond `fields.ts` and `matrices.ts`, `layout/`, `structures/`, `gym/`, `maths/`, `figure/`, most of `views/*.ts` (read: `registry`, `_view`,
    `margins`), most of `react/views`, `react/session/{layout,families,overlays,export,TapeView,RecordPanels,SpecialisedPanels}`, `producer/{serve,protocol,step,make}`. Event-default
    behaviour, focus, screen readers, ECharts' own cost per `setOption`, re-render counts and colour vision were not observed (no browser, no React profiler).

Entries by kind: hot-path 6, bug 3, consistency 4, duplicate 3, test-gap 2, a11y 6, api 1, rule 2, casts 1, native-array 1 (29 in all, F1 to F29).

---

### F1: a decision-surface frame costs 30 ms to build, and 4 to 5 ms of it is a PNG the browser never uses

kind: hot-path
evidence: probe `p2` (64x64 grid, 3 classes, 40 recorded steps, 300 rows): `decisionSurfaceFrames(record).option(7)` takes **30.2 ms** and returns a **2.03 MB** option; ten distinct frames 305 ms. CPU profile
(`prof.mjs`): `charts/_resample.ts:16-18` (`resample` and its `at` closure) 50% of self time, `svg/_deflate.ts:90` `deflateRaw` 9%, `svg/png.ts:39` `filtered` 8%, `decisionRegions`
(`charts/fields.ts:196`) 9%. `p3`: `resample(64x64 -> 200x200, 'bicubic')` 6.2 ms per class (`fields.ts:208`, `grids.map((g) => resample(g, rows, cols, 'bicubic'))`), bilinear 1.0 ms;
`pngDataUrl` of a 200x200 image 4.0 to 5.4 ms. `charts/matrices.ts:121` builds `imageRenderer(pngDataUrl(pixels), ...)` for every image series, and `react/Chart.tsx:41-45` then replaces it
with a canvas through `reencodeImages(series, canvasImage)` for the canvas renderer (the default), so the data URL is encoded for nothing on every frame the page draws.
what: the frame budget at 60 fps is 16.7 ms, ECharts' `setOption` comes on top (not measurable here), and `FramesPlayer` builds one such option per displayed frame, so the player of the library's flagship classifier view runs below display rate for any grid of this size. Every
image-based view (`decisionRegions`, `heatmap` smoothing, `contours` fills) shares the two costs.
fix: (a) make the data URL lazy: `imageRenderer` takes `() => string` and `imageSeries` passes `() => pngDataUrl(pixels)` (memoised), so only the SVG renderer and export pay for it (`reencodeImages` already swaps the encoding at draw time); (b) a separable bicubic (rows then columns, the `lo`/`hi` clamp range computed once per grid and the clamped index lookups hoisted out of the inner loop), which is about 4 cheap passes instead of 16 clamped taps per output pixel; (c) optionally resample once per class per `frames` object, not per frame, when the grid is unchanged (the recorded `steps` change it, so this only helps views without steps).
size: S (a, b), M (c)
risk: low (pixel output of (b) is the same function evaluated in another order: compare the golden option, tolerance 1e-12)
preserves behaviour: yes (the option's `renderItem` and image are equal when drawn)
changes public surface: no

### F2: views do not validate records and nothing catches a view that throws: bare records crash deep, unknown kinds crash or draw nothing

kind: consistency
evidence: probe `v1` calls `stillOf({ kind })` for all 61 kinds on a record with only its `kind`: **59 throw a `TypeError`** (`Cannot read properties of undefined (reading 'length' | 'map' | 'filter' | '0' | 'flatMap' | 'entries' | 're' | 'width')`; `complexPlane`, `equationFit` and `dataBias` `... is not iterable`), `pruningPath` an `MdmlError` from a tensor conversion, `conformal` returns a chart. With arrays present but empty, 51 still throw `TypeError`, 4 throw `MdmlError`s from deep builders (`figure: bad aspect 'NaN'`, `bars: give one value per category`, `ragged nested array`) and 6 draw. An unknown kind: `stillOf({ kind: 'nope' })` throws `TypeError: Cannot read properties of undefined (reading 'still')` (`views/registry.ts:664`, `viewFor` is `REGISTRY[kind] as StandardView<K>` with no check), and `react/views/ExpositionView.tsx:84-215` is a `switch` with no `default`, so an unknown kind returns `undefined` and renders nothing, silently. `grep` finds no error boundary in `packages/render` (the site has one in `apps/site/src/shell/ErrorBoundary.tsx`, which blanks the whole recipe). The records that can be malformed reach the views by `ProbeView` and `ProjectionView` (`react/session/SessionPanels.tsx:703`, `:749`, both guard with `isExposition`, which checks only `kind`: the engine review's F9) and by a tape file read from disk (F4). In-process records are guarded by the engine constructors, which validate lengths (probe `v2`: the constructors reject what the views cannot draw).
what: the engine review found records are not validated at construction beyond lengths and that `isExposition` trusts a list of kinds; the view is the next place a bad record shows, and it fails with an engine-internal message, in render, taking its subtree (React 19 unmounts a root with an uncaught render error) with it. A newer engine's kind in a tape also falls through the switch unannounced.
fix: (1) `viewFor` throws `MdmlError('value', "no standard view for record kind 'x' (known: ...)")` for an unknown kind, `ExpositionView` renders a one-line note with the kind for it (and `stillOf` the same error); (2) a small `RecordBoundary` class component in `react/views/_frame.tsx` (`getDerivedStateFromError`, keyed on `record`) that `ExpositionView` wraps its view in and that shows "this record could not be drawn: <message>" inside the figure's frame; (3) do not add per-field validation to 61 views: validation belongs to the constructors, and a deep `isExposition` (engine F9) is the one place to tighten for records from outside.
size: S (1, 2), M (3 with the engine side)
risk: low
preserves behaviour: yes for valid records
changes public surface: yes (one new export: `RecordBoundary`)

### F3: `SessionExport` records a GIF or WebM without pausing playback, cannot be cancelled and has no test

kind: bug
evidence: `react/session/SessionExport.tsx:123-160` (`record`): for each step it awaits probes, calls `onScrub(step)`, waits `nextPaint()` and rasterises; there is no `live.pause()` anywhere in the file (`grep -n "pause\|playing"` finds nothing) while `SessionPlayer`'s ticker (`SessionPlayer.tsx:117-135`) calls `onScrub` every frame for as long as the page plays, so a recording started during playback races the clock and takes frames of wherever the ticker has put the cursor. The sibling `FigureFrame.recordFrames` does pause first (`FigureFrame.tsx:139`). The loop has no abort: unmounting mid-recording keeps calling `onScrub` and `setBusy` for up to 200 frames, and the `finally` restores `onScrub(previous)` after the page may be gone. The file is 119 statements at **1.7%** (`record`, `saveTape`, `loadTape`, `capture` never run; coverage shows 33 of 33 functions unrun).
what: a user-visible defect in the most elaborate export, in the file nothing tests. `previous = at` is restored but a playing page then continues from where the recorder left it.
fix: call `live.pause()` (the handle has it) before the loop and restore playing after if it was; check an `AbortController`/ref set by an unmount effect each iteration; one harness test (see F18).
size: S
risk: low
preserves behaviour: no (fixes the recording)
changes public surface: no

### F4: a tape file whose method the page does not register crashes the page at render, and `parseTape` casts most of the file unchecked

kind: bug
evidence: `react/session/SessionExport.tsx:184-195` (`loadTape`) calls `parseTape(text)` inside a promise whose `catch` shows a note, then hands the file to `props.onLoadTape`; the page stores it and `useTape` (`react/session/useTape.ts:66`) calls `replayLive` in `useMemo`, which calls `replaySessions` -> `registry.need(file.method)` (`producer/tape.ts:329`), documented `@throws MdmlError` and thrown **during render**, outside the `catch`. `parseTape` (`producer/tape.ts:182-215`) checks the envelope (schema, method, seed, variants, label, head, kinds is an object) and then casts `kinds`, `fired` (`as Record<...>`), `probes` (`:204`, `as TapeProbe[]`) and `config` without looking: `probes: [1]` or a non-numeric `head` reach `ReplaySession`'s constructor (`tape.ts:237`), and `kinds` values that are not path kinds fall to the default `scalar` in `stubSource` (`remote.ts`, `default:` at `:78`).
what: choosing a tape from another page (the menu's "load JSON") is the one user-supplied input in the package; its method check should happen where the error can be shown. The casts are the ones in the cast count that matter.
fix: `onLoadTape` is called only after `replaySessions(file, registry)` succeeded (do the lookup in `loadTape` with the page's registry, or have `useTape` catch and return an `{ error }`); in `parseTape` check `head` is a finite non-negative integer not past the run's last step, `kinds` values are strings of the known set, `probes` entries have a numeric `t` and a string `path`; add the three cases to `tape.test.ts`.
size: S
risk: low
preserves behaviour: yes for valid files
changes public surface: no

### F5: a table-driven smoke over every record view closes most of the views gap for three tests

kind: test-gap
evidence: coverage: `react/views` 238 of 981 statements (24%) in 44 files (`ExpositionView.tsx` 14%, `stages.tsx` 5%, `trees.tsx` 2%, `fairness.tsx` 3%, `attributions.tsx` 7%, `paths.tsx` <10%); `views/registry.ts` 36 of 64 functions: **28 of the 61 `still` closures never run**; `views/{agents,epicycles,sequences}.ts` below 10%. The one test that crosses registry, component and recipe (`apps/site/src/gallery/coverage.test.ts`) checks that a recipe *names* each kind and view, not that `viewFor(kind).component` is a real export (the 61 names do all exist today, checked) nor that anything draws.
what: the registry is exhaustive by type, so a fixture per kind is the natural unit. About half of the kinds already have a record built in `views/*.test.ts` (`views.test.ts`, `planes.test.ts`, `fairness.test.ts`, `trust.test.ts`, ...); the rest have none in render.
fix: move the records the existing view tests build into one private module `views/_samples.ts`, typed `{ readonly [K in ExpositionKind]: RecordOf<K> }` (exhaustive at compile time like `REGISTRY`, one engine constructor call on the smallest legal input each, ~5 lines a kind), import it from those tests, and add `views/registry.smoke.test.ts` and `react/views/exposition.smoke.test.tsx` with three table tests (a loop inside one `it` each, so +3): (1) `stillOf(sample)` returns an object with a `series` or a `kind: 'scene'` for every kind, and `viewFor(k).kind === k`; (2) `renderToString(<ExpositionView record={sample} />)` does not throw and is non-empty (SSR runs every view body and builder `useMemo`, not effects, which is what the 24% lacks); (3) `viewFor(k).component` is an export of `react/views/index.ts` (`Object.keys(await import(...))`). One further row per kind with the arrays emptied documents what F2 decides. Offset: delete the bare `expect(stillOf(r)).toBeDefined()/toBeTruthy()` lines in the `rules` and `images` view tests that (1) subsumes.
size: M (the fixtures)
risk: low
preserves behaviour: yes
changes public surface: no

### F6: `Player` takes Space, arrows, Home and End from every button inside it and from its own fps menu

kind: a11y
evidence: `react/Player.tsx:102-114`: the root `div` has `onKeyDown` for Space (`c.toggle()`), ArrowLeft/Right, Home, End, skipping only `INPUT` targets, then `e.preventDefault()`. It wraps nine `<button>`s and the `Scrubber`. A keyboard user who tabs to "Go to the end" and presses Space toggles playback instead of activating the button (whether the browser then still fires the button's own click on keyup differs by browser: needs a browser). The fps menu is `Popover` (`controls/Popover.tsx:78` `createPortal(..., document.body)`), and React synthetic events bubble through portals along the React tree, so Space or an arrow on a focused `role="option"` in the menu reaches the same handler: the menu items cannot be activated from the keyboard without also toggling play. The second listener (`Player.tsx:83-96`) takes ArrowLeft/Right on the whole `window` for ten seconds after play stops (`RECENT_MS`), excluding `INPUT`, `TEXTAREA`, `SELECT` and editable text only: a focused ARIA slider, radio, tab or listbox elsewhere on the page (`DraggableLine`, `ModelSelector`, `NarrativeView`'s tabs) is stepped over. `defaultPrevented` protects only elements whose own handler prevents default.
what: the keyboard contract of a composite widget says the keys act on the focused control. The shortcuts are a convenience (and documented in the TSDoc), so they should yield to the control that has focus.
fix: in the root handler act only when `e.target === e.currentTarget` or the target is the scrubber (`role="slider"` inside), never for `BUTTON` or `[role=option]`; the window listener excludes `[role=slider|radio|tab|option|textbox|spinbutton]` and `isContentEditable` (the `isTyping` of `interaction/keys.ts:18` is the existing helper, extend it); one `Player` key test through the harness (F18).
size: S
risk: low
preserves behaviour: no (keys stop acting on focused buttons)
changes public surface: no

### F7: the download and the page-recording code is written four times, and two of the copies leak an object URL

kind: duplicate
evidence: `download(url, name)` (anchor, click) at `react/FigureFrame.tsx:119` and `react/session/SessionExport.tsx:51`; `nextPaint()` (two animation frames and a timeout) at `FigureFrame.tsx:126` (20 ms) and `SessionExport.tsx:46` (30 ms); the Blob-to-URL helper at `SessionExport.tsx:58` (revoked after 10 s) and inline at `FigureFrame.tsx:301-305`; two more inline copies that never revoke: `react/session/SessionPanels.tsx:713-716` (`ProbeView`'s JSON export) and `react/session/ZoomPanel.tsx:134-138` (the PNG export), each `URL.createObjectURL` without `revokeObjectURL`, a blob kept alive per click. The recording loop (seek, `nextPaint`, serialise, rasterise, restore the cursor) is `FigureFrame.recordFrames` (`:132-162`, up to 600 frames) and `SessionExport.record` (`:123-160`, up to 200), the same shape written twice with different limits and only one of them pausing (F3).
what: a rule of the repository ("a missing helper goes into a shared module first") and the cause of F3's divergence.
fix: `react/download.ts` with `saveUrl(url, name)`, `saveBlob(bytes, type, name)` (revokes after 10 s) and `nextPaint()`; the four sites call them; a `recordPages({ seek, capture, count, limit, signal })` shared by both recorders only if F3 is fixed in the same change (otherwise leave the loops).
size: S
risk: low
preserves behaviour: yes (and the two leaks go)
changes public surface: no (new module is private)

### F8: `recordOf` is written four times and the settings-gear block twice

kind: duplicate
evidence: the Phase 0 pair is `react/controls/ModelSelector.tsx:34-35` and `react/controls/SlotVariants.tsx:52-53` (`typeof v === 'object' && v !== null && !Array.isArray(v) ? (v as Readonly<Record<string, SpaceValue>>) : {}`), the same two lines again as `recordOf` at `react/session/slots.ts:26-27`, and as `asRecord` at `react/controls/space-controls.ts:60` (the scan missed it, under its token floor). The larger repeated unit is the gear button, its `Popover` and the `SpaceControls` inside it: `ModelSelector.tsx:98-143` and `SlotVariants.tsx:92-143` are the same 40 lines (identical class string for the gear, `anchors` ref map, `anchor={{ current: anchors.current[name] ?? null }}` creating a new object each render, `Object.keys(space.options[name].spaces ?? {}).length` for the knob count, `role="dialog"`, `className="w-64 p-3"`). Only the `onChange` and whether choosing selects differ.
what: the verified duplicate of the register (F-035) understated: it is one helper in four files plus one component in two. `anchor` read from a ref during render is correct only because the popover opens after a click re-renders.
fix: export `asRecord` from `controls/space-controls.ts` (it exists there) and delete the three copies; extract `OptionSettings({ name, space, value, onChange, disabled, open, onOpen, onClose })` (gear button, popover and controls, anchor held by `useRef` inside) into `controls/OptionSettings.tsx` and use it in both lists; the existing `ModelSelector.test.tsx` and `SlotVariants.test.tsx` cover it.
size: S
risk: low
preserves behaviour: yes
changes public surface: no

### F9: charts have no text alternative

kind: a11y
evidence: `react/Chart.tsx:167-174` renders `<div ref className style data-mdml-chart="">` with no `role`, `aria-label` or description; `react/FigureFrame.tsx:430` is a `<figure>` whose `title` is a visible `div`, not a `<figcaption>`/label (the caption is a real `figcaption`). ECharts can describe a chart itself (`aria: { enabled: true, label: { description } }`); `views/_view.ts:107` has the helper `ariaOf(description)`, used by **11 of the 46 view builders** (`complex-plane, fields, images, landscape-slice, matrix-marks, points-steps, reachability, simplex, stability-region, strips, equation-fit`); the other 35 (`steps`, `surfaces`, `paths`, `tables`, `trust`, `thresholds`, `stages`, `attributions`, `boards`, ...), every `charts/` builder and `trace/` builder set none (`grep aria: charts trace views` finds the helper only), and `theme/echarts.ts` `baseOption` (`:120`) does not enable it. `react/SceneView.tsx:265-266` does it right for scenes (`role="img"`, `aria-label`, `<title>`).
what: for a screen-reader user every chart is an empty div. The chart is data the page explains in prose, so the loss is real but not total; the fix is cheap because ECharts generates a default description from title, series types and axes when `aria.enabled` is true.
fix: `baseOption` in `theme/echarts.ts` gets `aria: { enabled: true }` (ECharts then describes any chart from its own title and series, and builders that call `ariaOf` override the description); `Chart` takes an optional `label` prop and puts `role="img"` and `aria-label` (the `label`, else the option's `aria.label.description`, else the `title`) on its container; `FigureFrame` links its title with `aria-labelledby`. Check the ECharts text for a multi-grid figure by ear before relying on it.
size: S
risk: low
preserves behaviour: yes (adds attributes; golden options that snapshot `baseOption` change by one key)
changes public surface: no

### F10: the expanded figure is an unmanaged modal, and the export menu and popovers are keyboard-incomplete

kind: a11y
evidence: `react/FigureFrame.tsx:448-499`: `role="dialog" aria-modal="true"` with no `aria-label`, no focus move into the dialog, no focus trap, no focus return to the "Expand" button, `onKeyDown={() => {}}` (a no-op silencing the linter), Escape handled on `window` (`:266-279`); the page behind stays in the tab order although `aria-modal` says it is inert. The export menu in the same file (`:373-410`) is a hand-rolled `role="menu"` of `role="menuitem"` buttons positioned `absolute`: no Escape, no outside-click close (it stays open until a format is chosen or the button toggled), no arrow keys, no focus management; it does not use the `Popover` the rest of the package uses, nor `controls/icons.tsx` (it has its own `Icon` and `ICON_*` paths) nor the class tokens (inline styles against `var(--wb-*)` classes). `controls/Popover.tsx` (all of `listbox`, `menu`, `dialog` roles) moves no focus on open, returns none on close, supports no arrow-key navigation and has 1 of 8 functions run in tests. `controls/SettingsButton.tsx:75-110` already does all of it for its dialog (focus into the dialog, Tab trap, Escape, focus back to the trigger).
what: three ways to get a dialog or menu in one package, one of them correct. The roles promise behaviour (`aria-modal`, `menu`, `listbox`) that the code does not provide.
fix: lift `SettingsButton`'s open/close/trap logic into `useDialogFocus(open, ref, triggerRef)` in `controls/`, use it in `Popover` (when `role` is `dialog`/`menu`/`listbox`) and in the `FigureFrame` modal (add `aria-label`), and render `FigureFrame`'s export menu through `Popover` with `role="menu"`; arrow-key movement among `menuitem`s is an additional 15 lines. Needs a browser to check focus order.
size: M
risk: medium (focus behaviour; the expanded figure is used on every page)
preserves behaviour: no (adds focus behaviour)
changes public surface: no

### F11: `Math.max(...x)` and `Math.min(...x)` over data throw `RangeError` from about 125,000 values (65 sites)

kind: bug
evidence: probe `p4`: `wantsLog(values)` (`views/_view.ts:38`, `Math.max(...values) / Math.min(...values)`) throws `RangeError: Maximum call stack size exceeded` at 130,000 values in Node (the limit is smaller in some browsers' engines: needs a browser), fine at 50,000. 65 spread sites in non-test source; the ones over data-length arrays: `charts/fields.ts:126-127`, `charts/annotations.ts:89-90`, `charts/distributions.ts:367, 607`, `charts/points.ts:94-95`, `views/_view.ts:38`, `views/thresholds.ts:58-59, 158`, `views/steps.ts:525`, `views/trust.ts:148, 201`, `views/tables.ts:154-161`, `views/boards.ts:210-211, 521-522`, `views/sequences.ts:86-87`, `views/epicycles.ts:126`, `react/session/overlays.ts:220-221, 245`, `trace/messages.ts:51`. The loop that does it safely is in the same file as the worst one: `views/_view.ts:42` `spanOf` (finite-only min/max in one pass), and `mdml-engine/base` has `maxOf`/`minOf` (`base/numbers.ts:149, 170`). A tape of a long run (a metric logged at every step of 200,000) reaches `wantsLog`, `metricCurves` and the step marker.
what: a latent crash for long runs, and the "no local copies of generic helpers" rule seen from the other side.
fix: replace the data-length spreads by `spanOf` (finite-aware, already there) or `maxOf`/`minOf`; the short ones over a handful of entries (`layout/*`, `structures/*`) can stay; one test at 200,000 values on `wantsLog` and `limitsOf`.
size: S
risk: low (identical results for finite input; `spanOf` ignores NaN where the spread returns NaN, check the three callers that rely on NaN propagation)
preserves behaviour: yes for finite input
changes public surface: no

### F12: the live handle is rebuilt with new function identities on every render and nothing in the package is memoised, so every panel re-renders on every playhead tick

kind: hot-path
evidence: `react/session/useSession.ts:351-366` defines `compute`, `halt`, `follow`, `play`, `pause`, `dispose` as plain closures inside the hook body, `:372-392` builds `handle` and `:393-404` the `variants` array of fresh `{ label, live: { ...handle, ... } }` objects each render; the producer, `version`, `status` and `progress` are stable between ticks but the object graph is not. `grep "memo("` finds no `React.memo` in `react/` (49 files use `useMemo`). `SessionPlayer` sets the page's cursor once per displayed frame (`SessionPlayer.tsx:105-110`, by design) and `SessionPanels` (`:90-163`) re-renders every `PanelView` below it with a new `live`, new `variants` (`variantsOf(live)` makes `[{ label: '', live }]` for a single session) and `JSON.stringify(panel)` keys. Panels do their own `useMemo` on `t`/`version`, which is why this works, but the memo dependencies that include `variants` (`CurvesView` `SessionPanels.tsx:407`, `SweepView` `:526`) change on every render, and panels that do not depend on the step (the `tape` panel, a `field` panel at a held step) re-render and diff on every tick for nothing. Not measured (needs the React profiler).
what: the playhead redraws at the display rate by design; the question is how much work a tick costs once it has re-rendered N panels. With the handle stable, `React.memo` on the panels (props `panel`, `live.session`, `live.version`, `t`) would skip the ones the tick does not touch.
fix: build the handle with `useMemo` keyed on `[producer, version, computing, playing, lastMs, stale, held, start, status, progress, done]` and the verbs with `useCallback`/refs (they already read `current.current`); build `list` in the same memo; then `React.memo` the panel components (`CurvesView`, `SweepView`, `SeriesView`, `RowsPanel`, `TensorView`, ...) and pass `variants` by the memoised value. Measure with the profiler on a page with eight panels before and after.
size: M
risk: medium (stale-closure hazards in the verbs; the strict-mode and unmount tests in `useSession.test.tsx` and `SessionPlayer.test.tsx` cover the lifecycle)
preserves behaviour: yes
changes public surface: no

### F13: inner sessions and probes compute synchronously in render, once per viewed step, with no cache across steps

kind: hot-path
evidence: `react/session/InnerPanel.tsx:84-99`: `useMemo(() => session(spec.method, { ...options, mode: 'eager' }), [spec, name, live.session, at, argsKey])` with `at = props.t` when the cursor is `own`, so the whole inner run (the length the page chose) is computed on the main thread, inside render, for every outer step the cursor visits, including every playhead tick; `useEffect(() => setOwn(null), [made.inner])` then resets the inner cursor on each of them. `react/session/RowsPanel.tsx:50-68` (`useProbe`) and `:88-106` (`useChannel`): `useMemo` keyed on `(session, t, path, args)`, running `live.session.probe(t, ...)` in render. React does not promise to keep a memo across renders, and a revisited step is recomputed (scrubbing back and forth, or looping), where `producer/client.ts` keeps a bounded LRU of probe answers for worker sessions (`remember`, `CACHE`) and the inline producer keeps none. A decision-surface probe (a grid of predictions) is the heavy one.
what: the work per frame is bounded by the page's choices, but nothing in the library bounds it, and the worker path has the cache that the inline path lacks.
fix: a small `Map`-based LRU keyed `(variant, t, path, args)` in `useProbe`/`useChannel` (module-level in `RowsPanel.tsx`, keyed by `session` via a `WeakMap` so it dies with the session), and the same for `InnerPanel`'s inner sessions keyed by `at`; for `own` inner cursors compute at the settled step (`useSettled` exists in `session/parts.tsx:44`).
size: S to M
risk: low
preserves behaviour: yes
changes public surface: no

### F14: the curves panel rebuilds and replaces the whole series set to move a one-pixel step marker

kind: hot-path
evidence: `react/session/SessionPanels.tsx:389-407`: `useMemo` over `[runs, variants, paths, mode, t, family]` calls `overlayCurves` and `cartesian([curves, ...stepMarker(curves, t, mode)])`, so each new `t` produces a new option holding all the curves, and `Chart` applies it with `setOption(option, { notMerge: true, lazyUpdate: true })` (`Chart.tsx:119-121`, `merge` defaults false). Probe `p1` (3 variants x 4 paths, 1,000 steps): the headless build is 1.3 to 1.7 ms, the option 12 series and **12,000 points, 224 KB**; the cost of handing that to ECharts at the display rate is the part that cannot be measured here (needs a browser), and it grows with run length (no `sampling`, `large` or `progressive` on the line series). `SweepView` (`:453-526`), `SeriesView` (`:557-572`) and `TrajectoryView` (`:595-623`) have the same shape. `stepMarker` marks the step with a `lines([...], name: '__step')` series.
what: the marker is the only thing that changes between ticks while the run is held. With animation off (`theme/echarts.ts:120`) the cost is the full diff and layout of every series.
fix: keep the curves option memoised on `[runs, paths, mode, family]` and move the marker out of it: either a second `Chart` `merge` update of only the marker series (`chart.setOption({ series: [marker] })` with a stable series `id`), or ECharts' `markLine` on the first series set through `merge`; add `sampling: 'lttb'` to line series above ~2,000 points. Measure with the browser profiler before and after on a 10,000-step run.
size: M
risk: medium (the legend and `__`-named series conventions)
preserves behaviour: yes (same picture)
changes public surface: no

### F15: variants and classes differ by colour alone, and the default palette is not colour-blind safe past four colours

kind: a11y
evidence: probe `p5`: `colourBlindCheck` (the library's own check, `theme/palettes.ts:408`) on `categoricalColour(0..9)` (tab10) reports `safe: false`, minimum pairwise distance 27.7 normal, **4.8 protanopia (pair 1,2: orange and green)**, 7.3 deuteranopia, 15.2 tritanopia, against its default threshold 10; the only test of it (`theme/theme.test.ts:94`) asserts safety of `tab10.slice(0, 4)`. `react/session/compare.ts:44-53` separates variants by colour (`categoricalColour(i)`) and path by dash only, so the several variants of a comparison (`useSessions`, `SlotVariants` swatches) are distinguished by hue alone; decision regions (`charts/fields.ts:219-224`), clusterings and class rows are colour-coded the same way; no chart uses ECharts' `aria.decal` (`grep decal` finds nothing); `categoricalColour` cycles after ten, so a comparison of eleven variants repeats colours with no other channel.
what: the library has the tooling (`simulateColourVision`, `colourBlindCheck`) and does not apply it to its default palette; a comparison of five or more variants cannot be told apart by a reader with protanopia.
fix: add a second channel for series in a comparison (a symbol per variant for points, a dash per variant in `overlayCurves` when `paths.length === 1`), turn on decal patterns for filled categorical series when `aria.decal` is cheap, and either document that more than four categories need a second channel or reorder tab10 for protanopia-safe adjacent pairs (the order in which variants take colours matters more than the set). A palette change is a visual change: keep it separate.
size: S (symbols and dashes), M (decals)
risk: low
preserves behaviour: no (visual)
changes public surface: no

### F16: no reduced-motion handling, and two views start playing on mount

kind: a11y
evidence: `grep prefers-reduced-motion|matchMedia` finds nothing in `packages/render`. `react/views/_frame.tsx:109-112`: `FramesPlayer` with `cycle` starts `usePlayback` with `{ loop: true, autoplay: true }`; the two views that use it are `AgentPlayer` (`react/views/agents.tsx`, 30 fps) and `EpicyclePlayer` (`epicycles.tsx`), so a page with either moves as soon as it mounts and loops for ever. The player has a pause button (so WCAG 2.2.2's mechanism exists) and ECharts animation is off globally (`theme/echarts.ts:120`), which are the good halves. The player also hijacks nothing on focus (see F6 for what it does with keys).
what: a reader who has asked the system for less motion still gets looping animation from mount.
fix: `usePlayback` reads `matchMedia('(prefers-reduced-motion: reduce)')` once (guarded for no `window`) and `autoplay` is ignored when it matches; `FramesPlayer cycle` shows its last frame then. A one-line test through the harness's stand-in `window`.
size: S
risk: low
preserves behaviour: no (reduced-motion readers only)
changes public surface: no

### F17: composite widgets claim ARIA patterns they do not implement, and `NarrativeView`'s Alt+Arrow shortcuts are the browser's back and forward

kind: a11y
evidence: `react/controls/ModelSelector.tsx:73-91` is a `role="radiogroup"` of `role="radio"` buttons, each its own tab stop, with no arrow-key movement and no roving `tabIndex` (the radiogroup pattern requires both); `react/session/NarrativeView.tsx:104-123` is a `role="tablist"` of `role="tab"` buttons with no `role="tabpanel"`, no `aria-controls`, no arrow keys and no roving index. `NarrativeView.tsx:96-100` handles `Alt+ArrowRight/Left` on the section and calls neither `preventDefault` nor `stopPropagation`; Alt+Left and Alt+Right are history back and forward in Chrome and Firefox on Windows and Linux, so the moment steps and the browser navigates (needs a browser). The `aria-label` of the group is set only `typeof props.label === 'string'` (`ModelSelector.tsx:75`, `SlotVariants.tsx:78`), so a node label leaves it unnamed. Healthy neighbours: `NumberField` (17 ARIA attributes), `Choice`, `Toggle`, `Select`, `DraggablePoints`/`DraggableLine` (focusable handles, arrow-key nudges), `Scrubber` (`role="slider"` with value range).
what: the roles are promises to assistive technology; a radiogroup that Tab walks through one by one and a tablist that cannot be arrowed are worse than plain buttons with `aria-pressed`.
fix: either implement the patterns (a `useRovingFocus(count)` of about 20 lines serves both) or use plain buttons with `aria-pressed`/`aria-current`; in `NarrativeView` use `Alt`-free keys (`[` and `]`, or PageUp/PageDown) or call `preventDefault`, and give the panel `role="tabpanel"` with `aria-labelledby`.
size: S to M
risk: low
preserves behaviour: no (keyboard behaviour)
changes public surface: no

### F18: the components readers touch most have their effects and handlers untested; six harness tests would reach them

kind: test-gap
evidence: coverage by file: `react/Chart.tsx` 24% (1 of 16 functions: instance creation, resize observer, `onEvents`, group connect, dispose are all effects), `react/FigureFrame.tsx` 21% (5 of 34: expand, Escape, export, recording), `react/Player.tsx` 30% (4 of 24: the keyboard handlers, the scrubber), `controls/Popover.tsx` 18% (1 of 8), `react/StructureView.tsx` 39%, `react/session/ZoomPanel.tsx` 38%, `react/session/SessionExport.tsx` 1.7%, `react/session/NarrativeView.tsx` 0% (its pure half `narrative.ts` is tested). The tests of these are `renderToString` (`react/smoke.test.tsx`), which runs no effects. `react/_harness.tsx` has the stand-in document, `fakeFrames`, `mountHook`, `mountElement`, `inAct`; it cannot host ECharts, so `loadEcharts` is the one thing to stub (`vi.mock('./echarts.ts')` with a recording fake instance).
what: the unrun parts are where F3, F6, F10 and F22 live.
fix: six tests, +6 offset by -3 from F5's net: (1) `Chart` with a fake `loadEcharts`: init with the theme name, `setOption` replace vs merge, `on`/`off` pairs for `onEvents`, `dispose` on unmount, re-init on mode change; (2) `FigureFrame`: expand, Escape closes and removes its window listeners (the harness's `addEventListener` is a no-op: give it a recording one), focus returns (after F10); (3) `Popover`: pointerdown outside and Escape call `onClose`, all four listeners removed on close; (4) `Player` keys: Space on the root toggles, Space on a button does not (after F6), the window arrows yield to a focused slider; (5) `SessionExport` `record` pauses and restores the cursor, a rejecting `onLoadTape` input shows a note, an unknown method shows a note (after F3, F4); (6) `NarrativeView`: Alt+Arrow moves through the marked steps and prevents default. Each is 25 to 40 lines.
size: M
risk: low
preserves behaviour: yes
changes public surface: no

### F19: 300 of 1,154 exports are used by nothing outside their own file and the tests; the register's candidates are real API, the helpers are not

kind: api
evidence: probe `unused.mjs` (word index over `packages/render/src` non-test source, `apps/`, `packages/meta`, docs): 1,154 exports in non-test non-barrel source; **300 with no use in any other source file or app** (158 interfaces, 21 types, 115 functions, 6 constants), of which 86 are named by a test and 214 by nothing but their own file; 37 functions and constants are named nowhere else at all (not even a test): `charts/bars.ts` `barOffset`; `charts/distributions.ts` `distributionRange`; `charts/multivariate.ts` `reorderColumns`; `charts/points.ts` `MARKERS`; `maths/tex.ts` `texErrorMarker`; `react/controls/{DatasetSelector,EnvironmentSelector}.tsx` `datasetTags`, `environmentTags`; `react/controls/number.ts` `validateNumber`; `react/session/RowsPanel.tsx` `useChannel`; `react/session/preferences.ts` `DEFAULT_PREFERENCES`; `react/session/trajectory.ts` `fullTensors`; `react/session/useSession.ts` `isComparison`; `react/session/useTape.ts` `replayLive`; `react/theme.tsx` `DEFAULT_COLOUR_MODE`, `usePageColourMode`; `react/views/fairness.tsx` `GroupContingencyTables`, `GroupDisparityTable`, `GroupReportWarnings`; `svg/export.ts` `svgToCanvas`; `views/attributions.ts` `FEATURE_VALUE_RAMP`; `views/components.ts` `screePanel`, `loadingsPanel`, `biplotPanel`, `topicsView`; `views/fairness.ts` `groupColour`, `groupRatesPanel`, `groupGapsPanel`, `groupRocPanel`; `views/paths.ts` `activeCounts`, `regularisationPathPanels`, `crossValidationPanel`, `learningCurvePanel`; `views/steps.ts` `mixtureScores`; `views/surfaces.ts` `surfaceRows`, `surfaceObjective`; `views/tables.ts` `coefficientIntervals`; `views/trees.ts` `nodePrediction`. The register's candidates (`episodeChart`, `messageTimeline`, `smoothEma`, `runsChart`, `runTable`, `histogramsOverTime`, `normHeatmap`, `easings`, `tweenTimeline`, `sweepTimeline`, `runTimeline`, `structureTimeline`, `interpolateScene`, `rasteriseScene`): each is defined in `trace/` or `animation/`, named by its test and its barrel, most carry `@index` and `@example` (`messageTimeline`, `runTable` and `easings` carry neither), they are the documented "traced run to figure" and "timeline to GIF" API of `llms.header.md`, and none is used by `apps/` (the three site recipes that import `mdml-render/trace` or `/animation` use other functions).
what: the two groups differ. The candidates are public, documented, tested API without a consumer in the repository: keep, and add the missing tags (`@index` for `messageTimeline`, `runTable`, `easings`; an `@example` for `easings`). The 37 are helpers exported only because the barrel `export *`s the file: the view panel builders (`screePanel`, `groupRatesPanel`, ...) are pieces of a view that `views/registry.ts` reaches through the view function, and the three `react/views/fairness.tsx` components (`GroupContingencyTables`, `GroupDisparityTable`, `GroupReportWarnings`) are parts of `GroupReportView` in the same file, over-exported and never run by a test (3.3% coverage).
fix: drop `export` on the 37 (or the 115 functions after a check against the docs site, which indexes by `@index` and is unaffected), which is a reviewed break of the public surface digest (`make api-surface-check`); keep exported types of exported functions; the fairness parts become module-private with F5's smoke render covering them through `GroupReportView`.
size: S
risk: low (the digest moves by the removed names, intended)
preserves behaviour: yes
changes public surface: yes (removals)

### F20: a record kind is routed in three places that nothing ties together

kind: consistency
evidence: `views/registry.ts` (`component: 'MarginView'` and `accepts` per kind), `react/views/ExpositionView.tsx` (a `switch` that names the component again and wires `points`/`labels` by hand per case, `:84-215`), and each view's own props. `component` is a free `string`; the exhaustiveness the type gives covers `kind`, not the name (all 61 names exist today: checked by script, but the only test that reads them, `apps/site/src/gallery/coverage.test.ts:73`, checks recipes, and a renamed component fails there only by accident). `clusteringSteps` is decided in two places, `registry.ts:164-170` (dendrogram if `merges`, reachability if `reachability` and one assignment, else frames) and `ExpositionView.tsx:101-106`, the same three-way branch; `decisionSurface` and `conformal` branch in `still` on whether steps or a stream exist (`registry.ts:125-129, 388-393`) while `ExpositionView` leaves it to the view. `ExpositionView` passes no `caption`, though `react/views/index.ts:2-8` and the generated index say all views take one. `plays` is a hand-written boolean per kind (18 true) with no check against the view actually having a player.
what: three tables that must agree, with a free string joining two of them.
fix: a shared predicate `clusteringVariant(record): 'dendrogram' | 'reachability' | 'steps'` in `views/steps.ts` used by both; the F5 table test (3) checks `component` against the barrel; add `caption` to `ExpositionViewProps` and pass it through `common`; derive `plays` from a `frames` property in the entry (optional, a follow-up).
size: S
risk: low
preserves behaviour: yes
changes public surface: yes (`ExpositionViewProps.caption`, one exported predicate)

### F21: the package's module-level state is small, and one piece is never cleared

kind: consistency
evidence: `react/Player.tsx:28` `let lastPlayed: string | undefined`, set in an effect (`:73`) and never cleared when its player unmounts, so after the page's only playing player is gone no other player's `lastPlayed !== id` check passes until one plays, harmless, but it is the state that made the session tests order-dependent (the register's F-034, fixed by per-file isolation, not at the source); `react/session/preferences.ts:20-21` `current` is read from `localStorage` once per page load and never re-read: no `storage` event listener, so two tabs show different eager/live settings until reload, and there is no reset for tests (the only other module state: `canvases` WeakMap in `Chart.tsx:25`, `leaving` WeakSet in `useSession.ts:170`, both fine).
what: small, but the repository has already paid for module-level state once (F-034), and the fix is two lines each.
fix: clear `lastPlayed` in the effect cleanup when it equals `id`; `subscribe` in `preferences.ts` also listens to `window` `storage` events for `KEY` and refreshes `current`; export `resetSessionPreferences()` for tests.
size: S
risk: low
preserves behaviour: yes
changes public surface: yes (one test-support export)

### F22: export and projection failures are swallowed

kind: consistency
evidence: `react/FigureFrame.tsx:307` (`.catch(() => undefined)` after a GIF/WebM recording that may have run for tens of seconds) and `:326` (`.catch(() => undefined)` after the PNG fallback also failed): the user presses Export and nothing happens, with no note, though `SessionExport` shows `setNote(...)` for the same failures (`session/SessionExport.tsx:108, 118, 155, 178, 193`). `react/session/SessionPanels.tsx:744` (`ProjectionView`: `catch { return undefined }`) renders `null` when a projection throws, where the sibling panels show a `Note` with the message (`RowsPanel.tsx` `useProbe` returns `{ error }` and `TrajectoryView` prints "surface unavailable: ..."). A recording of up to 600 frames (`FigureFrame.tsx:145`, each waiting two animation frames and 20 ms plus a rasterisation) cannot be cancelled.
what: three places where the reader gets a blank where the neighbouring component would say why.
fix: a `note` state in `FigureFrame` shown next to `busy`; `ProjectionView` shows `<Note>{message}</Note>`; a Cancel on the busy readout (an `AbortController` checked per frame, which F3 needs as well).
size: S
risk: low
preserves behaviour: no (adds messages)
changes public surface: no

### F23: `StructureView` passes an inline ref callback that sets state, so every render triggers another

kind: hot-path
evidence: `react/StructureView.tsx:95-99`: `ref={(el) => { box.current = el; setBoxEl(el) }}`. A ref callback with a new identity each render is called with `null` and then the element on every commit, so `setBoxEl(null)` and `setBoxEl(el)` run in each commit; the state ends equal, and React renders the component once more and bails out of the children. By reading (needs the React profiler): one extra component render per render, in a component that re-renders during drags and zoom (the pointer handlers set `view` state).
what: a known React pattern that turns a render into two; small here because the component is not large, but it runs on every pan frame.
fix: the ref callback `useCallback((el) => { box.current = el; setBoxEl(el) }, [])` (stable, so it runs only on mount and unmount), or drop `boxEl` and attach the wheel listener in a layout effect on `box.current`.
size: S
risk: low
preserves behaviour: yes
changes public surface: no

### F24: headless folders touch the DOM behind guards, and two test helpers ship in the package

kind: rule
evidence: `llms.header.md` says everything outside `src/react` is headless and runs in Node. `svg/export.ts:130-136` (`svgToCanvas`, `document.createElement`, `Image`), `animation/video.ts:154-157, 212-236, 257-264` (`VideoEncoder`, `MediaRecorder`, `document`) and `animation/frames.ts:64` (`typeof document !== 'undefined' && typeof Image`) use browser globals, each behind a `typeof` guard and documented as browser-only in its TSDoc (`@throws MdmlError ... not-implemented outside a browser`), so the rule holds in effect and the header says so (rasterising "needs a browser canvas"). `producer/_fake.ts` (98 lines, "a worker in memory, for tests") is in the package: `package.json` `files` includes `src`, `tsconfig.build.json` excludes only `*.test.*`, and `dist/producer/_fake.js` exists; it is imported only by `producer/producer.test.ts`, `producer/tape.test.ts` and `react/session/useSession.test.tsx`. `react/_harness.tsx` (287 lines, imports `vitest`) is in `src/react` under the same `files` entry and `tsconfig.build.json` includes it (the `dist` in the tree predates it).
what: nothing breaks; the shipped test helpers are noise in the published tarball, and `_harness.tsx` importing `vitest` from published source would fail a consumer who imported it. The browser-only exports are fine as documented.
fix: add `!src/**/_fake.ts`, `!src/**/_harness.tsx` to `files` and the two to the build `exclude`; optionally list the three browser-touching modules in `llms.header.md` next to the sentence that calls the rest headless.
size: S
risk: low
preserves behaviour: yes
changes public surface: no (neither is exported from a barrel)

### F25: the same hook and JSX patterns are written several times

kind: duplicate
evidence: (a) content-keyed memo by `JSON.stringify`: `controls/DatasetPicker.tsx:123` (`useDataset`), `controls/EnvironmentSelector.tsx:172` (`useEnvironment`), `session/useSession.ts:229`, `session/RowsPanel.tsx:57, 94`, `session/InnerPanel.tsx:82`, each with a `biome-ignore useExhaustiveDependencies`, and `SessionPanels.tsx:101` keying every panel by `JSON.stringify(panel)` per render (two identical panels in a layout collide on the key). (b) `ResizeObserver` measure effects with `typeof ResizeObserver === 'undefined'` guards: `Chart.tsx:110`, `FigureGrid.tsx:32`, `views/trees.tsx:46` (`useMeasuredWidth`). (c) The panel shell `useColourMode` + `useRun`/`useRuns` + `useMemo(option)` + `useScrubClick` + `<FigureFrame figure={FIG}><Chart figure={FIG} .../></FigureFrame>` in `SessionPanels.tsx`' `CurvesView` (`:373-418`), `SweepView` (`:422-541`), `SeriesView` (`:544-579`), `TrajectoryView` (`:582-634`), `TensorView` (`:638-688`), with `useRuns` (`:366`) a second copy of `useRun` (`RowsPanel.tsx:81`) over several variants. (d) `titled(\`${title} at step ${t}\`)`and`file(title)`repeated at nine`case`s of `PanelView`. The 29 record-view wrappers (`useViewFigure`+`useCallback(frames)`+`FramesPlayer`) are the same shape but already one function each over `_frame.tsx`: that is the healthy version of this.
what: none of it is wrong; (a) and (c) are where the next bug would be fixed in one copy.
fix: a `useByContent(value)`returning a stable reference while`JSON.stringify`is unchanged, replacing the`argsKey`lines;`ChartPanel({ title, filename, option, onInit, caption })` for the panel shell; leave (b) and (d).
size: S to M
risk: low
preserves behaviour: yes
changes public surface: no

### F26: every `useColourMode` call subscribes its own `MutationObserver` on the document

kind: hot-path
evidence: `react/theme.tsx:19-25, 40-45`: `useColourMode` calls `usePageColourMode`, i.e. `useSyncExternalStore(subscribePage, ...)` with `subscribePage` making a new `MutationObserver` observing `document.documentElement` attributes per subscriber. `useColourMode` is called by `Chart`, `FigureFrame`, `useViewFigure`, every session panel, `Player`-adjacent controls (`grep` counts it in 40 files), so a page of thirty figures holds a hundred or more observers on one element, all notified together when the class changes. By reading.
what: harmless at rest; a mode toggle runs every callback once per observer and re-renders every figure, which is needed anyway. The observers are the waste.
fix: one module-level observer with a listener set (created on the first subscriber, disconnected on the last), as `preferences.ts` does for its store.
size: S
risk: low
preserves behaviour: yes
changes public surface: no

### F27: TSDoc is present everywhere but `@example`, `@param` and `@returns` are patchy (functions in `.ts` modules)

kind: rule
evidence: `tsdoc.mjs` over the 574 exported functions of the non-private `.ts` modules: **0 without a doc comment**, 222 without `@example`, 38 without `@param`, 41 without `@returns` (`.tsx` and `export const` not scanned). Without `@param`/`@returns`: `react/controls/classes.ts:8` `cx`, `controls/number.ts:30` `hasRange`, `controls/space-controls.ts:248` `valueAt`, `interaction/keys.ts:18, 26` `isTyping`, `keyName`, `session/export.ts:93` `wrapText`, `session/families.ts:33, 46, 208` `hasFamily`, `logAxes`, `eventSteps`, `session/overlays.ts:73, 151` `isProbed`, `regionPoints`, `session/scrub.ts:17, 25` `recordedSteps`, `nearestStep`, `session/preferences.ts:54` `setSessionPreferences`, `producer/step.ts:43` `advanceInStep`, `gym/worlds.ts:225` `pathFrames`. Without `@example`, among the documented API: `animation/frames.ts:81` `timelineFrames`, `animation/video.ts:152, 255` `canEncodeVideo`, `encodeWebm`, `producer/client.ts:55` `workerProducer`, `producer/inline.ts:24` `inlineProducer`, `layout/graph.ts:67-137`. The `@index` lines checked (`Player`, `SessionPlayer`, `viewFor`, `ExpositionView`) point to the right `path:line`.
what: CLAUDE.md's `@example` rule is for the engine core; render has no stated standard, and the doc quality where it exists is high. The `@param`/`@returns` gaps are on the helpers that are also over-exported (F19).
fix: none for `@example`; add `@param`/`@returns` to the 17 above with F19 (or unexport them); say in `docs/structure.md` what render's standard is.
size: S
risk: low
preserves behaviour: yes
changes public surface: no

### F28: the 1,317 casts are mostly index narrowing and are not a defect; three are worth a look

kind: casts
evidence: `animation/gif.ts` (31 `as number` of 266 lines: `d[i] as number`, `counts[b] as number`) and `svg/_deflate.ts` (the same on typed arrays under `noUncheckedIndexedAccess`), `layout/layered.ts` (`layers[r] as string[]`) are index narrowing in kernels; `maths/derivations.ts` counts as dense only because the scan matches the word "as" in prose; `charts/annotations.ts:56` is `as unknown as SeriesOption` for an ECharts custom series, which ECharts' types do not express. The three to look at: `producer/tape.ts:195-201` (F4: parsed JSON cast to `Record<string, string>`, `TapeProbe[]`), `views/registry.ts:643, 664` (`REGISTRY[kind] as StandardView<K>`, `record as never`: the cast that hides F2's missing check) and `react/session/SessionExport.tsx` `props.layout as never`.
what: no action beyond the two entries that cover them.
fix: see F2 and F4; a typed `at(arr, i)` helper for the kernels would remove 60 casts and is for taste, not recommended.
size: n/a
risk: n/a
preserves behaviour: n/a
changes public surface: no

### F29: native arrays: nothing to migrate to tensors; the local type duplicates are already in the types scan

kind: native-array
evidence: render's arrays are plot data (series, option rows, scene coordinates, grids passed to ECharts), the engine's records arrive as `readonly number[]`/`number[][]` and leave as option data, and the places that compute on them (`charts/_resample.ts`, `layout/force.ts`, `views/trust.ts`) work on `Float64Array` or plain rows with loops sized by the plot, not by the model. The one numeric loop that would earn a kernel is the bicubic resample (F1) and it already works on `Float64Array`. The type restatements are in `findings.md` (`Pt` in `charts/lines.ts` and `svg/paths.ts`, `Point2` in `react/interaction/projection.ts`, `RiemannRule` in `charts/functions.ts`).
what: nothing to do here beyond the canonical-types proposal that already covers those names.
fix: none
size: n/a
risk: n/a
preserves behaviour: n/a
changes public surface: no
