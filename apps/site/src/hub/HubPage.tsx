import type { Json } from 'mdml-engine/base'
import { datasetCollections } from 'mdml-engine/datasets'
import { covarianceEllipse } from 'mdml-engine/numerics/geometry'
import { key } from 'mdml-engine/random'
import { type DictSpace, defaults } from 'mdml-engine/space'
import { type Tensor, toFlat } from 'mdml-engine/tensor'
import { contractOf } from 'mdml-engine/trace'
import { cartesian, type Layer, lines, points } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import {
  Chart,
  ControlPanel,
  DatasetPicker,
  ExpositionView,
  loadSelection,
  SpaceControls,
  selectEntry,
  useColourMode,
} from 'mdml-render/react'
import { categoricalColour } from 'mdml-render/theme'
import { type ReactNode, useMemo, useState } from 'react'
import { logsOf, SessionControls, type SessionSettings } from '../session/SessionControls.tsx'
import { TapeView } from '../session/TapeView.tsx'
import { type LiveSession, useSession } from '../session/useSession.ts'
import { PageHeader } from '../shell/layout.tsx'
import type { HubEntry } from './entries.ts'

const FIG = figure({ width: 520, height: 340 })

type Params = Record<string, never>

/**
 * The hub's host: any entry's estimator on a chosen dataset, run as a session. Dataset, hyperparameters and session
 * settings are controls; the tape is drawn generically; a probe panel shows the model at the scrubbed step.
 */
export function HubPage(props: { readonly entry: HubEntry }): ReactNode {
  const { entry } = props
  const estimator = entry.estimator
  const space = estimator.hyperSpace as DictSpace | undefined
  const collectionId = entry.datasets[0] ?? ''
  const [selection, setSelection] = useState(() => selectEntry(collectionId))
  const [params, setParams] = useState<Params>(() => (space === undefined ? {} : (defaults(space) as Params)))
  const [settings, setSettings] = useState<SessionSettings>({
    mode: 'eager',
    seed: 0,
    steps: 300,
    preset: 'defaults',
    chosen: [],
    every: 1,
  })
  const [at, setAt] = useState<number | null>(null)
  const loaded = useMemo(() => loadSelection(collectionId, selection), [collectionId, selection])
  const data = useMemo(() => {
    const d = loaded.data as unknown as { readonly x: Tensor; readonly y?: Tensor }
    return d.y === undefined ? { x: d.x } : { x: d.x, y: d.y }
  }, [loaded])
  const method = useMemo(() => estimator.with(params).method(data), [estimator, params, data])
  const paths = useMemo(() => Object.keys(contractOf(method, params, key(0))), [method, params])
  const logs = logsOf(settings)
  const live = useSession(
    method,
    { config: params, seed: settings.seed, mode: settings.mode, steps: settings.steps, logs },
    [method, JSON.stringify(params), settings.seed, settings.mode, settings.steps, JSON.stringify(logs)],
  )
  return (
    <div className="flex flex-col gap-4">
      <PageHeader title={entry.title} lede={entry.summary} eyebrow={{ label: 'Hub', path: '/hub' }} />
      <ControlPanel columns={4}>
        <DatasetPicker
          collection={datasetCollections[collectionId] ?? collectionId}
          value={selection}
          onChange={(s) => {
            setSelection(s)
            setAt(null)
          }}
        />
      </ControlPanel>
      {space === undefined ? null : (
        <ControlPanel columns={4}>
          <SpaceControls space={space} value={params} onChange={(p) => setParams(p as Params)} />
        </ControlPanel>
      )}
      <SessionControls paths={paths} settings={settings} onChange={setSettings} />
      <TapeView live={live} at={at} onScrub={setAt}>
        <ProbePanel
          entry={entry}
          live={live}
          paths={paths}
          at={at}
          x={data.x}
          {...(data.y === undefined ? {} : { y: data.y })}
        />
      </TapeView>
    </div>
  )
}

type Probed = { readonly value?: unknown; readonly error?: string }

// One probe of the session at a step, computed once per (session, step) and never throwing: a failure is returned as
// its message so a panel can say why and still draw its rows.
// biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
function useProbe(live: LiveSession<any, any>, t: number, path: string | undefined, args?: Json): Probed {
  const argsKey = JSON.stringify(args ?? null)
  // biome-ignore lint/correctness/useExhaustiveDependencies: keyed on the viewed step, not on every tape version
  return useMemo(() => {
    if (path === undefined) return {}
    try {
      return { value: live.session.probe(t, path, args) }
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) }
    }
  }, [live.session, t, path, argsKey])
}

const Note = (props: { readonly children: ReactNode }): ReactNode => (
  <div className="text-xs text-muted">{props.children}</div>
)

// The model at the scrubbed step. The training rows are always drawn (by class or target when the dataset has them);
// the probe overlays on top when it succeeds: clusterers the assignments and centres, regressors the fit, the rest
// their decision surface. Each probe is independent, and every chart stays mounted across frames.
function ProbePanel(props: {
  readonly entry: HubEntry
  readonly paths: readonly string[]
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly at: number | null
  readonly x: Tensor
  readonly y?: Tensor
}): ReactNode {
  const { entry, live, at, x, y, paths } = props
  const mode = useColourMode()
  const kind = entry.estimator.kind
  const t = at === null ? live.session.t : Math.min(at, live.session.t)
  const clusterer = kind === 'clusterer'
  const oneD = kind === 'regressor' && (x.shape[1] ?? 1) === 1 && y !== undefined
  const assignments = useProbe(live, t, clusterer ? 'model/assignments' : undefined)
  const centres = useProbe(live, t, clusterer ? 'model/centres' : undefined)
  // A mixture's parameters carry covariances: drawn as each component's 1 and 2 sigma ellipses.
  const params = useProbe(live, t, clusterer ? 'model/params' : undefined)
  const fit = useProbe(live, t, oneD ? 'model/predictions' : undefined)
  const surface = useProbe(live, t, clusterer || oneD ? undefined : 'model/decision-surface')
  const density = useProbe(live, t, clusterer && paths.includes('model/density') ? 'model/density' : undefined, {
    resolution: 40,
  })
  const xs = useMemo(() => toFlat(x), [x])
  const ys = useMemo(() => (y === undefined ? undefined : toFlat(y)), [y])
  const d = x.shape[1] ?? 1
  const n = x.shape[0] ?? 0
  const rowX = useMemo(() => Array.from({ length: n }, (_, i) => xs[i * d] ?? 0), [xs, n, d])
  const rowY = useMemo(
    () => Array.from({ length: n }, (_, i) => (d > 1 ? (xs[i * d + 1] ?? 0) : (ys?.[i] ?? 0))),
    [xs, ys, n, d],
  )
  const labels = assignments.value as number[] | undefined
  const classes = kind === 'classifier' && ys !== undefined ? Array.from(ys) : undefined
  const groups = labels ?? classes
  const centreRows = centres.value as number[][] | undefined
  const covs = (params.value as { readonly covariances?: number[][][] } | undefined)?.covariances
  const fitted = fit.value as number[] | undefined
  const errors = [assignments, centres, fit, surface, density].flatMap((p) => (p.error === undefined ? [] : [p.error]))
  const title = clusterer
    ? covs === undefined
      ? `probe: assignments and centres at step ${t}`
      : `probe: components at step ${t} (1σ solid, 2σ dashed)`
    : oneD
      ? `probe: the fit at step ${t}`
      : `probe: the rows at step ${t}`
  const option = useMemo(() => {
    const layers: Layer[] = [
      points([{ x: rowX, y: rowY, ...(groups === undefined ? {} : { groups }), opacity: 0.7 }], { mode }),
    ]
    if (centreRows !== undefined)
      layers.push(
        points(
          [
            {
              x: centreRows.map((c) => c[0] ?? 0),
              y: centreRows.map((c) => c[1] ?? 0),
              name: 'centres',
              size: 16,
              symbol: 'diamond',
            },
          ],
          { mode, colourOffset: 9 },
        ),
      )
    if (centreRows !== undefined && covs !== undefined && d >= 2) {
      centreRows.forEach((m, k) => {
        const c = covs[k]
        if (c === undefined) return
        const sigma = [
          [c[0]?.[0] ?? 0, c[0]?.[1] ?? 0],
          [c[1]?.[0] ?? 0, c[1]?.[1] ?? 0],
        ]
        for (const nStd of [1, 2]) {
          const flat = toFlat(covarianceEllipse([m[0] ?? 0, m[1] ?? 0], sigma, { nStd, points: 72 }).points)
          const ex = Array.from({ length: 73 }, (_, i) => flat[(2 * i) % flat.length] ?? 0)
          const ey = Array.from({ length: 73 }, (_, i) => flat[(2 * i + 1) % flat.length] ?? 0)
          layers.push(
            lines(
              [
                {
                  x: ex,
                  y: ey,
                  name: `__ring${k}-${nStd}`,
                  colour: categoricalColour(k),
                  width: nStd === 1 ? 2 : 1,
                  dash: nStd === 1 ? 'solid' : 'dashed',
                },
              ],
              { mode },
            ),
          )
        }
      })
    }
    if (fitted !== undefined && d === 1) {
      const order = rowX.map((_, i) => i).sort((a, b) => (rowX[a] ?? 0) - (rowX[b] ?? 0))
      layers.push(
        lines([{ x: order.map((i) => rowX[i] ?? 0), y: order.map((i) => fitted[i] ?? 0), name: 'fit' }], {
          mode,
          colourOffset: 1,
        }),
      )
    }
    return { ...cartesian(layers, { mode, figure: FIG, title }), animation: false }
  }, [rowX, rowY, groups, centreRows, covs, fitted, d, mode, title])
  const showSurface = surface.value !== undefined
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap gap-4">
        {showSurface ? (
          <ExpositionView
            record={surface.value as never}
            figure={FIG}
            points={rowsOf(xs, d)}
            {...(groups === undefined ? {} : { labels: groups })}
          />
        ) : (
          <Chart figure={FIG} option={option} />
        )}
        {density.value === undefined ? null : <ExpositionView record={density.value as never} figure={FIG} />}
      </div>
      {errors.length === 0 ? null : <Note>probe unavailable: {errors[0]}</Note>}
    </div>
  )
}

function rowsOf(flat: ArrayLike<number>, width: number): number[][] {
  const out: number[][] = []
  for (let i = 0; i + width <= flat.length; i += width)
    out.push(Array.from({ length: width }, (_, j) => flat[i + j] ?? 0))
  return out
}
