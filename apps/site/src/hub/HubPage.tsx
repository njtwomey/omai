import { datasetCollections } from 'mdml-engine/datasets'
import { key } from 'mdml-engine/random'
import { type DictSpace, defaults } from 'mdml-engine/space'
import { type Tensor, toFlat } from 'mdml-engine/tensor'
import { contractOf } from 'mdml-engine/trace'
import { cartesian, points } from 'mdml-render/charts'
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
        <ProbePanel entry={entry} live={live} at={at} x={data.x} />
      </TapeView>
    </div>
  )
}

// The model at the scrubbed step: clusterers draw the rows by assignment with the centres over them, classifiers and
// regressors their decision surface.
function ProbePanel(props: {
  readonly entry: HubEntry
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly at: number | null
  readonly x: Tensor
}): ReactNode {
  const { entry, live, at, x } = props
  const mode = useColourMode()
  const t = at === null ? live.session.t : Math.min(at, live.session.t)
  // biome-ignore lint/correctness/useExhaustiveDependencies: probes re-run as the tape grows
  return useMemo(() => {
    try {
      if (entry.estimator.kind === 'clusterer') {
        const labels = live.session.probe(t, 'model/assignments') as number[]
        const centres = live.session.probe(t, 'model/centres') as number[][]
        const flat = toFlat(x)
        const d = x.shape[1] ?? 2
        const rows = {
          x: labels.map((_, i) => flat[i * d] ?? 0),
          y: labels.map((_, i) => flat[i * d + 1] ?? 0),
        }
        const marks = { x: centres.map((c) => c[0] ?? 0), y: centres.map((c) => c[1] ?? 0) }
        return (
          <Chart
            figure={FIG}
            option={cartesian(
              [
                points([{ ...rows, groups: labels, opacity: 0.7 }], { mode }),
                points([{ ...marks, name: 'centres', size: 16, symbol: 'diamond' }], { mode, colourOffset: 9 }),
              ],
              { mode, figure: FIG, title: `probe: assignments and centres at step ${t}` },
            )}
          />
        )
      }
      const record = live.session.probe(t, 'model/decision-surface') as never
      return <ExpositionView record={record} figure={FIG} />
    } catch {
      return null
    }
  }, [live.session, live.version, t, mode, entry, x])
}
