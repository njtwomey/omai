import { conjugateModelOf, type SegmentModel } from 'omai-engine/datasets'
import { filtering } from 'omai-engine/dynamics'
import { defaults } from 'omai-engine/space'
import { asTensor, toFlat } from 'omai-engine/tensor'
import {
  bocpdDetector,
  type ChangepointData,
  changepointData,
  changepointDetectors,
  changepointPrecisionRecall,
} from 'omai-engine/timeseries'
import type { Config } from 'omai-engine/trace'
import {
  Choice,
  ControlPanel,
  DatasetPicker,
  type DatasetSelection,
  SessionPlayer,
  SpaceControls,
  selectEntry,
  useDataset,
  useSession,
  useSessionPreferences,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { StaleVeil } from '../../shared/StaleVeil.tsx'
import { BocpdView } from './BocpdView.tsx'

const DATA = 'changepoints'
const FIRST = (): DatasetSelection => selectEntry(DATA, 'changepointSeries')
// A detection within this many steps of a true changepoint counts as finding it.
const MARGIN = 10

export default function BocpdPage(): ReactNode {
  const [selection, setSelection] = useState<DatasetSelection>(FIRST)
  const generated = useDataset(DATA, selection)
  const series = useMemo(() => {
    try {
      return changepointData(generated.data as never)
    } catch {
      return undefined
    }
  }, [generated.data])
  if (series === undefined) return <p className="text-sm text-muted">This page needs a single-channel time series.</p>
  // What the generator planted, so the model can be set to match it.
  const planted = (generated.truth?.extra as { model?: SegmentModel } | undefined)?.model
  return <Page series={series} planted={planted} selection={selection} setSelection={setSelection} />
}

function Page(props: {
  readonly series: ChangepointData
  readonly planted: SegmentModel | undefined
  readonly selection: DatasetSelection
  readonly setSelection: (s: DatasetSelection) => void
}): ReactNode {
  const { series, planted, selection, setSelection } = props
  const y = useMemo(() => Array.from(toFlat(asTensor(series.x))), [series.x])
  const truth = series.changepoints ?? []
  const method = useMemo(() => bocpdDetector.method(series), [series])
  const [config, setConfig] = useState<Config>(() => defaults(method.config) as Config)
  const [follow, setFollow] = useState(true)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  // With "model follows the data" on, the segment model is the one that generated the series (its true priors).
  const effective = useMemo<Config>(
    () =>
      follow && planted !== undefined
        ? ({ ...config, model: filtering.segmentModelValueOf(conjugateModelOf(planted)) } as Config)
        : config,
    [config, follow, planted],
  )
  const live = useSession(
    method,
    { config: effective, mode: prefs.mode, steps: y.length },
    [method, JSON.stringify(effective)],
    {
      hold: prefs.hold,
    },
  )
  const reset = (): void => {
    setConfig(defaults(method.config) as Config)
    setFollow(true)
    setSelection(FIRST())
  }
  const scores = useMemo(() => {
    const model = filtering.segmentModels.make(effective['model'])
    return changepointDetectors.names.map((name) => {
      const knobs = { ...(defaults(changepointDetectors.space.options[name] as never) as object), kind: name }
      const withModel = name === 'bocpd' ? { ...knobs, model: effective['model'], hazard: effective['hazard'] } : knobs
      try {
        const found = changepointDetectors.make(withModel as never)(y)
        const s = changepointPrecisionRecall(truth, found, MARGIN)
        return { name, found: found.length, ...s, model: name === 'bocpd' ? model.kind : undefined }
      } catch {
        return { name, found: 0, precision: Number.NaN, recall: Number.NaN, f1: Number.NaN, model: undefined }
      }
    })
  }, [y, truth, effective])
  const controls = (
    <div className="grid grid-cols-1 items-start gap-3 md:grid-cols-2">
      <ControlPanel title="data and the kind of shift" columns={1}>
        <DatasetPicker collection={DATA} value={selection} onChange={setSelection} />
      </ControlPanel>
      <ControlPanel title="the detector: segment model, hazard, thresholds" columns={1}>
        <Choice
          label="segment model"
          value={follow ? 'data' : 'own'}
          options={[
            { value: 'data', label: planted === undefined ? 'default' : 'follows the data' },
            { value: 'own', label: 'set by hand' },
          ]}
          onChange={(v) => setFollow(v === 'data')}
        />
        <SpaceControls
          space={method.config}
          value={effective}
          onChange={(v) => {
            setConfig(v as Config)
            if (follow && (v as Config)['model'] !== effective['model']) setFollow(false)
          }}
        />
      </ControlPanel>
    </div>
  )
  return (
    <div className="flex flex-col gap-4">
      <StaleVeil stale={live.stale}>
        <BocpdView live={live} at={at} y={y} truth={truth} onScrub={setAt} />
      </StaleVeil>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={reset} />
      <Fold title="Configuration: the data, the kind of shift, the model and the hazard">{controls}</Fold>
      <Fold title="Compare with other detectors">
        <Scores rows={scores} truth={truth.length} margin={MARGIN} />
      </Fold>
    </div>
  )
}

// A section closed until asked for, so the page opens on the picture alone.
function Fold(props: { readonly title: string; readonly children: ReactNode }): ReactNode {
  return (
    <details className="group rounded-lg border border-grid bg-surface">
      <summary className="cursor-pointer select-none px-4 py-2.5 text-sm font-medium text-fg">{props.title}</summary>
      <div className="border-t border-line p-4">{props.children}</div>
    </details>
  )
}

function Scores(props: {
  readonly rows: readonly {
    readonly name: string
    readonly found: number
    readonly precision: number
    readonly recall: number
    readonly f1: number
  }[]
  readonly truth: number
  readonly margin: number
}): ReactNode {
  const f = (v: number): string => (Number.isNaN(v) ? '–' : v.toFixed(2))
  return (
    <section className="flex flex-col gap-2 text-sm">
      <h2 className="font-medium text-fg">
        The same series, five detectors ({props.truth} true changepoints, a detection within {props.margin} steps
        counts)
      </h2>
      <table className="w-full max-w-2xl text-left">
        <thead className="text-xs text-muted">
          <tr>
            <th className="py-1 font-normal">detector</th>
            <th className="font-normal">found</th>
            <th className="font-normal">precision</th>
            <th className="font-normal">recall</th>
            <th className="font-normal">F1</th>
          </tr>
        </thead>
        <tbody>
          {props.rows.map((r) => (
            <tr key={r.name} className="border-t border-line">
              <td className="py-1">{r.name}</td>
              <td>{r.found}</td>
              <td>{f(r.precision)}</td>
              <td>{f(r.recall)}</td>
              <td>{f(r.f1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-muted">
        The other detectors run with their own defaults; BOCPD uses the model and hazard set above. CUSUM and
        Page-Hinkley report when they alarm, not where the change was, so the margin forgives their delay.
      </p>
    </section>
  )
}
