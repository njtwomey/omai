import {
  classifierBudgetKnobs,
  classifierDataKnobs,
  classifierKnobs,
  classifierModelKnobs,
  classifierObjective,
  classifierProblemKnobs,
  classifierSteps,
} from 'omai-engine/bayesian-examples'
import type { ChainDiagnostics, DecisionSurface, PosteriorPairs } from 'omai-engine/exposition'
import { defaults, dict } from 'omai-engine/space'
import { type Config, Log, metricSeries } from 'omai-engine/trace'
import { cartesian, lines } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import {
  ChainDiagnosticsView,
  Chart,
  ControlPanel,
  DecisionSurfaceView,
  type LiveSession,
  PosteriorPairsView,
  SeedInput,
  SessionPlayer,
  SpaceControls,
  useColourMode,
  useProbe,
  useRun,
  useSession,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { StaleVeil } from '../../shared/StaleVeil.tsx'
import { classifierMethod, classifyProducer } from './method.ts'

// biome-ignore lint/suspicious/noExplicitAny: a session is generic over its method
type Live = LiveSession<any, any>

const START = defaults(classifierMethod.config) as Config
const INFERENCE = dict({ inference: classifierKnobs.spaces['inference'] as never, ...classifierBudgetKnobs.spaces })
const CURVE = figure({ width: 520, height: 300 })
const SURFACE = 41
// The reference: a long NUTS run of the same problem, in a worker of its own once the chosen run has ended.
const REFERENCE_SIZE = 25
const REFERENCE = { kind: 'nuts', numChains: 2, numWarmup: 200, numDraws: 300, thin: 1 }

const pick = (config: Config, keys: readonly string[]): Config =>
  Object.fromEntries(keys.filter((k) => k in config).map((k) => [k, config[k]])) as Config
const num = (v: unknown, digits = 3): string => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '-')

/** A probe's record when it is the kind asked for. */
function useRecord<R>(live: Live, t: number, path: string, kind: string, args?: never): R | undefined {
  const p = useProbe(live, t, path, args)
  const v = p.value as { kind?: string } | null | undefined
  return v?.kind === kind ? (v as R) : undefined
}

export default function ClassifyPage(): ReactNode {
  const [config, setConfig] = useState<Config>(START)
  const [at, setAt] = useState<number | null>(null)
  // The tape keeps what the page draws from it: the objective curve of the chosen method (a sampler's draws come from probes).
  const objective = classifierObjective(config)
  const set = (v: Config): void => setConfig((c) => ({ ...c, ...v }))
  const live = useSession(
    classifierMethod,
    {
      config,
      seed: Number(config['dataSeed']),
      steps: classifierSteps(config),
      logs: objective === undefined ? {} : { [objective.path]: Log() },
    },
    [JSON.stringify(config)],
    { start: 'end', hold: false, producer: classifyProducer },
  )
  const problem = JSON.stringify(pick(config, classifierProblemKnobs))
  // biome-ignore lint/correctness/useExhaustiveDependencies: the reference depends on the problem alone, not on the method
  const referenceConfig = useMemo(
    () => ({
      ...config,
      inference: { ...((defaults(classifierMethod.config) as Config)['inference'] as object), ...REFERENCE },
      budgetMs: 20000,
      budgetSteps: 0,
    }),
    [problem],
  )
  const reference = useSession(
    classifierMethod,
    { config: referenceConfig, seed: 7, steps: classifierSteps(referenceConfig) },
    [problem],
    { start: 'end', hold: false, producer: classifyProducer, active: live.done },
  )
  const t = Math.max(0, Math.min(at ?? live.head, live.head))
  return (
    <div className="flex flex-col gap-5">
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">The experiment</h2>
        <div className="grid grid-cols-1 items-start gap-3 md:grid-cols-2">
          <ControlPanel title="data" columns={2}>
            <SpaceControls
              space={classifierDataKnobs}
              value={pick(config, Object.keys(classifierDataKnobs.spaces))}
              onChange={set}
              exclude={['dataSeed']}
              segmentedUpTo={1}
            />
            <SeedInput value={Number(config['dataSeed'])} onChange={(v) => set({ dataSeed: v })} />
          </ControlPanel>
          <ControlPanel title="model" columns={2}>
            <SpaceControls
              space={classifierModelKnobs}
              value={pick(config, Object.keys(classifierModelKnobs.spaces))}
              onChange={set}
              segmentedUpTo={1}
            />
          </ControlPanel>
        </div>
      </section>
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">The inference</h2>
        <ControlPanel title="one method, its knobs and the time budget (0 for none)" columns={4}>
          <SpaceControls
            space={INFERENCE}
            value={pick(config, Object.keys(INFERENCE.spaces))}
            onChange={set}
            segmentedUpTo={1}
          />
        </ControlPanel>
        <StaleVeil stale={live.stale}>
          <Outputs
            live={live}
            reference={reference}
            t={t}
            config={config}
            player={<SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => setConfig(START)} start="end" />}
          />
        </StaleVeil>
      </section>
    </div>
  )
}

function Outputs(props: {
  readonly live: Live
  readonly reference: Live
  readonly t: number
  readonly config: Config
  readonly player: ReactNode
}): ReactNode {
  const { live, reference, t, config, player } = props
  const surface = useRecord<DecisionSurface>(live, t, 'classify/surface', 'decisionSurface', { size: SURFACE } as never)
  const pairs = useRecord<PosteriorPairs>(live, t, 'classify/pairs', 'posteriorPairs')
  const diagnostics = useRecord<ChainDiagnostics>(live, t, 'classify/diagnostics', 'chainDiagnostics')
  const scores = useProbe(live, t, 'classify/scores').value as Record<string, number | null> | null | undefined
  const objective = classifierObjective(config)
  const status = live.session.status
  const done = status !== undefined
  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-muted" role="status">
        {live.error !== undefined
          ? live.error
          : `${live.session.t} of about ${classifierSteps(config)} steps ${
              !done ? 'computing' : status === 'stopped' ? 'stopped by the budget' : 'finished'
            }${live.producer?.kind === 'worker' ? ' (in a worker)' : ''}`}
      </p>
      <h2 className="text-sm font-semibold">What to observe</h2>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div>
          {surface === undefined ? <Waiting done={done} /> : <DecisionSurfaceView record={surface} />}
          <p className="mt-1 text-xs text-muted">
            The boundary is where the predictive probability crosses one half, and the fading colours mark where the
            posterior is unsure; the pair plot shows how the coefficients trade off against each other, which is where
            that uncertainty comes from.
          </p>
        </div>
        <div>
          {pairs === undefined ? <Waiting done={done} /> : <PosteriorPairsView record={pairs} controls={false} />}
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div>
          {diagnostics !== undefined ? (
            <ChainDiagnosticsView record={diagnostics} />
          ) : objective !== undefined ? (
            <Objective live={live} path={objective.path} label={objective.label} t={t} />
          ) : (
            <p className="text-sm text-muted">
              No curve for this method: a sampler shows its chains once it has draws, and EP and Pathfinder end in a few
              steps.
            </p>
          )}
        </div>
        <Numbers scores={scores} />
      </div>
      {player}
      <Check live={live} reference={reference} t={t} scores={scores} />
    </div>
  )
}

const Waiting = (props: { readonly done: boolean }): ReactNode => (
  <p className="text-sm text-muted">
    {props.done
      ? 'The budget ended before the first draw: raise the time budget or shorten the warm-up.'
      : 'Waiting for the first draws...'}
  </p>
)

function Numbers(props: { readonly scores: Record<string, number | null> | null | undefined }): ReactNode {
  const s = props.scores
  const rows: [string, string][] =
    s === null || s === undefined
      ? []
      : [
          ['test accuracy', num(s['testAccuracy'])],
          ['held-out log predictive', num(s['testLogPredictive'])],
          ['train accuracy', num(s['trainAccuracy'])],
          ['draws used', String(s['draws'] ?? '-')],
          ...(s['maxRhat'] === null || s['maxRhat'] === undefined
            ? []
            : ([['max R-hat', num(s['maxRhat'])]] as [string, string][])),
          ...(s['minEss'] === null || s['minEss'] === undefined
            ? []
            : ([['min ESS', num(s['minEss'], 0)]] as [string, string][])),
          ...(s['divergences'] === null || s['divergences'] === undefined
            ? []
            : ([['divergences', String(s['divergences'])]] as [string, string][])),
        ]
  return <Table title="The numbers" head={['', '']} rows={rows} />
}

function Check(props: {
  readonly live: Live
  readonly reference: Live
  readonly t: number
  readonly scores: Record<string, number | null> | null | undefined
}): ReactNode {
  const { live, reference, t } = props
  const ready = reference.done
  const rt = reference.head
  const refScores = useProbe(reference, rt, ready ? 'classify/scores' : undefined).value
  const refSurface = useProbe(reference, rt, ready ? 'classify/surface' : undefined, {
    size: REFERENCE_SIZE,
  } as never).value
  const args = refScores && refSurface ? { scores: refScores, surface: refSurface } : undefined
  const check = useProbe(live, t, args === undefined ? undefined : 'classify/check', args as never).value as
    | Record<string, number>
    | null
    | undefined
  const rows: [string, string][] =
    check === null || check === undefined
      ? []
      : [
          ['largest mean error (in reference sds)', num(check['maxMeanError'])],
          ['median spread ratio (method / reference)', num(check['medianSdRatio'])],
          ['largest boundary difference (probability)', num(check['maxSurfaceDifference'])],
          ['mean boundary difference', num(check['meanSurfaceDifference'], 4)],
          ['reference test accuracy', num(check['referenceAccuracy'])],
          ['reference held-out log predictive', num(check['referenceLogPredictive'])],
        ]
  return (
    <div className="flex flex-col gap-1">
      <h2 className="text-sm font-semibold">How we know it works</h2>
      <p className="text-xs text-muted">
        The reference is a long NUTS run (2 chains, 200 warm-up, 300 draws) of the same problem, computed in a worker of
        its own once this run has ended and kept while only the method changes. The exact posterior has no closed form,
        and a grid of it is out of reach beyond three coefficients, so the check is against the reference: the error of
        each coefficient's mean in units of the reference spread, and how far the predictive surface moves.{' '}
        {!live.done
          ? 'Waiting for the run to end.'
          : !ready
            ? 'The reference is computing...'
            : args === undefined
              ? 'The reference run did not reach its draws within its own budget (20 s): lower the degree or the rows.'
              : ''}
      </p>
      {rows.length > 0 ? <Table title="" head={['', '']} rows={rows} /> : null}
    </div>
  )
}

function Table(props: {
  readonly title: string
  readonly head: [string, string]
  readonly rows: [string, string][]
}): ReactNode {
  return (
    <div>
      {props.title === '' ? null : <h3 className="mb-1 text-xs font-semibold text-muted">{props.title}</h3>}
      <table className="text-sm">
        <tbody>
          {props.rows.map(([k, v]) => (
            <tr key={k}>
              <td className="pr-6 text-muted">{k}</td>
              <td className="tabular-nums">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Objective(props: {
  readonly live: Live
  readonly path: string
  readonly label: string
  readonly t: number
}): ReactNode {
  const { live, path, label, t } = props
  const mode = useColourMode()
  const run = useRun(live)
  const series = useMemo(() => {
    try {
      return metricSeries(run, path)
    } catch {
      return undefined
    }
  }, [run, path])
  const option = useMemo(() => {
    if (series === undefined) return undefined
    const keep = series.steps.map((s, i) => [s, series.values[i] ?? Number.NaN] as const).filter(([s]) => s <= t)
    return cartesian(
      [lines([{ x: keep.map((k) => k[0]), y: keep.map((k) => k[1]), name: label, width: 2 }], { mode })],
      {
        mode,
        figure: CURVE,
        x: { label: 'step' },
        y: { label },
      },
    )
  }, [series, t, label, mode])
  return option === undefined ? (
    <p className="text-sm text-muted">No {label} on the tape yet.</p>
  ) : (
    <Chart option={option} figure={CURVE} />
  )
}
