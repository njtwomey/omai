import { posteriorPairs } from 'omai-engine/exposition'
import { type AnyInferenceMethod, anytimeKnobs, methodConfig } from 'omai-engine/probprog/inference'
import {
  type BayesianGamFit,
  type BayesianGamReference,
  bayesianGamAgreement,
  bayesianGamMethods,
  bayesianGamNotes,
} from 'omai-engine/probprog/models'
import { choice, defaults, dict, int, real, type SpaceValue } from 'omai-engine/space'
import { tensorSeries } from 'omai-engine/trace'
import { type WorkerLike, workerProducer, workerStage } from 'omai-render/producer'
import {
  Chart,
  ControlPanel,
  DatasetPicker,
  type DatasetSelection,
  type LiveSession,
  PosteriorPairsView,
  SessionPlayer,
  SpaceControls,
  selectEntry,
  Toggle,
  useColourMode,
  useDataset,
  useJob,
  useRun,
  useScrubClick,
  useSession,
} from 'omai-render/react'
import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import { type CurveData, curveData } from '../../shared/curve-data.ts'
import { StaleVeil } from '../../shared/StaleVeil.tsx'
import { fitChart, inclusionChart, NARROW, traceChart, WIDE } from './charts.ts'
import { methodOf, nameOf } from './spec.ts'

const DATA = 'regression1d'
const MODEL = dict({
  prior: choice(['penalty', 'spikeSlab', 'ard'], {
    default: 'penalty',
    label: 'prior on the coefficients',
    descriptions: {
      penalty:
        'smoothness penalty: a Gaussian prior with the wiggliness penalty as precision, and a smoothing parameter that is itself inferred',
      ard: 'automatic relevance determination: each B-spline coefficient has a precision of its own with a gamma prior; a large one prunes the basis',
      spikeSlab:
        'spike and slab: each B-spline coefficient is in a wide slab (the basis is used) or a narrow spike at zero (it is off)',
    },
  }),
  basis: choice(['ps', 'cr', 'bs', 'tp'], {
    default: 'ps',
    label: 'spline basis',
    descriptions: {
      ps: 'P-spline: B-splines on equally spaced knots (the random-walk prior with the penalty)',
      cr: 'cubic regression spline: values at knots, the penalty is the integrated squared second derivative (penalty prior only)',
      bs: 'B-splines with knots at quantiles of x and the integrated squared derivative penalty',
      tp: 'thin-plate regression spline, a truncated eigenbasis of the thin-plate penalty (penalty prior only)',
    },
  }),
  k: int(5, 40, { default: 12, label: 'knots (basis size)' }),
  degree: int(1, 5, { default: 3, label: 'spline degree', when: { key: 'basis', in: ['ps', 'bs'] } }),
  order: int(1, 3, { default: 2, label: 'penalty order', when: { key: 'basis', in: ['ps'] } }),
  logLambdaMean: real(-6, 10, {
    default: 0,
    label: 'prior mean of log smoothing parameter',
    when: { key: 'prior', in: ['penalty'] },
  }),
  logLambdaSd: real(0.5, 6, {
    default: 3,
    label: 'prior sd of log smoothing parameter',
    when: { key: 'prior', in: ['penalty'] },
  }),
  inclusion: real(0.02, 0.98, {
    default: 0.5,
    label: 'prior probability a basis is on',
    when: { key: 'prior', in: ['spikeSlab'] },
  }),
  slabScale: real(0.1, 5, { default: 1, label: 'slab sd (in sd of y)', when: { key: 'prior', in: ['spikeSlab'] } }),
  alphaShape: real(0.1, 5, { default: 1, label: 'gamma shape of the precisions', when: { key: 'prior', in: ['ard'] } }),
  alphaRate: real(0.01, 5, {
    default: 0.5,
    label: 'gamma rate of the precisions',
    when: { key: 'prior', in: ['ard'] },
  }),
  sparsity: real(0, 1, {
    default: 1,
    label: 'sparsity (0: plain ridge, 1: full ARD)',
    when: { key: 'prior', in: ['ard'] },
  }),
  spikeRatio: real(0.005, 0.3, {
    default: 0.03,
    label: 'spike sd (fraction of slab)',
    when: { key: 'prior', in: ['spikeSlab'] },
  }),
})
type Model = {
  prior: 'penalty' | 'spikeSlab' | 'ard'
  basis: 'ps' | 'cr' | 'bs' | 'tp'
  k: number
  degree: number
  order: number
  logLambdaMean: number
  logLambdaSd: number
  inclusion: number
  slabScale: number
  spikeRatio: number
  alphaShape: number
  alphaRate: number
  sparsity: number
}
const GRID = 120
const round = (v: number): number => Math.round(v * 1e6) / 1e6
const PRODUCER = workerProducer({
  createWorker: () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as WorkerLike,
})
const BUDGET = dict({
  budgetMs: { ...anytimeKnobs.spaces['budgetMs'], default: 2000, label: 'time budget' },
  budgetSteps: { ...anytimeKnobs.spaces['budgetSteps'], label: "step budget (0: the method's own length)" },
} as never)
type Budget = { budgetMs: number; budgetSteps: number }
const START = {
  data: undefined,
  model: defaults(MODEL) as unknown as Model,
  method: defaults(bayesianGamMethods().space) as Record<string, SpaceValue>,
  budget: defaults(BUDGET) as unknown as Budget,
}

export default function BayesianGamPage(): ReactNode {
  const [selection, setSelection] = useState<DatasetSelection>(() => selectEntry(DATA, 'sineCurve'))
  const [model, setModel] = useState<Model>(START.model)
  const [method, setMethod] = useState(START.method)
  const [budget, setBudget] = useState<Budget>(START.budget)
  // Every method of the library that runs on the model (the verdicts are of the prior chosen), and why the others do not.
  const offer = useMemo(() => bayesianGamMethods(model.prior), [model.prior])
  const generated = useDataset(DATA, selection)
  const curve = useMemo(() => curveData(generated.data, generated.truth as never), [generated.data, generated.truth])
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 items-start gap-3 md:grid-cols-2">
        <ControlPanel title="the experiment: data and model" columns={1}>
          <DatasetPicker collection={DATA} value={selection} onChange={setSelection} />
          <SpaceControls
            space={MODEL}
            value={model}
            onChange={(v) => setModel(v as unknown as Model)}
            segmentedUpTo={0}
            admissible={(path, option) =>
              path[0] !== 'basis' || model.prior === 'penalty' || option === 'ps' || option === 'bs'
            }
            why={() => 'this prior needs a local basis: ps or bs'}
          />
        </ControlPanel>
        <ControlPanel title="the inference: one method, its knobs and the budget" columns={1}>
          <SpaceControls
            space={offer.space}
            value={method}
            onChange={(v) => setMethod(v as unknown as Record<string, SpaceValue>)}
            segmentedUpTo={0}
            admissible={(_, option) => !(String(option) in offer.unavailable)}
            why={(_, option) => offer.unavailable[String(option)]}
          />
          <SpaceControls space={BUDGET} value={budget} onChange={(v) => setBudget(v as unknown as Budget)} />
          <details className="text-xs text-muted">
            <summary className="cursor-pointer">
              Methods that cannot run on this model, and what is known of the others
            </summary>
            <ul className="mt-1 flex flex-col gap-1">
              {Object.entries({ ...offer.unavailable, ...bayesianGamNotes }).map(([kind, why]) => (
                <li key={kind}>
                  <b>{kind}</b>: {why}
                </li>
              ))}
            </ul>
          </details>
        </ControlPanel>
      </div>
      {curve === undefined ? (
        <p className="text-sm text-muted">This page needs a one-dimensional regression data set.</p>
      ) : (
        <Fit curve={curve} model={model} method={method} budget={budget} />
      )}
    </div>
  )
}

function Fit(props: {
  readonly curve: CurveData
  readonly model: Model
  readonly method: Record<string, SpaceValue>
  readonly budget: Budget
}): ReactNode {
  const { curve, model, method, budget } = props
  const [at, setAt] = useState<number | null>(null)
  const { kind, config } = methodConfig(method)
  const configKey = JSON.stringify(config)
  const options = useMemo(
    () => ({
      prior: model.prior,
      basis: model.prior !== 'penalty' && model.basis !== 'bs' ? 'ps' : model.basis,
      k: model.k,
      degree: model.degree,
      order: model.order,
      logLambdaPrior: { mean: model.logLambdaMean, sd: model.logLambdaSd },
      inclusion: model.inclusion,
      slabScale: model.slabScale,
      spikeRatio: model.spikeRatio,
      alphaShape: model.alphaShape,
      alphaRate: model.alphaRate,
      sparsity: model.sparsity,
    }),
    [model],
  )
  const rows = useMemo(() => ({ x: curve.x.map(round), y: curve.y.map(round) }), [curve])
  // The method is described by its name, which a worker rebuilds it from (see spec.ts).
  const name = useMemo(() => nameOf({ kind, ...rows, options, fit: { gridSize: GRID } }), [kind, rows, options])
  const inference = useMemo(() => methodOf(name) as AnyInferenceMethod, [name])
  const steps =
    budget.budgetSteps > 0
      ? budget.budgetSteps
      : 'numDraws' in config
        ? Number(config['numWarmup'] ?? 0) + Number(config['numDraws']) + 10
        : 3000
  // The time budget: the worker is stopped when it is spent (what is on the tape stays). It is counted from the first
  // step, so that the worker starting up (loading the engine) is not charged to the method.
  const [spent, setSpent] = useState(false)
  const [started, setStarted] = useState(false)
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new problem, method or budget is a new run
  useEffect(() => {
    setSpent(false)
    setStarted(false)
  }, [name, configKey, budget.budgetMs, steps])
  // biome-ignore lint/correctness/useExhaustiveDependencies: the clock starts once, at the first step
  useEffect(() => {
    if (!started || budget.budgetMs <= 0) return
    const timer = setTimeout(() => setSpent(true), budget.budgetMs)
    return () => clearTimeout(timer)
  }, [started, name, configKey, budget.budgetMs, steps])
  const live = useSession(inference, { config: config as never, steps }, [inference, configKey, steps], {
    start: 'end',
    active: !spent,
    producer: PRODUCER,
  })
  // The exact posterior of the problem, in its own worker: it does not wait for the run, nor the run for it.
  const stages = useMemo(
    () => [
      workerStage<undefined, BayesianGamReference>({
        name: 'exact posterior',
        createWorker: () => new Worker(new URL('./reference.worker.ts', import.meta.url), { type: 'module' }),
        input: () => ({ ...rows, options, fit: { gridSize: GRID } }),
      }),
    ],
    [rows, options],
  )
  useEffect(() => {
    if (live.head >= 1) setStarted(true)
  }, [live.head])
  // A run the budget stopped leaves the cursor at the start: show where it got to.
  useEffect(() => {
    if (spent && live.head > 0) setAt(null)
  }, [spent, live.head])
  // The exact posterior exists for the penalty prior only: with spike and slab the indicators make it a sum over subsets.
  const exact = options.prior === 'penalty'
  const job = useJob({ stages, inputs: [stages], active: exact })
  const reference = exact ? (job.result as BayesianGamReference | undefined) : undefined
  const scale = useMemo(() => {
    const m = curve.y.reduce((a, v) => a + v, 0) / curve.y.length
    return Math.sqrt(curve.y.reduce((a, v) => a + (v - m) ** 2, 0) / curve.y.length) || 1
  }, [curve])
  const reset = (): void => setAt(null)
  return (
    <div className="flex flex-col gap-3">
      <StaleVeil stale={live.stale}>
        <View
          live={live}
          at={at}
          curve={curve}
          scale={scale}
          reference={reference}
          kind={kind}
          spikeSlab={!exact}
          onScrub={setAt}
        />
      </StaleVeil>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={reset} />
      <p className="max-w-3xl text-xs text-muted">
        What to observe: the credible band is narrow where there are data and opens up where there are few (the ends,
        gaps); the smoothing parameter is not set but inferred, so the band includes the doubt about how wiggly the
        curve is. Scrub the player to see the posterior at an earlier step of the run.
      </p>
    </div>
  )
}

function View(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the session is generic over the method
  readonly live: LiveSession<any, any>
  readonly at: number | null
  readonly curve: CurveData
  readonly scale: number
  readonly reference: BayesianGamReference | undefined
  readonly kind: string
  readonly spikeSlab: boolean
  readonly onScrub: (t: number | null) => void
}): ReactNode {
  const { live, curve, scale, reference, kind, spikeSlab } = props
  const [showBases, setShowBases] = useState(true)
  const mode = useColourMode()
  const run = useRun(live)
  const wanted = Math.max(0, Math.min(props.at ?? live.head, live.head))
  // A posterior costs the worker as much as a few steps of the run: while the run goes on and the view follows its head, it is
  // asked for about twice a second, not at every step; scrubbing or a finished run asks at once.
  const [t, setT] = useState(wanted)
  const latest = useRef(wanted)
  latest.current = wanted
  const following = props.at === null && live.computing
  useEffect(() => {
    if (!following) setT(wanted)
  }, [following, wanted])
  useEffect(() => {
    if (!following) return
    const timer = setInterval(() => setT(latest.current), 500)
    return () => clearInterval(timer)
  }, [following])
  // A worker answers a probe a moment later: the last answer stays until the one for this step is there.
  // biome-ignore lint/correctness/useExhaustiveDependencies: the version moves when an answer arrives
  const probed = useMemo(() => {
    try {
      return { value: live.session.probe(t, 'gam/fit') }
    } catch {
      return {}
    }
  }, [live.session, live.version, t])
  const fit = (probed.value ?? undefined) as BayesianGamFit | undefined
  const trace = useMemo(() => {
    const path = kind === 'svi' ? 'svi/elbo' : kind === 'pathfinder' ? 'pathfinder/elbo' : 'mcmc/logDensity'
    if (path === 'mcmc/logDensity') {
      const ev = tensorSeries(run, path)
      return {
        label: 'log density',
        steps: ev.map((e) => e.step),
        values: ev.map(
          (e) =>
            (e.values ?? [e.summary?.mean ?? Number.NaN]).reduce((a, b) => a + b, 0) /
            Math.max(1, (e.values ?? [0]).length),
        ),
      }
    }
    const ev = run.events.filter((e) => e.type === 'metric' && e.name === path) as { step: number; value: number }[]
    return { label: 'ELBO', steps: ev.map((e) => e.step), values: ev.map((e) => e.value) }
  }, [run, kind])
  const steps = useMemo(() => Array.from({ length: live.head + 1 }, (_, i) => i), [live.head])
  const seek = useScrubClick(steps, live.head, props.onScrub)
  const truth = useMemo(() => curve.truthAt?.(fit?.grid ? [...fit.grid] : []), [curve, fit?.grid])
  const fitted = useMemo(
    () => fitChart({ fit, curve, truth, reference, bases: showBases, mode }),
    [fit, curve, truth, reference, showBases, mode],
  )
  const loss = useMemo(() => traceChart(trace, t, live.head, mode), [trace, t, mode, live.head])
  const inclusion = useMemo(
    () => (fit?.inclusion === undefined ? undefined : inclusionChart(fit.inclusion, mode)),
    [fit, mode],
  )
  const pairs = useMemo(
    () =>
      fit === undefined
        ? undefined
        : posteriorPairs([{ name: kind, draws: fit.draws }], {
            frames: 2,
            ...(reference === undefined
              ? {}
              : {
                  truth: { logLambda: reference.hyper.logLambda[0], logSigma: reference.hyper.logSigma[0] },
                  referenceLabel: 'exact mean',
                }),
          }),
    [fit, kind, reference],
  )
  const agreement =
    fit !== undefined && reference !== undefined ? bayesianGamAgreement(fit, reference, scale) : undefined
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-1 gap-x-4 gap-y-2 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Chart option={fitted} figure={WIDE} />
        <Chart option={loss} figure={NARROW} onInit={seek} />
      </div>
      <Toggle
        label="show the basis functions, weighted by their coefficients"
        value={showBases}
        onChange={setShowBases}
      />
      {inclusion === undefined ? null : <Chart option={inclusion} figure={NARROW} />}
      {pairs === undefined ? (
        <p className="text-sm text-muted">
          No draws yet (step {t}: a sampler warms up first). If the time budget ends before the first draws, give it
          more.
        </p>
      ) : (
        <PosteriorPairsView record={pairs} controls={false} />
      )}
      <div className="max-w-3xl text-xs text-muted">
        <b>How we know it works.</b>{' '}
        {spikeSlab
          ? 'With this prior there is no closed form to check against (the indicators make the posterior a sum over subsets of the bases), so there is no reference band: read the bars, which should fall where the data have structure (the sampled curves and the bars agree across methods, and across seeds of one).'
          : 'Given the smoothing parameter and the noise the coefficients are Gaussian in closed form, so quadrature over those two gives the exact posterior of the curve (dotted band, cross on the pair plot). '}
        {agreement === undefined
          ? spikeSlab
            ? ''
            : 'Computing the reference…'
          : `At this step ${kind} is ${agreement.meanError.toFixed(3)} response sd from the exact mean curve, its band is ${agreement.bandRatio.toFixed(2)} times the exact width, and its mean log smoothing parameter is ${agreement.logLambdaShift >= 0 ? '+' : ''}${agreement.logLambdaShift.toFixed(2)} from the exact one.`}
      </div>
    </div>
  )
}
