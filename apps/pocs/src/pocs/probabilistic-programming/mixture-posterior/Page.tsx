import { band, cartesian, histogram, lines } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import type { StageWorker } from 'omai-render/producer'
import {
  ChainDiagnosticsView,
  Choice,
  ControlPanel,
  Player,
  PosteriorPairsView,
  ResponsiveChart,
  SeedInput,
  useColourMode,
  usePlayback,
} from 'omai-render/react'
import { type ReactNode, useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'
import { CurveChart } from '../../../shared/CurveChart.tsx'
import { type PoolHandle, type PoolState, runPool, type StatusLine } from '../../../shared/workerPool.ts'
import { jobId, METHODS, type MixtureRequest, type MixtureResult, waitingTimes } from './compute.ts'
import { type Coordinates, chainsAt, coordinatesOf, fitTrace, pairsOf, predictiveAt } from './shown.ts'

const BUDGETS = [10, 30, 60] as const
const TICKS = 101
const GRID = Array.from({ length: 141 }, (_, i) => 35 + i * 0.5)

const button =
  'cursor-pointer rounded-md border border-line bg-bg px-3 py-1.5 text-xs text-fg hover:bg-surface disabled:cursor-default disabled:opacity-50'
const SECTION = 'flex min-w-0 flex-col gap-2'
const HEADING = 'text-xs font-medium uppercase tracking-wide text-muted'
const phaseText: Record<StatusLine['phase'], string> = {
  queued: 'queued',
  running: 'running',
  finishing: 'preparing the playback',
  done: 'done',
  stopped: 'not started',
  failed: 'failed',
}
const endingText: Record<string, string> = {
  budget: 'budget spent',
  completed: 'ran its steps',
  stopped: 'stopped',
  failed: 'failed',
}

const makeWorker = (): StageWorker =>
  new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as StageWorker

type Labelling = 'unordered' | 'ordered' | 'both'

/** The posterior predictive density of one job at the playhead over the histogram of the data, with the latest draw's components. */
function Density(props: {
  readonly result: MixtureResult
  readonly t: number
  readonly minutes: readonly number[]
}): ReactNode {
  const { result, t, minutes } = props
  const mode = useColourMode()
  const predictive = useMemo(() => predictiveAt(result, GRID, t), [result, t])
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          histogram(minutes, { bins: 20, range: [40, 100], density: true, colour: 'tab7', opacity: 0.45, mode }),
          ...(predictive === undefined
            ? []
            : [
                band([{ x: GRID, lower: predictive.lower, upper: predictive.upper, colour: 'tab0', opacity: 0.3 }], {
                  mode,
                }),
                lines(
                  [
                    ...predictive.components.map((c, k) => ({
                      x: GRID,
                      y: c,
                      name: `component ${k + 1}`,
                      colour: `tab${k + 1}`,
                      width: 1.5,
                      dash: 'dashed' as const,
                    })),
                    { x: GRID, y: predictive.mean, name: 'predictive mean', colour: 'tab0', width: 2.5 },
                  ],
                  { mode },
                ),
              ]),
        ],
        {
          mode,
          figure,
          tooltip: 'axis',
          x: { label: 'waiting time (minutes)', limits: [35, 105] },
          y: { label: 'density', limits: [0, 0.075] },
        },
      ),
    [minutes, predictive, mode],
  )
  return (
    <div className={SECTION}>
      <div className="flex justify-between text-xs text-muted">
        <span className="text-fg">{result.label}</span>
        <span className="tabular-nums">{predictive?.draws ?? 0} draws</span>
      </div>
      <ResponsiveChart aspect={1.6} narrowAspect={1.3} build={build} />
    </div>
  )
}

export default function MixturePage(): ReactNode {
  const [components, setComponents] = useState(2)
  const [method, setMethod] = useState('nuts')
  const [labelling, setLabelling] = useState<Labelling>('both')
  const [seed, setSeed] = useState(1)
  const [budget, setBudget] = useState<number>(10)
  const [pool, setPool] = useState<PoolState<MixtureResult> | undefined>(undefined)
  const handle = useRef<PoolHandle | undefined>(undefined)
  const [now, setNow] = useState(() => performance.now())
  const running = pool?.running === true
  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => setNow(performance.now()), 100)
    return () => clearInterval(timer)
  }, [running])
  useEffect(() => () => handle.current?.cancel(), [])

  const spec = METHODS.find((m) => m.id === method) ?? (METHODS[0] as (typeof METHODS)[number])
  const start = (): void => {
    handle.current?.cancel()
    const orders = labelling === 'both' ? [false, true] : [labelling === 'ordered']
    const requests: MixtureRequest[] = orders.map((ordered) => ({
      method: jobId(spec.kind, ordered),
      kind: spec.kind,
      ordered,
      components,
      budgetMs: budget * 1000,
      seed,
    }))
    setPool(undefined)
    handle.current = runPool(
      requests,
      Object.fromEntries(requests.map((r) => [r.method, `${spec.label}, ${r.ordered ? 'ordered' : 'unordered'}`])),
      {
        ...(typeof Worker === 'undefined' ? {} : { createWorker: makeWorker }),
        run: async (request, host) => (await import('./compute.ts')).runMixture(request, host),
        onChange: setPool,
      },
    )
  }
  const results = useMemo(
    () => (pool === undefined ? [] : Object.values(pool.results).sort((a, b) => Number(a.ordered) - Number(b.ordered))),
    [pool],
  )
  const finished = pool !== undefined && !pool.running && results.length > 0
  const elapsed = pool === undefined ? 0 : ((pool.endedAt ?? now) - pool.startedAt) / 1000
  const share =
    pool === undefined || pool.lines.length === 0
      ? 0
      : Math.min(1, (elapsed * 1000) / (budget * 1000 * Math.ceil(pool.lines.length / Math.max(1, pool.workers))))

  return (
    <div className="flex flex-col gap-4">
      <p className="m-0 max-w-3xl text-sm text-muted">
        <b className="font-medium text-fg">Label switching</b> is the posterior's symmetry under renaming the
        components: swapping two means and their weights leaves the likelihood unchanged, so a free mixture of K
        components has K! equal modes. A chain sits in one of them and different chains in different ones, so the raw
        draws of one component's mean mix up all the components.
      </p>
      <div className="flex flex-col gap-3 rounded-lg border border-line bg-surface p-3">
        <ControlPanel columns={5} title="Problem">
          <Choice
            label="components"
            value={components}
            options={[2, 3, 4].map((k) => ({ value: k, label: String(k) }))}
            onChange={setComponents}
          />
          <Choice
            label="method"
            value={method}
            options={METHODS.map((m) => ({
              value: m.id,
              label: m.id === 'svi-mixture' ? 'SVI mix' : m.id === 'svi' ? 'SVI' : m.label,
            }))}
            onChange={setMethod}
          />
          <Choice
            label="means"
            value={labelling}
            options={[
              { value: 'both', label: 'both' },
              { value: 'unordered', label: 'free' },
              { value: 'ordered', label: 'ordered' },
            ]}
            onChange={setLabelling}
          />
          <SeedInput value={seed} onChange={setSeed} />
          <Choice
            label="time budget"
            value={budget}
            options={BUDGETS.map((b) => ({ value: b, label: `${b} s` }))}
            onChange={setBudget}
          />
        </ControlPanel>
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex gap-2">
            <button type="button" className={button} disabled={running} onClick={start}>
              {pool === undefined ? 'Run' : 'Run again'}
            </button>
            <button type="button" className={button} disabled={!running} onClick={() => handle.current?.stop()}>
              Stop
            </button>
          </div>
          <div className="flex min-w-40 flex-1 flex-col gap-1">
            <div className="flex justify-between text-xs tabular-nums text-muted">
              <span>{pool === undefined ? 'ready' : running ? 'running' : 'finished'}</span>
              <span>
                {elapsed.toFixed(1)} s · {budget} s budget
                {pool === undefined || pool.workers === 0
                  ? ''
                  : ` · ${pool.workers} worker${pool.workers === 1 ? '' : 's'}`}
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-bg">
              <div
                className="h-full rounded-full bg-(--wb-accent)"
                style={{ width: `${100 * (pool === undefined ? 0 : running ? share : 1)}%` }}
              />
            </div>
          </div>
        </div>
        {pool === undefined ? null : (
          <ul className="m-0 grid list-none gap-x-6 gap-y-1 p-0 text-xs text-muted sm:grid-cols-2">
            {pool.lines.map((l) => (
              <li key={l.id} className="flex justify-between gap-3 tabular-nums">
                <span className="text-fg">{l.label}</span>
                <span title={l.error}>
                  {phaseText[l.phase]}
                  {l.step > 0 && l.phase !== 'queued' ? ` · ${l.step} steps` : ''}
                  {l.phase === 'running' ? ` · ${(l.elapsedMs / 1000).toFixed(1)} s` : ''}
                  {l.error === undefined ? '' : `: ${l.error.slice(0, 60)}`}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
      {finished ? <Outputs key={pool.startedAt} results={results} /> : null}
      {pool === undefined ? (
        <p className="text-sm text-muted">
          Choose the number of components, the method and whether the means are ordered, then press Run. Nothing is
          drawn while the workers run.
        </p>
      ) : null}
    </div>
  )
}

/** What the run produced, drawn at the playhead of one player. */
function Outputs(props: { readonly results: readonly MixtureResult[] }): ReactNode {
  const { results } = props
  const minutes = useMemo(() => waitingTimes().minutes, [])
  const total = Math.max(1, ...results.map((r) => r.elapsedMs))
  const playback = usePlayback(TICKS, { fps: 12 })
  const seek = useRef(playback.seek)
  seek.current = playback.seek
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new set of results opens on its last frame
  useEffect(() => seek.current(TICKS - 1), [results])
  const t = useDeferredValue((playback.frame / (TICKS - 1)) * total)
  const [coordinates, setCoordinates] = useState<Coordinates>('means')
  const [focus, setFocus] = useState(results[0]?.method ?? '')
  const focused = results.find((r) => r.method === focus) ?? (results[0] as MixtureResult)
  const pairs = useMemo(() => pairsOf(results, coordinates), [results, coordinates])
  const diagnostics = useMemo(() => chainsAt(focused, coordinates, t), [focused, coordinates, t])
  const trace = useMemo(
    () =>
      coordinatesOf(coordinates, focused.components).map((j) => ({
        name: focused.names[j] as string,
        j,
        ...fitTrace(focused, j, t),
      })),
    [focused, coordinates, t],
  )
  const traceCurves = useMemo(
    () => trace.map((c, i) => ({ name: c.name, colour: `tab${i}`, width: 2, x: c.x, y: c.y })),
    [trace],
  )
  const traceX = useMemo(() => ({ label: 'seconds', limits: [0, total / 1000] as [number, number] }), [total])
  const traceY = useMemo(() => ({ label: 'mean of the fit' }), [])
  const notes = results.filter((r) => r.message !== undefined)

  return (
    <div className="flex flex-col gap-4">
      <div className="sticky top-11 z-10 flex flex-col gap-1 rounded-lg border border-line bg-surface p-2">
        <Player
          controller={playback}
          stamp={(f) => `${((f / (TICKS - 1)) * (total / 1000)).toFixed(1)} s of ${(total / 1000).toFixed(1)} s`}
          fpsChoices={[2, 5, 12, 24, 60]}
          size="sm"
        />
      </div>
      {notes.map((r) => (
        <p key={r.method} className="m-0 text-sm text-muted">
          {r.label}: {r.message}.
        </p>
      ))}
      <div className={SECTION}>
        <div className={HEADING}>
          The data, the predictive density of the draws so far (mean and 90% band) and the components of the latest draw
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          {results.map((r) => (
            <Density key={r.method} result={r} t={t} minutes={minutes} />
          ))}
        </div>
      </div>
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <div className={SECTION}>
          <div className={HEADING}>Pair plot of the draws</div>
          <ControlPanel columns={2}>
            <Choice
              label="coordinates"
              value={coordinates}
              options={[
                { value: 'means', label: 'the component means' },
                { value: 'weights', label: 'the component weights' },
                { value: 'both', label: 'means and weights' },
              ]}
              onChange={setCoordinates}
            />
          </ControlPanel>
          {pairs === undefined ? (
            <p className="text-sm text-muted">No draws to show.</p>
          ) : (
            <PosteriorPairsView record={pairs} cursor={t} controls={false} filename="mixture-pairs" />
          )}
          <p className="m-0 text-xs text-muted">
            With free means each mode appears twice, once for each naming of the components; ordered, only the one with
            the smaller mean first.
          </p>
        </div>
        <div className={SECTION}>
          <div className={HEADING}>Trace of the chosen coordinates, R-hat and ESS</div>
          <ControlPanel columns={2}>
            <Choice
              label="job"
              value={focused.method}
              options={results.map((r) => ({ value: r.method, label: r.label }))}
              onChange={setFocus}
            />
          </ControlPanel>
          <p className="m-0 text-xs text-muted">
            {focused.label}: {endingText[focused.ending] ?? focused.ending}, {focused.steps} steps in{' '}
            {(focused.elapsedMs / 1000).toFixed(1)} s{focused.chains > 1 ? `, ${focused.chains} chains` : ''}.
          </p>
          {focused.organisation === 'chain' ? (
            diagnostics === undefined ? (
              <p className="text-sm text-muted">The chains have too few draws to diagnose at this point.</p>
            ) : (
              <ChainDiagnosticsView record={diagnostics} filename="mixture-chains" />
            )
          ) : (
            <>
              <CurveChart curves={traceCurves} x={traceX} y={traceY} />
              <p className="m-0 text-xs text-muted">
                A variational fit has one distribution rather than chains, so there is no R-hat: the trace is the fit's
                mean at each snapshot.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
