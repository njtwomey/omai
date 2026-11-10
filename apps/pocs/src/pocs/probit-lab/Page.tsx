import { polynomialFeatureMap } from 'omai-engine/probprog/models'
import type { StageWorker } from 'omai-render/producer'
import {
  ChainDiagnosticsView,
  Choice,
  ControlPanel,
  NumberField,
  Player,
  PosteriorComparisonView,
  SeedInput,
  Toggle,
  usePlayback,
} from 'omai-render/react'
import { type ReactNode, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'
import { CurveChart } from '../../shared/CurveChart.tsx'
import { Boundary } from './Boundary.tsx'
import { DEFAULT_METHODS, METHODS } from './methods.ts'
import { type PairMethod, PairPlots } from './PairPlots.tsx'
import { type PoolHandle, type PoolState, runPool, type StatusLine } from './pool.ts'
import { DATASETS, makeProblem } from './problem.ts'
import { CoefficientTable, MethodTable } from './Results.tsx'
import { chainsAt, colourOf, column, frameAt, gridAt, marginalsAt, visibleRows } from './shown.ts'
import type { DatasetId, MethodResult, Problem, ProblemSettings } from './types.ts'

const BUDGETS = [3, 10, 30, 60] as const
const TICKS = 101
const GRID = 28
const MAX_COORDINATES = 6

const button =
  'cursor-pointer rounded-md border border-line bg-bg px-3 py-1.5 text-xs text-fg hover:bg-surface disabled:cursor-default disabled:opacity-50'

const phaseText: Record<StatusLine['phase'], string> = {
  queued: 'queued',
  running: 'running',
  finishing: 'preparing the playback',
  done: 'done',
  stopped: 'not started',
  failed: 'failed',
}

const makeWorker = (): StageWorker =>
  new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as StageWorker

/** What a run was made for, kept with its results so changing the controls afterwards leaves what is shown alone. */
interface Shown {
  readonly problem: Problem
  readonly labels: readonly string[]
  readonly budgetMs: number
}

const SECTION = 'flex min-w-0 flex-col gap-2'
const HEADING = 'text-xs font-medium uppercase tracking-wide text-muted'

export default function ProbitLab(): ReactNode {
  const [settings, setSettings] = useState<ProblemSettings>({
    dataset: 'moons',
    rows: 200,
    noise: 0.2,
    seed: 1,
    degree: 3,
    priorScale: 1,
  })
  const [budget, setBudget] = useState<number>(10)
  const [chosen, setChosen] = useState<ReadonlySet<string>>(new Set(DEFAULT_METHODS))
  const [pool, setPool] = useState<PoolState | undefined>(undefined)
  const [shown, setShown] = useState<Shown | undefined>(undefined)
  const handle = useRef<PoolHandle | undefined>(undefined)
  const [now, setNow] = useState(() => performance.now())
  const running = pool?.running === true
  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => setNow(performance.now()), 100)
    return () => clearInterval(timer)
  }, [running])
  useEffect(() => () => handle.current?.cancel(), [])

  const dataset = DATASETS.find((d) => d.id === settings.dataset) ?? (DATASETS[0] as (typeof DATASETS)[number])
  const start = (): void => {
    handle.current?.cancel()
    const problem = makeProblem(settings)
    const picked = METHODS.filter((m) => chosen.has(m.id))
    if (picked.length === 0) return
    const labels = polynomialFeatureMap(problem.train.x, settings.degree).names
    const budgetMs = budget * 1000
    setShown({ problem, labels, budgetMs })
    setPool(undefined)
    handle.current = runPool(
      picked.map((m) => ({ problem, method: m.id, budgetMs, seed: settings.seed, gridSize: GRID })),
      Object.fromEntries(picked.map((m) => [m.id, m.label])),
      { ...(typeof Worker === 'undefined' ? {} : { createWorker: makeWorker }), onChange: setPool },
    )
  }
  const set = <K extends keyof ProblemSettings>(key: K, value: ProblemSettings[K]): void =>
    setSettings((s) => ({ ...s, [key]: value }))

  const results: MethodResult[] = useMemo(
    () =>
      pool === undefined
        ? []
        : METHODS.flatMap((m) => (pool.results[m.id] === undefined ? [] : [pool.results[m.id] as MethodResult])),
    [pool],
  )
  const finished = pool !== undefined && !pool.running && results.length > 0 && shown !== undefined

  // The status of a run in progress: the clock, a bar over all the methods' budgets, a line for each.
  const elapsed = pool === undefined ? 0 : ((pool.endedAt ?? now) - pool.startedAt) / 1000
  // The bar counts the clock against the time the run can take: the budget for each wave of methods the workers go through.
  const share =
    pool === undefined || shown === undefined || pool.lines.length === 0
      ? 0
      : Math.min(1, (elapsed * 1000) / (shown.budgetMs * Math.ceil(pool.lines.length / Math.max(1, pool.workers))))

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 rounded-lg border border-line bg-surface p-3">
        <ControlPanel columns={4} title="Problem">
          <Choice
            label="dataset"
            value={settings.dataset}
            options={DATASETS.map((d) => ({ value: d.id, label: d.label }))}
            onChange={(v: DatasetId) =>
              setSettings((s) => ({ ...s, dataset: v, noise: (DATASETS.find((d) => d.id === v) ?? dataset).noise[2] }))
            }
          />
          <NumberField
            label="rows"
            value={settings.rows}
            min={40}
            max={1000}
            integer
            onChange={(v) => set('rows', v)}
          />
          <NumberField
            label="noise"
            value={settings.noise}
            min={dataset.noise[0]}
            max={dataset.noise[1]}
            onChange={(v) => set('noise', v)}
          />
          <SeedInput value={settings.seed} onChange={(v) => set('seed', v)} />
          <Choice
            label="polynomial degree"
            value={settings.degree}
            options={[1, 2, 3, 4, 5, 6].map((d) => ({
              value: d,
              label: `${d} (${((d + 1) * (d + 2)) / 2} coefficients)`,
            }))}
            onChange={(v) => set('degree', v)}
          />
          <NumberField
            label="prior scale"
            value={settings.priorScale}
            min={0.1}
            max={10}
            log
            onChange={(v) => set('priorScale', v)}
          />
          <Choice
            label="time budget"
            value={budget}
            options={BUDGETS.map((b) => ({ value: b, label: `${b} s` }))}
            onChange={setBudget}
          />
        </ControlPanel>
        <div className="flex flex-col gap-1">
          <div className={HEADING}>Methods</div>
          <div className="grid gap-x-4 gap-y-0 sm:grid-cols-2 lg:grid-cols-4">
            {METHODS.map((m) => (
              <Toggle
                key={m.id}
                label={m.label}
                description={m.note}
                value={chosen.has(m.id)}
                disabled={running}
                onChange={(on) =>
                  setChosen((old) => {
                    const next = new Set(old)
                    if (on) next.add(m.id)
                    else next.delete(m.id)
                    return next
                  })
                }
              />
            ))}
          </div>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex gap-2">
            <button type="button" className={button} disabled={running || chosen.size === 0} onClick={start}>
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
                {elapsed.toFixed(1)} s · {budget} s for each method
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
          <ul className="m-0 grid list-none gap-x-6 gap-y-1 p-0 text-xs text-muted sm:grid-cols-2 lg:grid-cols-3">
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
      {finished ? <Outputs key={pool.startedAt} results={results} shown={shown} /> : null}
      {pool === undefined ? (
        <p className="text-sm text-muted">
          Tick the methods, set the budget and press Run. The methods compute in Web Workers and nothing is drawn while
          they do: when the budget is spent the pair plots, the decision boundaries, the marginals and the diagnostics
          appear, on one playhead.
        </p>
      ) : null}
    </div>
  )
}

/** What the run produced, drawn at the playhead of one player. */
function Outputs(props: { readonly results: readonly MethodResult[]; readonly shown: Shown }): ReactNode {
  const { results, shown } = props
  const { problem, labels } = shown
  const total = Math.max(1, ...results.map((r) => r.elapsedMs))
  const playback = usePlayback(TICKS, { fps: 12 })
  // A new run opens on its end (the answer); Play runs it from the start.
  const seek = useRef(playback.seek)
  seek.current = playback.seek
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new set of results opens on its last frame
  useEffect(() => seek.current(TICKS - 1), [results])
  const t = useDeferredValue((playback.frame / (TICKS - 1)) * total)

  const p = labels.length
  const [coords, setCoords] = useState<number[]>(() => Array.from({ length: Math.min(4, p) }, (_, i) => i))
  const [which, setWhich] = useState<string>('all')
  const [focus, setFocus] = useState<string>(results[0]?.id ?? '')
  const focusResult = results.find((r) => r.id === focus) ?? (results[0] as MethodResult)
  const overlaid = which === 'all' ? results : results.filter((r) => r.id === which)

  const pairs: PairMethod[] = useMemo(
    () =>
      overlaid.map((r) => {
        const rows = visibleRows(r, t)
        return {
          name: r.label,
          colour: colourOf(r.id),
          columns: Object.fromEntries(coords.map((j) => [j, column(r, rows, j)])),
        }
      }),
    [overlaid, coords, t],
  )
  const comparison = useMemo(() => marginalsAt(overlaid, coords.slice(0, 4), t), [overlaid, coords, t])
  const diagnostics = useMemo(() => chainsAt(focusResult, coords, t), [focusResult, coords, t])
  const curve = focusResult.curve
  const curveDots = useMemo(() => {
    if (curve === undefined) return []
    const step = focusResult.frames[frameAt(focusResult, t)]?.step ?? 0
    const keep = Array.from(curve.steps, (s, i) => (s <= step ? i : -1)).filter((i) => i >= 0)
    return [
      {
        name: curve.label,
        colour: `tab${colourOf(focusResult.id)}`,
        width: 2,
        x: keep.map((i) => curve.steps[i] as number),
        y: keep.map((i) => curve.values[i] as number),
      },
    ]
  }, [curve, focusResult, t])
  const curveX = useMemo(
    () => ({ label: 'step', limits: [0, Math.max(1, ...(curve?.steps ?? [1]))] as [number, number] }),
    [curve],
  )
  const curveY = useMemo(() => ({ label: curve?.label ?? '' }), [curve])

  const selectable = Array.from({ length: p }, (_, j) => j)
  return (
    <div className="flex flex-col gap-4">
      <div className={SECTION}>
        <div className={HEADING}>The methods</div>
        <MethodTable results={results} />
      </div>
      <div className="sticky top-11 z-10 flex flex-col gap-1 rounded-lg border border-line bg-surface p-2">
        <Player
          controller={playback}
          stamp={(f) => `${((f / (TICKS - 1)) * (total / 1000)).toFixed(1)} s of ${(total / 1000).toFixed(1)} s`}
          fpsChoices={[2, 5, 12, 24, 60]}
          size="sm"
        />
      </div>
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <div className={SECTION}>
          <div className={HEADING}>Pair plots of the posterior draws</div>
          <ControlPanel columns={2}>
            <Choice
              label="methods"
              value={which}
              options={[
                { value: 'all', label: 'all overlaid' },
                ...results.map((r) => ({ value: r.id, label: r.label })),
              ]}
              onChange={setWhich}
            />
          </ControlPanel>
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
            {overlaid.map((r) => (
              <span key={r.id} className="flex items-center gap-1 text-muted">
                <span
                  className="inline-block size-2.5 rounded-full"
                  style={{ background: `var(--wb-tab${colourOf(r.id)}, currentColor)` }}
                />
                {r.label}
                <span className="tabular-nums">({visibleRows(r, t).length} draws)</span>
              </span>
            ))}
          </div>
          <div className="flex flex-wrap gap-x-3 gap-y-0.5">
            {selectable.map((j) => (
              <label key={j} className="flex cursor-pointer items-center gap-1 text-xs text-muted">
                <input
                  type="checkbox"
                  checked={coords.includes(j)}
                  disabled={!coords.includes(j) && coords.length >= MAX_COORDINATES}
                  onChange={(e) =>
                    setCoords((old) =>
                      e.target.checked ? [...old, j].sort((a, b) => a - b) : old.filter((x) => x !== j),
                    )
                  }
                />
                w{j} <span className="opacity-70">{labels[j]}</span>
              </label>
            ))}
          </div>
          {coords.length < 2 ? (
            <p className="text-sm text-muted">Tick at least two coefficients.</p>
          ) : (
            <PairPlots methods={pairs} coordinates={coords} labels={labels.map((l, j) => `w${j} ${l}`)} />
          )}
        </div>
        <div className={SECTION}>
          <div className={HEADING}>Decision boundary: the predictive probability, at the playhead</div>
          <div className="grid gap-2 sm:grid-cols-2">
            {overlaid.map((r) => (
              <Boundary key={r.id} problem={problem} grid={gridAt(r, t)} size={r.gridSize} title={r.label} />
            ))}
          </div>
          <p className="m-0 text-xs text-muted">
            Filled dots are the training rows, open circles the held-out ones; the line is where the predictive
            probability is one half.
          </p>
        </div>
        <div className={`${SECTION} lg:col-span-2`}>
          <div className={HEADING}>Marginals of the first four chosen coefficients</div>
          {comparison === undefined ? (
            <p className="text-sm text-muted">No draws yet at this point of the run.</p>
          ) : (
            <PosteriorComparisonView record={comparison} filename="probit-lab-marginals" />
          )}
          <CoefficientTable results={overlaid} coordinates={coords} labels={labels} t={t} />
        </div>
        <div className={SECTION}>
          <div className={HEADING}>Trace and diagnostics</div>
          <ControlPanel columns={2}>
            <Choice
              label="method"
              value={focusResult.id}
              options={results.map((r) => ({ value: r.id, label: r.label }))}
              onChange={setFocus}
            />
          </ControlPanel>
          {focusResult.organisation === 'chain' ? (
            diagnostics === undefined ? (
              <p className="text-sm text-muted">The chains have too few draws to diagnose at this point.</p>
            ) : (
              <ChainDiagnosticsView record={diagnostics} filename="probit-lab-chains" />
            )
          ) : curve === undefined ? (
            <p className="text-sm text-muted">
              {focusResult.label} records no curve over its steps; its frames are at {focusResult.frames.length} points
              of the run.
              {focusResult.message === undefined ? '' : ` ${focusResult.message}.`}
            </p>
          ) : (
            <>
              <CurveChart curves={curveDots} x={curveX} y={curveY} />
              {focusResult.message === undefined ? null : (
                <p className="m-0 text-xs text-muted">{focusResult.message}.</p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
