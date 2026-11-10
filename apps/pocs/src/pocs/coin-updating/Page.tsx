import type { Curve } from 'omai-render/charts'
import type { StageWorker } from 'omai-render/producer'
import { Choice, ControlPanel, NumberField, Player, SeedInput, Toggle, usePlayback } from 'omai-render/react'
import { type ReactNode, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'
import { CurveChart } from '../../shared/CurveChart.tsx'
import { type PoolHandle, type PoolState, runPool, type StatusLine } from '../../shared/workerPool.ts'
import {
  type CoinRequest,
  type CoinResult,
  DEFAULT_METHODS,
  type Family,
  type Frame,
  METHODS,
  makeProblem,
  PRESETS,
  type PresetId,
  type Problem,
  runCoin,
  type Settings,
} from './compute.ts'

const phaseText: Record<StatusLine['phase'], string> = {
  queued: 'queued',
  running: 'fitting',
  finishing: 'finishing',
  done: 'done',
  stopped: 'not started',
  failed: 'failed',
}
const button =
  'cursor-pointer rounded-md border border-line bg-bg px-3 py-1.5 text-xs text-fg hover:bg-surface disabled:cursor-default disabled:opacity-50'
const HEADING = 'text-xs font-medium uppercase tracking-wide text-muted'
const cell = 'px-2 py-1 text-right tabular-nums whitespace-nowrap'

const makeWorker = (): StageWorker =>
  new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as StageWorker

const WORDS: Record<Family, { parameter: string; unit: string; truth: [number, number, number]; label: string }> = {
  beta: { parameter: 'bias of the coin', unit: 'p', truth: [0.02, 0.98, 0.3], label: 'Coin: beta-Bernoulli' },
  gamma: { parameter: 'rate of the counter', unit: 'rate', truth: [0.5, 12, 3.2], label: 'Counter: gamma-Poisson' },
}
const colourOf = (id: string): string =>
  `tab${
    Math.max(
      0,
      METHODS.findIndex((m) => m.id === id),
    ) % 10
  }`
const fmt = (v: number, digits = 3): string => (Number.isFinite(v) ? v.toPrecision(digits) : '-')

export default function CoinUpdating(): ReactNode {
  const [settings, setSettings] = useState<Settings>({ family: 'beta', truth: 0.3, n: 60, preset: 'flat', seed: 1 })
  const [chosen, setChosen] = useState<ReadonlySet<string>>(new Set(DEFAULT_METHODS))
  const [pool, setPool] = useState<PoolState<CoinResult> | undefined>(undefined)
  const [shown, setShown] = useState<Problem | undefined>(undefined)
  const handle = useRef<PoolHandle | undefined>(undefined)
  const [now, setNow] = useState(() => performance.now())
  const running = pool?.running === true
  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => setNow(performance.now()), 100)
    return () => clearInterval(timer)
  }, [running])
  useEffect(() => () => handle.current?.cancel(), [])

  const set = <K extends keyof Settings>(key: K, value: Settings[K]): void =>
    setSettings((s) => ({ ...s, [key]: value }))
  const words = WORDS[settings.family]
  const start = (): void => {
    handle.current?.cancel()
    const problem = makeProblem(settings)
    const picked = METHODS.filter((m) => m.id === 'exact' || chosen.has(m.id))
    setShown(problem)
    setPool(undefined)
    handle.current = runPool<CoinRequest, CoinResult>(
      picked.map((m) => ({ method: m.id, problem })),
      Object.fromEntries(picked.map((m) => [m.id, m.label])),
      {
        ...(typeof Worker === 'undefined' ? {} : { createWorker: makeWorker }),
        run: runCoin,
        onChange: setPool,
      },
    )
  }
  const results = useMemo(
    () => METHODS.flatMap((m) => (pool?.results[m.id] === undefined ? [] : [pool.results[m.id] as CoinResult])),
    [pool],
  )
  const finished = pool !== undefined && !pool.running && shown !== undefined && pool.results['exact'] !== undefined
  // The bar counts the fits done of all the fits asked for.
  const fits = pool === undefined || shown === undefined ? 0 : pool.lines.length * shown.ladder.length
  const done = pool?.lines.reduce((a, l) => a + (l.phase === 'done' ? (shown?.ladder.length ?? 0) : l.step), 0) ?? 0
  const elapsed = pool === undefined ? 0 : ((pool.endedAt ?? now) - pool.startedAt) / 1000

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 rounded-lg border border-line bg-surface p-3">
        <ControlPanel columns={4} title="Problem">
          <Choice
            label="model"
            value={settings.family}
            options={(['beta', 'gamma'] as const).map((f) => ({ value: f, label: WORDS[f].label }))}
            onChange={(f: Family) => setSettings((s) => ({ ...s, family: f, truth: WORDS[f].truth[2] }))}
          />
          <NumberField
            label={`true ${words.parameter}`}
            value={settings.truth}
            min={words.truth[0]}
            max={words.truth[1]}
            onChange={(v) => set('truth', v)}
          />
          <NumberField
            label="observations"
            value={settings.n}
            min={5}
            max={200}
            integer
            onChange={(v) => set('n', v)}
          />
          <SeedInput value={settings.seed} onChange={(v) => set('seed', v)} />
          <Choice
            label="prior"
            value={settings.preset}
            options={(Object.keys(PRESETS[settings.family]) as PresetId[]).map((id) => ({
              value: id,
              label: id,
            }))}
            onChange={(v: PresetId) => set('preset', v)}
          />
        </ControlPanel>
        <p className="m-0 text-xs text-muted">Prior: {PRESETS[settings.family][settings.preset].note}.</p>
        <div className="flex flex-col gap-1">
          <div className={HEADING}>Approximations (the exact posterior is always fitted)</div>
          <div className="grid gap-x-4 sm:grid-cols-2 lg:grid-cols-3">
            {METHODS.filter((m) => m.id !== 'exact').map((m) => (
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
            <button type="button" className={button} disabled={running} onClick={start}>
              {pool === undefined ? 'Run' : 'Run again'}
            </button>
            <button type="button" className={button} disabled={!running} onClick={() => handle.current?.stop()}>
              Stop
            </button>
          </div>
          <div className="flex min-w-40 flex-1 flex-col gap-1">
            <div className="flex justify-between text-xs tabular-nums text-muted">
              <span>{pool === undefined ? 'ready' : running ? 'fitting' : 'finished'}</span>
              <span>
                {elapsed.toFixed(1)} s · {done} of {fits} fits
                {pool === undefined || pool.workers === 0 ? '' : ` · ${pool.workers} workers`}
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-bg">
              <div
                className="h-full rounded-full bg-(--wb-accent)"
                style={{ width: `${100 * (pool === undefined ? 0 : running ? done / Math.max(1, fits) : 1)}%` }}
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
                  {l.phase === 'running' ? ` · fit ${l.step + 1} of ${shown?.ladder.length ?? 0}` : ''}
                  {l.error === undefined ? '' : `: ${l.error.slice(0, 60)}`}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
      {finished ? <Outputs key={pool.startedAt} results={results} problem={shown} /> : null}
      {pool === undefined ? (
        <p className="text-sm text-muted">
          Press Run: every method is fitted in a Web Worker to the first n observations at a ladder of n, with nothing
          drawn while it works. Then the posterior at each n appears on one playhead: scrub or play the number of
          observations seen.
        </p>
      ) : null}
    </div>
  )
}

/** The observations as a strip: each one a cell, the ones seen so far bright and a marker where the playhead is. */
function Strip(props: { readonly problem: Problem; readonly n: number }): ReactNode {
  const { problem, n } = props
  const y = problem.observations
  const top = Math.max(1, ...y)
  return (
    <svg
      viewBox={`0 0 ${y.length} 3`}
      preserveAspectRatio="none"
      className="h-8 w-full"
      role="img"
      aria-label={`the ${y.length} observations, ${n} seen`}
    >
      {y
        .map((v, id) => ({ v, id }))
        .map(({ v, id: i }) => {
          const h = problem.family === 'beta' ? v * 1.8 + 0.5 : 0.2 + (2 * v) / top
          return (
            <rect
              key={i}
              x={i + 0.1}
              y={3 - h}
              width={0.8}
              height={h}
              className="fill-(--wb-accent)"
              opacity={i < n ? 1 : 0.2}
            />
          )
        })}
      <line x1={n} x2={n} y1={0} y2={3} className="stroke-fg" strokeWidth={0.12} vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

/** What the run produced, at the playhead of one player: the ladder position is the number of observations seen. */
function Outputs(props: { readonly results: readonly CoinResult[]; readonly problem: Problem }): ReactNode {
  const { results, problem } = props
  const { ladder, grid, domain, family, truth, observations } = problem
  const playback = usePlayback(ladder.length, { fps: 3 })
  // A new run opens on its end (all the observations); Play runs it from the start.
  const seek = useRef(playback.seek)
  seek.current = playback.seek
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new set of results opens on its last frame
  useEffect(() => seek.current(ladder.length - 1), [results])
  const at = useDeferredValue(playback.frame)
  const n = ladder[at] ?? 0
  const words = WORDS[family]
  const seen = observations.slice(0, n)
  const total = seen.reduce((a, b) => a + b, 0)
  const frameOf = (r: CoinResult): Frame | undefined => r.frames[at]
  const exact = results.find((r) => r.id === 'exact') as CoinResult
  const prior = exact.frames[0]

  const { curves, limits } = useMemo(() => {
    const frameOf = (r: CoinResult): Frame | undefined => r.frames[at]
    const shownNow = results.flatMap((r) => (frameOf(r) === undefined ? [] : [r]))
    const peak = (f: Frame | undefined): number => (f === undefined ? 0 : Math.max(...f.density))
    const top = 1.15 * (peak(frameOf(exact)) || 1)
    const floor = -0.07 * top * (shownNow.length + 1)
    const out: Curve[] = []
    if (prior !== undefined && n > 0)
      out.push({ x: grid, y: Array.from(prior.density), name: 'prior', colour: 'muted', dash: 'dashed', width: 1.5 })
    out.push({
      x: [truth, truth],
      y: [floor, top],
      name: `true ${words.unit}`,
      colour: 'fg',
      dash: 'dotted',
      width: 1.5,
    })
    for (const [i, r] of shownNow.entries()) {
      const f = frameOf(r) as Frame
      const colour = r.id === 'exact' ? 'fg' : colourOf(r.id)
      out.push({
        x: grid,
        y: Array.from(f.density),
        name: r.label,
        colour,
        width: r.id === 'exact' ? 2.4 : 1.6,
        ...(r.id === 'exact' ? { area: 0.12 } : {}),
      })
      // The 95% interval, a bar under the axis.
      const y = -0.07 * top * (i + 1)
      out.push({ x: [f.lower, f.upper], y: [y, y], name: `${r.label} 95%`, colour, width: 4, opacity: 0.85 })
    }
    return { curves: out, limits: [floor, top] as [number, number] }
  }, [results, at, grid, truth, words.unit, prior, n, exact])
  const xAxis = useMemo(
    () => ({ label: words.unit, limits: [domain[0], domain[1]] as [number, number] }),
    [domain, words.unit],
  )
  const yAxis = useMemo(
    () => ({ label: 'posterior density', limits, format: (v: number) => (v < 0 ? '' : String(v)) }),
    [limits],
  )

  // The distance of each approximation to the exact posterior along the ladder, with the playhead on it.
  const distance = useMemo(() => {
    const out: Curve[] = results
      .filter((r) => r.id !== 'exact')
      .map((r) => ({
        x: r.frames.filter((f) => f.n > 0).map((f) => f.n),
        y: r.frames.filter((f) => f.n > 0).map((f) => f.w1),
        name: r.label,
        colour: colourOf(r.id),
        width: 1.6,
      }))
    const top = Math.max(0.05, ...out.flatMap((c) => c.y as number[]).filter(Number.isFinite))
    if (n > 0) out.push({ x: [n, n], y: [0, top], name: 'now', colour: 'fg', dash: 'dotted', width: 1 })
    return out
  }, [results, n])
  const distanceX = useMemo(() => ({ label: 'observations seen', scale: 'log' as const }), [])
  const distanceY = useMemo(() => ({ label: 'Wasserstein distance to exact (exact sds)' }), [])

  return (
    <div className="flex flex-col gap-4">
      <div className="sticky top-11 z-10 flex flex-col gap-1 rounded-lg border border-line bg-surface p-2">
        <Player
          controller={playback}
          stamp={() => `n = ${n} of ${observations.length}`}
          fpsChoices={[1, 2, 3, 6, 12]}
          size="sm"
        />
        <Strip problem={problem} n={n} />
        <div className="text-xs tabular-nums text-muted">
          {family === 'beta'
            ? `${total} heads in the first ${n} flips`
            : `${total} events in the first ${n} counts, ${n === 0 ? 'none yet' : `mean ${(total / n).toFixed(2)}`}`}
        </div>
      </div>
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-2">
          <div className={HEADING}>The posterior of the {words.parameter}, with its 95% interval</div>
          <CurveChart curves={curves} x={xAxis} y={yAxis} />
        </div>
        <div className="flex min-w-0 flex-col gap-2">
          <div className={HEADING}>How far each approximation is from the exact one</div>
          <CurveChart curves={distance} x={distanceX} y={distanceY} />
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr className="border-b border-line text-muted">
              <th className={`${cell} text-left`}>at n = {n}</th>
              <th className={cell}>mean</th>
              <th className={cell}>sd</th>
              <th className={cell}>95% interval</th>
              <th className={cell}>KL to exact</th>
              <th className={cell}>W1 to exact</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => {
              const f = frameOf(r)
              return (
                <tr key={r.id} className="border-b border-line/50" title={r.message}>
                  <td className={`${cell} text-left`}>
                    <span style={{ color: r.id === 'exact' ? 'var(--wb-fg)' : `var(--wb-${colourOf(r.id)})` }}>●</span>{' '}
                    {r.label}
                    {f === undefined ? ' (stopped before here)' : ''}
                  </td>
                  <td className={cell}>{f === undefined ? '-' : fmt(f.mean)}</td>
                  <td className={cell}>{f === undefined ? '-' : fmt(f.sd)}</td>
                  <td className={cell}>{f === undefined ? '-' : `${fmt(f.lower)} to ${fmt(f.upper)}`}</td>
                  <td className={cell}>{f === undefined || r.id === 'exact' ? '-' : fmt(f.kl, 2)}</td>
                  <td className={cell}>{f === undefined || r.id === 'exact' ? '-' : fmt(f.w1, 2)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="m-0 text-xs text-muted">
        The dashed curve is the prior, the dotted line the true {words.unit}; the bars under the axis are the 95%
        intervals. KL is of the exact posterior from the method on the exact posterior's equal-mass bins, W1 the
        Wasserstein distance in exact posterior sds.
      </p>
    </div>
  )
}
