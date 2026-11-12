import type { StageWorker } from 'omai-render/producer'
import { Choice, ControlPanel, NumberField, Player, SeedInput, Select, usePlayback } from 'omai-render/react'
import { type ReactNode, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'
import { type PoolHandle, type PoolState, runPool, type StatusLine } from '../../../shared/workerPool.ts'
import {
  countsAt,
  type Dataset,
  FORMS,
  type Form,
  frameAt,
  makeData,
  type PoolingRequest,
  type PoolingResult,
  type Settings,
} from './compute.ts'
import { effectName, PairPlot } from './PairPlot.tsx'
import { Shrinkage } from './Shrinkage.tsx'

const CHAINS = 3
const WARMUP = 400
const DRAWS = 400
const TICKS = 101

const button =
  'cursor-pointer rounded-md border border-line bg-bg px-3 py-1.5 text-xs text-fg hover:bg-surface disabled:cursor-default disabled:opacity-50'
const HEADING = 'text-xs font-medium uppercase tracking-wide text-muted'
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

export default function PartialPooling(): ReactNode {
  const [settings, setSettings] = useState<Settings>({ groups: 20, smallest: 3, tau: 0.1, seed: 1 })
  const [pool, setPool] = useState<PoolState<PoolingResult> | undefined>(undefined)
  const [shown, setShown] = useState<Dataset | undefined>(undefined)
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
  const start = (): void => {
    handle.current?.cancel()
    const data = makeData(settings)
    setShown(data)
    setPool(undefined)
    handle.current = runPool<PoolingRequest, PoolingResult>(
      FORMS.map((method) => ({ method, data, seed: settings.seed, chains: CHAINS, warmup: WARMUP, draws: DRAWS })),
      Object.fromEntries(FORMS.map((f) => [f, `${f} NUTS`])),
      {
        ...(typeof Worker === 'undefined' ? {} : { createWorker: makeWorker }),
        run: async (request, host) => (await import('./compute.ts')).runPooling(request, host),
        onChange: setPool,
      },
    )
  }

  const elapsed = pool === undefined ? 0 : ((pool.endedAt ?? now) - pool.startedAt) / 1000
  const share =
    pool === undefined
      ? 0
      : pool.lines.reduce((a, l) => a + Math.min(1, l.step / (WARMUP + DRAWS + 1)), 0) / pool.lines.length
  const finished = pool !== undefined && !pool.running && shown !== undefined && pool.results[FORMS[0]] !== undefined

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 rounded-lg border border-line bg-surface p-3">
        <ControlPanel columns={4} title="Groups">
          <NumberField
            label="groups"
            value={settings.groups}
            min={4}
            max={40}
            integer
            onChange={(v) => set('groups', v)}
          />
          <NumberField
            label="smallest group size"
            value={settings.smallest}
            min={1}
            max={30}
            integer
            onChange={(v) => set('smallest', v)}
          />
          <NumberField
            label="true spread between groups (tau)"
            value={settings.tau}
            min={0.02}
            max={1}
            log
            onChange={(v) => set('tau', v)}
          />
          <SeedInput value={settings.seed} onChange={(v) => set('seed', v)} />
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
                {elapsed.toFixed(1)} s · {CHAINS} chains of {WARMUP} warm-up and {DRAWS} draws
                {pool === undefined || pool.workers === 0 ? '' : ` · ${pool.workers} workers`}
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
                  {l.step > 0 && l.phase !== 'queued' ? ` · ${l.step} iterations` : ''}
                  {l.phase === 'running' ? ` · ${(l.elapsedMs / 1000).toFixed(1)} s` : ''}
                  {l.error === undefined ? '' : `: ${l.error.slice(0, 60)}`}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
      {finished ? <Outputs key={pool.startedAt} results={pool.results} data={shown} /> : null}
      {pool === undefined ? (
        <p className="text-sm text-muted">
          Groups of very different sizes each estimate an effect from their own mean. Press Run: both forms of the
          hierarchical model sample in Web Workers, then the shrinkage plot and the pair plot appear on one playhead.
        </p>
      ) : null}
    </div>
  )
}

/** What the runs recorded, drawn at the playhead of one player. */
function Outputs(props: {
  readonly results: Readonly<Record<string, PoolingResult>>
  readonly data: Dataset
}): ReactNode {
  const { results, data } = props
  const [form, setForm] = useState<Form>('centred')
  const result = results[form] ?? (Object.values(results)[0] as PoolingResult)
  const total = Math.max(...Object.values(results).map((r) => r.steps))
  const playback = usePlayback(TICKS, { fps: 12 })
  const seek = useRef(playback.seek)
  seek.current = playback.seek
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new run opens on its last frame
  useEffect(() => seek.current(TICKS - 1), [results])
  const t = useDeferredValue(Math.round((playback.frame / (TICKS - 1)) * total))

  const last = data.size.length - 1
  const [pair, setPair] = useState<readonly [number, number]>([0, last])
  const choices = useMemo(() => data.size.map((_, j) => ({ value: j, label: effectName(data, j) })), [data])
  const frame = useMemo(() => frameAt(result, t), [result, t])
  const counts = useMemo(() => countsAt(result, t), [result, t])

  return (
    <div className="flex flex-col gap-4">
      <div className="sticky top-11 z-10 flex flex-col gap-1 rounded-lg border border-line bg-surface p-2">
        <Player
          controller={playback}
          stamp={(f) => {
            const step = Math.round((f / (TICKS - 1)) * total)
            return step <= result.warmup
              ? `warm-up, iteration ${step}`
              : `draw ${step - result.warmup} of ${DRAWS} in each chain`
          }}
          fpsChoices={[2, 5, 12, 24, 60]}
          size="sm"
        />
      </div>
      <div className="flex flex-wrap items-end gap-x-6 gap-y-2">
        <ControlPanel columns={1}>
          <Choice label="parametrisation" value={form} options={FORMS} onChange={setForm} />
        </ControlPanel>
        <p className="m-0 pb-1 text-sm tabular-nums text-muted">
          {counts.draws} draws so far,{' '}
          <span className={counts.divergent > 0 ? 'text-(--wb-danger,#cf222e)' : ''}>{counts.divergent} divergent</span>
        </p>
      </div>
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-2">
          <div className={HEADING}>Shrinkage: each group alone, pooled, and partially pooled</div>
          <Shrinkage data={data} frame={frame} pooledLabel={`partial pooling (${form} NUTS), 90%`} />
          <p className="m-0 text-xs text-muted">
            Small groups move far towards the common mean and their intervals narrow; large groups stay where their own
            mean puts them.
          </p>
        </div>
        <div className="flex min-w-0 flex-col gap-2">
          <div className={HEADING}>The draws: group mean, spread and two group effects</div>
          <ControlPanel columns={2}>
            <Select
              label="first group"
              value={pair[0]}
              options={choices}
              onChange={(v: number) => setPair([v, pair[1]])}
            />
            <Select
              label="second group"
              value={pair[1]}
              options={choices}
              onChange={(v: number) => setPair([pair[0], v])}
            />
          </ControlPanel>
          <PairPlot data={data} result={result} groups={pair} cursor={t} />
          <p className="m-0 text-xs text-muted">
            Red points are divergent transitions. In the centred form they collect where the spread is small and the
            group effects are tied to the mean; the non-centred form has none.
          </p>
        </div>
      </div>
    </div>
  )
}
