// The run contract: a task reports, the state never goes backwards, a stop ends the run with what the task kept, a failure
// is reported, and the pieces that build on it (the EP hooks, the pacer) report and stop as the pages need.
import { describe, expect, it } from 'vitest'
import { epHooks, pacer } from './ep/sweeps.ts'
import { elapsedOf, going, type RunProgress, type RunState, type RunTask, startRun } from './run.ts'

const now = (): number => 0
const tick = { yieldToPage: () => Promise.resolve(), now }

/** Starts a task and collects every state it reports, with a promise of the last. */
function collect<R>(task: RunTask<R>): { states: RunState<R>[]; ended: Promise<RunState<R>>; stop: () => void } {
  const states: RunState<R>[] = []
  let resolve: (s: RunState<R>) => void = () => undefined
  const ended = new Promise<RunState<R>>((r) => {
    resolve = r
  })
  const handle = startRun(
    task,
    (s) => {
      states.push(s)
      if (!going(s.phase)) resolve(s)
    },
    tick,
  )
  return { states, ended, stop: handle.stop }
}

describe('startRun', () => {
  it('is queued at once, then running, then done with the bar full', async () => {
    const { states, ended } = collect<number>(async (host) => {
      for (let i = 1; i <= 4; i++) {
        host.report({ stage: 'work', done: i, total: 10 })
        await host.yieldToPage()
      }
      return 42
    })
    const end = await ended
    expect(states[0]?.phase).toBe('queued')
    expect(states.map((s) => s.phase)).toContain('running')
    expect(end.phase).toBe('done')
    expect(end.value).toBe(42)
    expect(end.progress?.done).toBe(end.progress?.total)
    expect(end.endedAt).toBeDefined()
  })

  it('never lets the progress of a stage go backwards or past its total, and a new stage starts again', async () => {
    const { states, ended } = collect<null>(async (host) => {
      for (const done of [1, 5, 3, 7, 99, 2]) host.report({ stage: 'a', done, total: 10 })
      host.report({ stage: 'b', done: 1, total: 4 })
      return null
    })
    await ended
    const a = states.flatMap((s) => (s.progress?.stage === 'a' ? [s.progress.done] : []))
    expect(a).toEqual([1, 5, 5, 7, 10, 10])
    const b = states.filter((s) => s.progress?.stage === 'b').map((s) => s.progress?.done)
    expect(b).toEqual([1, 4])
  })

  it('goes to converged when the last report says so, and the bar is full', async () => {
    const { ended } = collect<number>(async (host) => {
      host.report({ done: 12, total: 300, converged: true })
      return 1
    })
    const end = await ended
    expect(end.phase).toBe('converged')
    expect(end.progress?.done).toBe(300)
  })

  it('stops where it stands and keeps what the task returned', async () => {
    let seen = 0
    const run = collect<number>(async (host) => {
      let n = 0
      while (!host.stopped() && n < 1000) {
        n += 1
        seen = n
        host.report({ done: n, total: 1000 })
        await host.yieldToPage()
        if (n === 5) run.stop()
      }
      return n
    })
    const end = await run.ended
    expect(end.phase).toBe('stopped')
    expect(end.value).toBe(seen)
    expect(seen).toBeLessThan(1000)
    expect(end.progress?.done).toBeLessThan(1000)
  })

  it('a stop before the task starts ends the run without calling it', async () => {
    let called = false
    const run = collect<number>(async () => {
      called = true
      return 1
    })
    run.stop()
    const end = await run.ended
    expect(end.phase).toBe('stopped')
    expect(called).toBe(false)
  })

  it('tells the task to stop through onStop (a worker is terminated there)', async () => {
    let terminated = false
    const run = collect<string>((host) => {
      host.onStop(() => {
        terminated = true
      })
      return new Promise((resolve) => {
        host.onStop(() => resolve('partial'))
        setTimeout(() => run.stop(), 0)
      })
    })
    const end = await run.ended
    expect(terminated).toBe(true)
    expect(end.phase).toBe('stopped')
    expect(end.value).toBe('partial')
  })

  it('reports a failure with its message and no value', async () => {
    const { ended } = collect<number>(async (host) => {
      host.report({ done: 1, total: 3 })
      throw new Error('no posterior')
    })
    const end = await ended
    expect(end.phase).toBe('failed')
    expect(end.error).toBe('no posterior')
    expect(end.value).toBeUndefined()
  })

  it('cancel silences the run: nothing more is reported and the task is told to stop', async () => {
    let sawStop = false
    const states: RunState<number>[] = []
    const handle = startRun<number>(
      async (host) => {
        while (!host.stopped()) await host.yieldToPage()
        sawStop = true
        return 1
      },
      (s) => states.push(s),
      tick,
    )
    await tick.yieldToPage()
    await tick.yieldToPage()
    handle.cancel()
    const count = states.length
    await new Promise((r) => setTimeout(r, 10))
    expect(sawStop).toBe(true)
    expect(states.length).toBe(count)
  })

  it('measures the elapsed time against the clock while it goes and until it ended', () => {
    const base: RunState<number> = {
      phase: 'running',
      progress: undefined,
      startedAt: 100,
      endedAt: undefined,
      value: undefined,
      error: undefined,
    }
    expect(elapsedOf(base, 350)).toBe(250)
    expect(elapsedOf({ ...base, phase: 'done', endedAt: 180 }, 999)).toBe(80)
  })
})

describe('the EP hooks', () => {
  const host = (reports: RunProgress[], stopAfter = Number.POSITIVE_INFINITY) => ({
    report: (p: RunProgress) => reports.push(p),
    stopped: () => reports.length >= stopAfter,
    yieldToPage: () => Promise.resolve(),
  })

  it('report the sweep over the cap and the change against the tolerance', () => {
    const reports: RunProgress[] = []
    const hooks = epHooks(host(reports), 'EP')
    hooks.onSweep?.({ sweep: 41, cap: 300, change: 3e-3, tol: 1e-4 })
    expect(reports[0]).toMatchObject({ stage: 'EP', done: 41, total: 300, converged: false })
    expect(reports[0]?.detail).toContain('sweep 41 of 300')
    expect(reports[0]?.detail).toContain('3.0e-3')
    expect(reports[0]?.detail).toContain('1.0e-4')
  })

  it('jump to a full bar when the change is under the tolerance', () => {
    const reports: RunProgress[] = []
    epHooks(host(reports), 'EP').onSweep?.({ sweep: 41, cap: 300, change: 1e-6, tol: 1e-4 })
    expect(reports[0]).toMatchObject({ done: 300, total: 300, converged: true })
  })

  it('the pacer reports every round, hands the thread back only now and then, and says when to stop', async () => {
    const reports: RunProgress[] = []
    let yields = 0
    const h = host(reports, 4)
    const pace = pacer(
      {
        ...epHooks(h, 'Gibbs', 'sweep'),
        yieldToPage: async () => {
          yields += 1
        },
      },
      1e9,
    )
    const stopped: boolean[] = []
    for (let i = 1; i <= 6; i++) stopped.push(await pace(i, 6))
    expect(reports.length).toBe(6)
    expect(yields).toBe(0)
    expect(stopped.slice(0, 3)).toEqual([false, false, false])
    expect(stopped[3]).toBe(true)
  })
})
