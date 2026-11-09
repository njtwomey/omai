import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { type FakeFrames, fakeFrames, mountHook, resetGlobals } from './_harness.tsx'
import { type JobOptions, type JobStage, useJob } from './useJob.ts'

let frames: FakeFrames
beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
  frames = fakeFrames()
})
afterEach(() => {
  frames.restore()
  vi.useRealTimers()
  resetGlobals()
})

// One paint: the animation frame, then the zero timer the hook waits for after it.
async function paint(n = 1): Promise<void> {
  for (let i = 0; i < n; i++) {
    await frames.frame()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
  }
}

interface Deferred<T> {
  readonly promise: Promise<T>
  resolve(value: T): void
}
function defer<T>(): Deferred<T> {
  let resolve: (value: T) => void = () => undefined
  const promise = new Promise<T>((r) => {
    resolve = r
  })
  return { promise, resolve }
}

const stage = (name: string, run: JobStage['run'], weight?: number): JobStage => ({
  name,
  run,
  ...(weight === undefined ? {} : { weight }),
})

describe('useJob stages and progress', () => {
  it('runs the stages in order, each on the previous result, with the stage readable while it waits to run', async () => {
    const seen: unknown[] = []
    const stages = [
      stage('sample', () => {
        seen.push('sample')
        return 1
      }),
      stage('double', (prev: number) => {
        seen.push(['double', prev])
        return prev * 2
      }),
      stage('label', async (prev: number) => {
        seen.push(['label', prev])
        return `n=${prev}`
      }),
    ]
    const m = await mountHook(() => useJob<string>({ stages }))
    // Before any work: running, the first stage named, nothing computed yet.
    expect([m.current.phase, m.current.stage, m.current.stageName, m.current.progress, m.current.count]).toEqual([
      'running',
      0,
      'sample',
      0,
      3,
    ])
    expect(seen).toEqual([])
    await paint()
    expect(seen).toEqual(['sample'])
    expect([m.current.stage, m.current.stageName, m.current.progress]).toEqual([1, 'double', 1 / 3])
    await paint()
    expect([m.current.stage, m.current.stageName, m.current.progress]).toEqual([2, 'label', 2 / 3])
    await paint()
    expect(seen).toEqual(['sample', ['double', 1], ['label', 2]])
    expect([m.current.phase, m.current.progress, m.current.result, m.current.stageName]).toEqual([
      'done',
      1,
      'n=2',
      'label',
    ])
    // Nothing is left to run.
    expect([frames.pending, vi.getTimerCount()]).toEqual([0, 0])
  })

  it('counts stages by their weights, and takes progress a stage reports itself', async () => {
    const reports: number[] = []
    const stages = [
      stage('light', () => 0, 1),
      stage(
        'heavy',
        async (_: number, ctx) => {
          ctx.progress(0.5)
          reports.push(ctx.stage)
          await Promise.resolve()
          return 'ok'
        },
        3,
      ),
    ]
    const m = await mountHook(() => useJob({ stages }))
    await paint()
    expect(m.current.progress).toBe(0.25)
    await paint()
    // Half of the heavy stage, which is three quarters of the bar, on top of the light one.
    expect(m.current.phase).toBe('done')
    expect(reports).toEqual([1])
    expect(m.current.result).toBe('ok')
  })

  it('finishes a job with no stages', async () => {
    const m = await mountHook(() => useJob({ stages: [] }))
    expect([m.current.phase, m.current.progress, m.current.result]).toEqual(['done', 1, undefined])
  })
})

describe('useJob slices of a generator stage', () => {
  it('runs a generator in short slices, one per frame, reporting what it yields', async () => {
    const yielded: number[] = []
    const stages = [
      stage('long', function* () {
        for (let i = 1; i <= 100; i++) {
          yielded.push(i / 100)
          yield i / 100
        }
        return 'finished'
      }),
    ]
    const m = await mountHook(() => useJob<string>({ stages, sliceMs: 8 }))
    await paint() // the first paint gives the first slice
    const first = yielded.length
    expect(first).toBeGreaterThan(1)
    expect(first).toBeLessThan(100)
    expect(m.current.progress).toBe(yielded[first - 1])
    expect(m.current.phase).toBe('running')
    await paint()
    expect(yielded.length).toBe(2 * first)
    expect(m.current.progress).toBeCloseTo(yielded[2 * first - 1] as number, 12)
    await paint(40)
    expect(m.current.phase).toBe('done')
    expect(m.current.result).toBe('finished')
    expect(yielded.length).toBe(100)
  })

  it('passes a generator result to the next stage, and stops a generator that is cancelled', async () => {
    let closed = false
    const gen = (): Generator<number, number> =>
      (function* () {
        try {
          for (let i = 0; i < 1000; i++) yield i / 1000
          return 7
        } finally {
          closed = true
        }
      })()
    const m = await mountHook(() =>
      useJob({
        stages: [stage('gen', () => gen()), stage('next', (n: number) => n + 1)],
      }),
    )
    await paint()
    expect(m.current.phase).toBe('running')
    expect(closed).toBe(false)
    await m.unmount()
    expect(closed).toBe(true)
    expect([frames.pending, vi.getTimerCount()]).toEqual([0, 0])
  })
})

describe('useJob active, inputs and unmounting', () => {
  it('does nothing while inactive: no work, no frame request, no timer; it starts when activated', async () => {
    const run = vi.fn(() => 1)
    const options: JobOptions = { stages: [stage('a', run)], active: false }
    const m = await mountHook(() => useJob(options))
    await paint(2)
    expect(run).not.toHaveBeenCalled()
    expect([m.current.phase, m.current.stage, m.current.stageName]).toEqual(['idle', -1, ''])
    expect([frames.pending, vi.getTimerCount()]).toEqual([0, 0])
    ;(options as { active: boolean }).active = true
    await m.rerender()
    await paint()
    expect(run).toHaveBeenCalledTimes(1)
    expect(m.current.phase).toBe('done')
  })

  it('keeps a finished result across deactivation, and cancels a running job that is deactivated', async () => {
    const run = vi.fn(() => 'r')
    const options = { stages: [stage('a', run)], active: true }
    const m = await mountHook(() => useJob<string>(options))
    await paint()
    expect(m.current.result).toBe('r')
    options.active = false
    await m.rerender()
    expect([m.current.phase, m.current.result]).toEqual(['done', 'r'])
    options.active = true
    await m.rerender()
    await paint()
    expect(run).toHaveBeenCalledTimes(1)
    // A running job: deactivating stops it and leaves nothing scheduled.
    const slow = { stages: [stage('s', () => defer().promise)], active: true }
    const n = await mountHook(() => useJob(slow))
    await paint()
    slow.active = false
    await n.rerender()
    expect([n.current.phase, frames.pending, vi.getTimerCount()]).toEqual(['idle', 0, 0])
  })

  it("starts a new job when the inputs change, and drops the old job's late result", async () => {
    const pending: Deferred<string>[] = []
    const after = vi.fn((prev: string) => `after ${prev}`)
    let input = 1
    const stages = [
      stage('slow', () => {
        const d = defer<string>()
        pending.push(d)
        return d.promise
      }),
      stage('after', after),
    ]
    const m = await mountHook(() => useJob<string>({ stages, inputs: [input] }))
    await paint()
    expect(pending.length).toBe(1)
    input = 2
    await m.rerender()
    await paint()
    expect(pending.length).toBe(2)
    // The first job's promise settles late: nothing of it reaches the page or the next stage.
    await act(async () => {
      pending[0]?.resolve('old')
      await vi.advanceTimersByTimeAsync(0)
    })
    await paint()
    expect(after).not.toHaveBeenCalled()
    expect([m.current.phase, m.current.result]).toEqual(['running', undefined])
    await act(async () => {
      pending[1]?.resolve('new')
      await vi.advanceTimersByTimeAsync(0)
    })
    await paint()
    expect([m.current.phase, m.current.result]).toEqual(['done', 'after new'])
    expect(after).toHaveBeenCalledTimes(1)
    // The same inputs on a later render do not start it again.
    await m.rerender()
    await paint()
    expect(pending.length).toBe(2)
  })

  it('cancels on unmount: late results are dropped and nothing stays scheduled', async () => {
    const d = defer<number>()
    const after = vi.fn(() => 1)
    const m = await mountHook(() => useJob({ stages: [stage('a', () => d.promise), stage('b', after)] }))
    await paint()
    await m.unmount()
    await act(async () => {
      d.resolve(1)
      await vi.advanceTimersByTimeAsync(200)
    })
    expect(after).not.toHaveBeenCalled()
    expect([frames.pending, vi.getTimerCount()]).toEqual([0, 0])
  })

  it('runs the first stage once in StrictMode, though its effect runs twice', async () => {
    const run = vi.fn(() => 1)
    const m = await mountHook(() => useJob({ stages: [stage('a', run)] }), { strict: true })
    await paint(2)
    expect(run).toHaveBeenCalledTimes(1)
    expect(m.current.phase).toBe('done')
  })
})

describe('useJob errors and restart', () => {
  it('reports the failed stage and its error, skips the rest, and restarts from the first stage', async () => {
    let fail = true
    const first = vi.fn(() => 1)
    const last = vi.fn(() => 'end')
    const stages = [
      stage('a', first),
      stage('b', () => {
        if (fail) throw new Error('boom')
        return 2
      }),
      stage('c', last),
    ]
    const m = await mountHook(() => useJob<string>({ stages }))
    await paint(2)
    expect(m.current.phase).toBe('error')
    expect([m.current.stage, m.current.stageName]).toEqual([1, 'b'])
    expect((m.current.error as Error).message).toBe('boom')
    expect(m.current.result).toBeUndefined()
    expect(last).not.toHaveBeenCalled()
    fail = false
    await act(async () => m.current.restart())
    await paint(3)
    expect(first).toHaveBeenCalledTimes(2)
    expect([m.current.phase, m.current.result, m.current.error]).toEqual(['done', 'end', undefined])
  })

  it('reports a rejected promise and a throwing generator', async () => {
    const m = await mountHook(() => useJob({ stages: [stage('a', () => Promise.reject(new Error('no')))] }))
    await paint()
    expect([m.current.phase, (m.current.error as Error).message]).toEqual(['error', 'no'])
    const g = await mountHook(() =>
      useJob({
        stages: [
          stage('g', function* () {
            yield 0.1
            throw new Error('mid')
          }),
        ],
      }),
    )
    await paint()
    expect([g.current.phase, (g.current.error as Error).message]).toEqual(['error', 'mid'])
  })
})
