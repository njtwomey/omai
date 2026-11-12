import { describe, expect, it } from 'vitest'
import { serveStory, type WorkerScope } from './protocol.ts'
import { initialRun, nextSeed, stepRun } from './runState.ts'

function scope(): { readonly s: WorkerScope; readonly sent: unknown[] } {
  const sent: unknown[] = []
  const s: WorkerScope = { postMessage: (m) => sent.push(m), onmessage: null }
  return { s, sent }
}

const settle = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0))

describe('the run state', () => {
  it('is idle until started, then runs, then holds its result', () => {
    let s = initialRun<number>(3)
    expect(s.phase).toBe('idle')
    s = stepRun(s, { type: 'status', sentence: 'too early' })
    expect(s.status).toBe('')
    s = stepRun(s, { type: 'start', seed: 3 })
    expect(s).toMatchObject({ phase: 'running', runs: 1, seed: 3 })
    s = stepRun(s, { type: 'status', sentence: 'sampling' })
    expect(s.status).toBe('sampling')
    s = stepRun(s, { type: 'result', value: 42, elapsedMs: 1500 })
    expect(s).toMatchObject({ phase: 'done', result: 42, elapsedMs: 1500, status: '' })
  })

  it('keeps the old result while a new run goes, and ignores late events', () => {
    let s = stepRun(stepRun(initialRun<number>(1), { type: 'start', seed: 1 }), {
      type: 'result',
      value: 1,
      elapsedMs: 1,
    })
    s = stepRun(s, { type: 'start', seed: nextSeed(1) })
    expect(s).toMatchObject({ phase: 'running', result: 1, runs: 2 })
    expect(s.seed).not.toBe(1)
    const done = stepRun(s, { type: 'result', value: 2, elapsedMs: 5 })
    expect(stepRun(done, { type: 'result', value: 3, elapsedMs: 9 }).result).toBe(2)
  })

  it('reports an error and can be started again', () => {
    let s = stepRun(initialRun<number>(1), { type: 'start', seed: 1 })
    s = stepRun(s, { type: 'error', message: 'boom' })
    expect(s).toMatchObject({ phase: 'error', error: 'boom' })
    s = stepRun(s, { type: 'start', seed: 1 })
    expect(s).toMatchObject({ phase: 'running', error: undefined })
  })

  it('re-rolls the seed to a different one each time', () => {
    const seeds = new Set<number>()
    let seed = 1
    for (let i = 0; i < 50; i++) {
      seeds.add(seed)
      seed = nextSeed(seed)
    }
    expect(seeds.size).toBe(50)
  })
})

describe('the worker wire', () => {
  it('answers a run with its statuses and then its result', async () => {
    const { s, sent } = scope()
    serveStory(s, (seed, say) => {
      say('first')
      say('second')
      return { seed }
    })
    s.onmessage?.({ data: { type: 'run', seed: 7 } })
    await settle()
    expect(sent).toEqual([
      { type: 'status', sentence: 'first' },
      { type: 'status', sentence: 'second' },
      { type: 'result', value: { seed: 7 } },
    ])
  })

  it('answers a failure with its message, and ignores what is not a run', async () => {
    const { s, sent } = scope()
    serveStory(s, () => {
      throw new Error('no draws')
    })
    s.onmessage?.({ data: 'hello' })
    s.onmessage?.({ data: { type: 'run', seed: 1 } })
    await settle()
    expect(sent).toEqual([{ type: 'error', message: 'no draws' }])
  })

  it('computes nothing until it is asked', async () => {
    let calls = 0
    const { s, sent } = scope()
    serveStory(s, () => ++calls)
    await settle()
    expect(calls).toBe(0)
    expect(sent).toEqual([])
  })
})
