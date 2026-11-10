// Simulation-based calibration as a method: a step is a replication, the ranks are the state, a run of it ranks the
// truth as `sbc` does, resumes from a serialised state, and reads as the tests of the ranks so far.
import { describe, expect, it } from 'vitest'
import { foldIn, key, normal } from '../../random/index.ts'
import { item } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { runAnytimeAsync } from '../inference/index.ts'
import { sbc } from './sbc.ts'
import { sbcMethod } from './sbc-method.ts'

// m ~ N(0, 1), y ~ N(m, 1): the posterior is N(y / 2, 1 / 2). `narrow` scales the posterior's width.
const problem = (narrow: number) => ({
  numRanks: 19,
  simulate: (k: ReturnType<typeof key>) => {
    const m = item(normal(foldIn(k, 0), []))
    return { data: m + item(normal(foldIn(k, 1), [])), truth: { m } }
  },
  fit: (y: number, k: ReturnType<typeof key>) => ({
    m: Array.from(normal(k, [19], { loc: y / 2, scale: narrow * Math.SQRT1_2 }).data),
  }),
})

describe('sbcMethod', () => {
  it('takes a replication a step and ranks the truth exactly as sbc does', () => {
    const p = problem(1)
    const method = sbcMethod({ ...p, key: key(3), replications: 30 })
    const s = session(method, { steps: 30, mode: 'eager' })
    const whole = sbc({ ...p, key: key(3), replications: 30 })
    expect(s.t).toBe(30)
    expect(s.status).toBe('completed')
    expect(s.state.ranks['m']).toEqual(whole.quantities[0]?.ranks)
    expect(method.result(s.state)?.quantities[0]?.histogram).toEqual(whole.quantities[0]?.histogram)
  })

  it('has no result before the tenth replication, and one from then on', () => {
    const method = sbcMethod({ ...problem(1), replications: 50 })
    const s = session(method, { steps: 50, mode: 'live' })
    s.advance({ steps: 9 })
    expect(method.result(s.state)).toBeUndefined()
    s.advance({ steps: 1 })
    expect(method.result(s.state)?.replications).toBe(10)
  })

  it('resumes from a serialised state to the same ranks', () => {
    const method = sbcMethod({ ...problem(1), replications: 24 })
    const whole = session(method, { steps: 24, mode: 'eager', checkpointEvery: 1 })
    const part = session(method, { steps: 24, mode: 'live', checkpointEvery: 1 })
    part.advance({ steps: 11 })
    const json = JSON.parse(JSON.stringify(part.algorithm.serialise?.(part.state)))
    const fresh = session(method, { steps: 24, mode: 'live', checkpointEvery: 1 })
    const resumed = session(method, {
      steps: 24,
      mode: 'live',
      start: { state: fresh.algorithm.deserialise?.(json) as never, t: 11 },
    })
    resumed.advance()
    expect(resumed.state.ranks).toEqual(whole.state.ranks)
  })

  it('finds a posterior that is too narrow, with a run driven in slices of a budget', async () => {
    const method = sbcMethod({ ...problem(0.4), replications: 150, nullSimulations: 300 })
    const s = session(method, { steps: 150 })
    const seen: number[] = []
    const r = await runAnytimeAsync(s, {
      snapshot: { everySteps: 50 },
      yieldToPage: () => Promise.resolve(),
      onSnapshot: (x) => seen.push(x.state.done),
    })
    expect(r.reason).toBe('ended')
    expect(seen).toEqual([50, 100, 150])
    const result = method.result(s.state)
    expect(result?.passes).toBe(false)
    expect(result?.quantities[0]?.diagnosis).toMatch(/U-shaped/)
  })
})
