import { describe, expect, it } from 'vitest'
import { optimisers } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { matmul, tensor, toFlat } from '../tensor/index.ts'
import { type Algorithm, policies, recorder, runAlgorithm, stateToJson } from '../trace/index.ts'
import * as NN from './index.ts'

const w = [
  [1.5, -1, 0.2],
  [-0.5, 2, -0.3],
  [0.3, 0.3, 1],
]
const model: NN.LogitsFunction = (x) => matmul(x, tensor(w))
const x = tensor([
  [0.4, -0.2, 0.7],
  [1, 0.5, -0.3],
  [-0.2, 0.9, 0.1],
])
const y = tensor([0, 1, 1], { dtype: 'int32' })

const close = (got: ArrayLike<number>, want: readonly number[]): void => {
  expect(got.length).toBe(want.length)
  for (const [i, v] of want.entries()) expect(got[i]).toBeCloseTo(v, 12)
}

/** Run to the end, and again with a JSON round trip half way: the two end states are the same. */
function resumes<S>(alg: Algorithm<S, string>, steps: number, half: number): void {
  const whole = runAlgorithm(alg, { maxSteps: steps })
  const first = runAlgorithm(alg, { maxSteps: half })
  const stored = JSON.parse(JSON.stringify(alg.serialise?.(first.state)))
  const rest = runAlgorithm(alg, {
    maxSteps: steps - half,
    start: { state: stored, step: first.steps, serialised: true },
  })
  const json = (v: unknown): unknown => JSON.parse(JSON.stringify(stateToJson(v)))
  expect(json(rest.state)).toEqual(json(whole.state))
}

describe('carliniWagnerAlgorithm', () => {
  const opts = { steps: 40, c: 5 }

  it('is carliniWagnerL2: the numbers the function gave before it was an algorithm', () => {
    const r = NN.carliniWagnerL2(model, x, y, opts)
    close(
      toFlat(r.x),
      [
        0.33440534628577495, -0.1958195131047441, 0.6851854911326593, 1, 0.5, -0.3, 0.3459180454959055,
        0.351986772167742, 0.642221581958845,
      ],
    )
    expect(r.success).toEqual([true, true, true])
    close(r.distance, [0.0673765889575231, 0, 0.9446423948876911])
    const s = runAlgorithm(NN.carliniWagnerAlgorithm(model, x, y, opts), { maxSteps: 40 }).state
    expect(Array.from(s.best)).toEqual(Array.from(toFlat(r.x)))
    expect(s.step).toBe(40)
  })

  it('resumes from a JSON checkpoint, optimiser state included', () => {
    resumes(NN.carliniWagnerAlgorithm(model, x, y, opts), 40, 15)
  })

  it('logs its contract by path with the right shapes', () => {
    const rec = recorder()
    runAlgorithm(NN.carliniWagnerAlgorithm(model, x, y, opts), {
      maxSteps: 6,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    for (const p of ['attack/step', 'attack/loss', 'attack/margin', 'attack/norm', 'attack/success'])
      expect(metric(p)).toHaveLength(6)
    expect(metric('attack/step')).toEqual([1, 2, 3, 4, 5, 6])
    const deltas = rec.events.flatMap((e) => (e.type === 'tensor' && e.name === 'attack/delta' ? [e] : []))
    expect(deltas.map((e) => e.shape)).toEqual(Array.from({ length: 6 }, () => [3, 3]))
    expect(Object.keys(NN.carliniWagnerContract)).toContain('attack/delta')
  })

  it('minimises its objective: the loss falls over the run', () => {
    const rec = recorder()
    runAlgorithm(NN.carliniWagnerAlgorithm(model, x, y, { steps: 60, c: 5, learningRate: 0.02 }), {
      maxSteps: 60,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true },
    })
    const loss = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'attack/loss' ? [e.value] : []))
    expect(loss[loss.length - 1]).toBeLessThan(loss[0] as number)
  })

  it('takes any optimiser of the table as its slot', () => {
    for (const kind of ['sgd', 'adam', 'rmsprop', 'adagrad', 'lion'] as const) {
      const optimiser = optimisers.make({ kind })(0.05)
      const s = runAlgorithm(NN.carliniWagnerAlgorithm(model, x, y, { steps: 10, c: 5, optimiser }), {
        maxSteps: 10,
      }).state
      expect(s.step).toBe(10)
      expect(Number.isFinite(s.loss)).toBe(true)
    }
  })
})

describe('deepFoolAlgorithm', () => {
  it('is deepFool: the numbers the function gave before it was an algorithm', () => {
    const r = NN.deepFool(model, x)
    close(
      toFlat(r.x),
      [
        0.35812945945945945, -0.1935583783783784, 0.7225456756756756, 0.7449796, 0.7550204, -0.3, 0.36102039999999996,
        0.33897960000000005, 0.1,
      ],
    )
    expect(r.success).toEqual([true, true, true])
    close(r.distance, [0.047989000259913185, 0.3606533083618117, 0.7934026584479787])
  })

  it('resumes from a JSON checkpoint', () => {
    resumes(NN.deepFoolAlgorithm(model, x), 50, 1)
  })

  it('stops when every input has left its class, and logs the share fooled rising to 1', () => {
    const rec = recorder()
    const run = runAlgorithm(NN.deepFoolAlgorithm(model, x), {
      maxSteps: 50,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    expect(run.state.active.every((a) => !a)).toBe(true)
    expect(run.steps).toBeLessThan(50)
    const share = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'attack/success' ? [e.value] : []))
    expect(share).toHaveLength(run.steps)
    expect(share[share.length - 1]).toBe(1)
    const points = rec.events.flatMap((e) => (e.type === 'tensor' && e.name === 'attack/point' ? [e] : []))
    expect(points[0]?.shape).toEqual([3, 3])
  })

  it('moves an input only while it is on its original class', () => {
    const a = NN.deepFoolAlgorithm(model, x)
    const s0 = a.init(key(0))
    const s1 = a.step(s0, { log: () => {}, emit: () => {}, wants: () => false } as never)
    const s2 = a.step(s1, { log: () => {}, emit: () => {}, wants: () => false } as never)
    s1.active.forEach((alive, i) => {
      if (!alive) expect(toFlat(s2.x).slice(3 * i, 3 * i + 3)).toEqual(toFlat(s1.x).slice(3 * i, 3 * i + 3))
    })
  })
})

describe('squareAlgorithm', () => {
  const opts = { epsilon: 0.6, queries: 30, key: key(3) }

  it('is squareAttack: the numbers the function gave before it was an algorithm', () => {
    const r = NN.squareAttack(model, x, y, opts)
    close(
      toFlat(r.x),
      [
        -0.19999999999999996, 0.39999999999999997, 1.2999999999999998, 0.4, -0.09999999999999998, 0.3,
        0.39999999999999997, 0.30000000000000004, 0.7,
      ],
    )
    expect(r.success).toEqual([true, true, true])
    close(r.distance, [0.6, 0.6, 0.6])
    expect(r.queries).toEqual([1, 1, 8])
  })

  it('resumes from a JSON checkpoint', () => {
    resumes(NN.squareAlgorithm(model, x, y, { ...opts, epsilon: 0.2 }), 30, 7)
  })

  it('spends queries only on inputs not yet fooled, and never loses margin', () => {
    const hard = { epsilon: 0.2, queries: 25, key: key(1) }
    const a = NN.squareAlgorithm(model, x, y, hard)
    let s = a.init(key(0))
    const ctx = { log: () => {}, emit: () => {}, wants: () => false } as never
    for (let q = 0; q < 25 && a.done?.(s, undefined as never) === undefined; q++) {
      const next = a.step(s, ctx)
      for (const [i, m] of next.margins.entries()) expect(m).toBeGreaterThanOrEqual(s.margins[i] as number)
      for (const [i, c] of next.queries.entries()) expect(c).toBe((s.queries[i] as number) + (s.done[i] ? 0 : 1))
      s = next
    }
  })

  it('logs its contract by path', () => {
    const rec = recorder()
    runAlgorithm(NN.squareAlgorithm(model, x, y, { ...opts, epsilon: 0.2 }), {
      maxSteps: 4,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('attack/queries').length).toBeGreaterThan(0)
    expect(metric('attack/step')[0]).toBe(1)
    expect(Object.keys(NN.squareContract)).toContain('attack/queries')
  })
})
