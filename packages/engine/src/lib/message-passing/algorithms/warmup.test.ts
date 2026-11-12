// The damped warm-up of expectation propagation: the first sweeps of a run are damped by at least a half, then the run goes on at its
// own damping. It changes the path and not the fixed point, and a run without it is the run it always was.
import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../../trace/index.ts'
import * as M from '../index.ts'

const graph = () =>
  M.compileGraph(
    {
      variables: [
        { name: 'x', family: { kind: 'gaussian' } },
        { name: 'y', family: { kind: 'bernoulli' }, observed: 1 },
      ],
      factors: [M.gaussianPrior('x', 0, 1), { operator: 'probit', args: ['y', 'x'] }],
    },
    M.defaultOperators(),
  )

describe('the warm-up of an EP run', () => {
  it('damps the first sweeps (the messages after one sweep differ) and reaches the same fixed point', () => {
    const g = graph()
    const plain = runAlgorithm(M.epAlgorithm(g, { tol: 1e-12 }), { maxSteps: 1 })
    const warm = runAlgorithm(M.epAlgorithm(g, { tol: 1e-12, warmup: 3 }), { maxSteps: 1 })
    const before = (plain.state as M.MpState).messages.flat()
    const after = (warm.state as M.MpState).messages.flat()
    expect(after.some((v, i) => Math.abs(v - (before[i] ?? 0)) > 1e-3)).toBe(true)
    const end = (o: { warmup?: number }) => {
      const r = runAlgorithm(M.epAlgorithm(g, { tol: 1e-12, ...o }), { maxSteps: 300 })
      return { mean: Number(M.marginalsOf(g, r.state as M.MpState)['x']?.['mean']), sweeps: r.steps }
    }
    expect(Math.abs(end({ warmup: 3 }).mean - end({}).mean)).toBeLessThan(1e-9)
  })

  it('is no change without it: zero sweeps, or none given, are the same run to the last bit', () => {
    const g = graph()
    const messages = (o: { warmup?: number }) =>
      (runAlgorithm(M.epAlgorithm(g, { tol: 1e-12, ...o }), { maxSteps: 20 }).state as M.MpState).messages
    expect(messages({ warmup: 0 })).toEqual(messages({}))
  })

  it('`auto` is none on a graph with no label that selects a component, and the knob reads off, on and auto', () => {
    const g = graph()
    expect(M.hasExchangeableLabels(g)).toBe(false)
    const auto = runAlgorithm(M.epAlgorithm(g, { tol: 1e-12, warmup: 'auto' }), { maxSteps: 1 }).state as M.MpState
    const none = runAlgorithm(M.epAlgorithm(g, { tol: 1e-12 }), { maxSteps: 1 }).state as M.MpState
    expect(auto.messages).toEqual(none.messages)
    expect(M.warmupOf({ warmup: 'off' })).toEqual({ warmup: 0 })
    expect(M.warmupOf({ warmup: 'on' })).toEqual({ warmup: 3 })
    expect(M.warmupOf({ warmup: 'auto' })).toEqual({ warmup: 'auto' })
    expect(M.warmupOf({})).toEqual({})
  })
})
