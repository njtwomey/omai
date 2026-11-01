import { describe, expect, it } from 'vitest'
import { mul, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { sparsityOf } from './pruning.ts'
import { iterativeMagnitudePruning, iterativeMagnitudePruningAlgorithm } from './pruning-iterative.ts'
import { pruneRoundSchedules, pruneScopes } from './pruning-options.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const policy = {
  ...policies.minimal,
  metrics: true,
  tensors: true,
  roles: ['output' as const, 'state' as const],
  detail: 'full' as const,
}
const metricsOf = (rec: ReturnType<typeof recorder>, name: string): number[] =>
  rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))

const initial = { w: tensor([[1, -2, 3, -4, 0.5, 6, -0.25, 8]]), v: tensor([[0.1, 5, -7, 0.3]]) }
// "Training" doubles every weight: the survivors keep their order, so the masks are those of the initial magnitudes.
const train = (p: Record<string, unknown>): Record<string, Value> =>
  Object.fromEntries(Object.entries(p).map(([k, v]) => [k, mul(v as Value, 2)]))

describe('iterativeMagnitudePruningAlgorithm', () => {
  const options = { finalSparsity: 0.75, rounds: 3 }

  it('prunes a further share each round and the function is that run', () => {
    const rec = recorder()
    const alg = iterativeMagnitudePruningAlgorithm(initial, train as never, options)
    const run = runAlgorithm(alg, { maxSteps: 10, tracer: rec, policy })
    expect(run.steps).toBe(3)
    expect(metricsOf(rec, 'prune/round')).toEqual([1, 2, 3])
    const sparsity = metricsOf(rec, 'prune/sparsity')
    expect(sparsity[2]).toBeCloseTo(0.75, 12)
    expect(sparsity.every((v, i) => i === 0 || v > (sparsity[i - 1] ?? 0))).toBe(true)
    const r = iterativeMagnitudePruning(initial, train as never, options)
    expect(toArray(r.masks['w'] as never)).toEqual(toArray(run.state.masks['w'] as never))
    expect(r.rounds.map((x) => x.sparsity)).toEqual(sparsity)
    // 12 weights, 75% pruned: 9 zeroed in the ticket's masks.
    const total = [r.masks['w'], r.masks['v']].flatMap((m) => Array.from(toFlat(m as never)))
    expect(total.filter((v) => v === 0)).toHaveLength(9)
    expect(sparsityOf(r.ticket['v'] as never) + sparsityOf(r.ticket['w'] as never)).toBeGreaterThan(0)
  })

  it('logs the masks as one vector and the loss of the evaluator', () => {
    const rec = recorder()
    const alg = iterativeMagnitudePruningAlgorithm(initial, train as never, {
      ...options,
      evaluate: (_trained, _masks, round) => 10 - round,
    })
    runAlgorithm(alg, { maxSteps: 10, tracer: rec, policy })
    expect(metricsOf(rec, 'prune/loss')).toEqual([9, 8, 7])
    const masks = rec.events.filter((e) => e.type === 'tensor' && e.name === 'prune/mask')
    expect(masks).toHaveLength(3)
    expect(masks[0]?.type === 'tensor' ? masks[0].shape : []).toEqual([12])
  })

  it('resumes from a JSON checkpoint with tensors in its parameter trees', () => {
    const alg = iterativeMagnitudePruningAlgorithm(initial, train as never, options)
    const whole = runAlgorithm(alg, { maxSteps: 10 })
    const half = runAlgorithm(alg, { maxSteps: 1 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 9, start: { state: stored, step: 1, serialised: true } })
    expect(rest.steps).toBe(whole.steps)
    expect(toArray(rest.state.masks['w'] as never)).toEqual(toArray(whole.state.masks['w'] as never))
    expect(toArray(rest.state.params['v'] as never)).toEqual(toArray(whole.state.params['v'] as never))
    expect(rest.state.history).toHaveLength(3)
  })

  it('takes its scope and schedule from option tables', () => {
    const layerwise = iterativeMagnitudePruningAlgorithm(initial, train as never, {
      ...options,
      scope: { kind: 'layerwise' },
      schedule: { kind: 'linear' },
    })
    const run = runAlgorithm(layerwise, { maxSteps: 10 })
    // Every tensor is pruned by the same share: 0.75 of 8 and of 4.
    expect(toFlat(run.state.masks['w'] as never).filter((v) => v === 0)).toHaveLength(6)
    expect(toFlat(run.state.masks['v'] as never).filter((v) => v === 0)).toHaveLength(3)
    expect(run.state.history.map((h) => h.sparsity)).toEqual([0.25, 0.5, 0.75])
    expect(pruneScopes.names).toEqual(['global', 'layerwise'])
    expect(pruneRoundSchedules.make({ kind: 'polynomial', power: 3 })(3, 3, 0.8)).toBeCloseTo(0.8, 12)
    expect(pruneRoundSchedules.properties({ kind: 'linear' }).frontLoaded).toBe(false)
  })

  it('rejects a final sparsity outside [0, 1]', () => {
    expect(() => iterativeMagnitudePruningAlgorithm(initial, train as never, { finalSparsity: 1.5 })).toThrow(
      /sparsity/,
    )
  })
})
