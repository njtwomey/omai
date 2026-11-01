import { describe, expect, it } from 'vitest'
import { concat, exp, mul, neg, sliceAxis, sub, tensor, type Value } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import { bvpContract, bvpSolution, solveBvp, solveBvpAlgorithm } from './boundary.ts'

const sinFun = (_x: unknown, y: Value): Value => concat([sliceAxis(y, 0, 1, 2), neg(sliceAxis(y, 0, 0, 1))], 0)
const sinBc = (ya: Value, yb: Value): Value => concat([sliceAxis(ya, 0, 0, 1), sub(sliceAxis(yb, 0, 0, 1), 1)], 0)
const sinMesh = [0, 0.5, 1, Math.PI / 2]
const sinGuess = tensor([
  [0, 0, 0, 0],
  [1, 1, 1, 1],
])
// Bratu's problem y'' = -2 exp(y), y(0) = y(1) = 0, from a zero guess on five nodes: the mesh is refined.
const bratuFun = (_x: unknown, y: Value): Value =>
  concat([sliceAxis(y, 0, 1, 2), neg(mul(exp(sliceAxis(y, 0, 0, 1)), 2))], 0)
const bratuBc = (ya: Value, yb: Value): Value => concat([sliceAxis(ya, 0, 0, 1), sliceAxis(yb, 0, 0, 1)], 0)
const bratuMesh = Array.from({ length: 5 }, (_, i) => i / 4)
const bratuGuess = tensor([bratuMesh.map(() => 0), bratuMesh.map(() => 0)])

const make = (tol = 1e-6, maxNodes?: number) =>
  solveBvpAlgorithm(bratuFun as never, bratuBc as never, bratuMesh, bratuGuess, {
    tol,
    ...(maxNodes === undefined ? {} : { maxNodes }),
  })

// What solveBvp returned before it became a run of the algorithm (13 significant digits).
const OLD = {
  sin: { iterations: 4, nodes: 37, value: 0.3616154300182 },
  bratu: { iterations: 4, nodes: 49, status: 0, at037: 0.3055610036885, at05: 0.3289524185847 },
  capped: { iterations: 1, nodes: 5, status: 1 },
}

describe('solveBvpAlgorithm', () => {
  it('is solveBvp: run to the end it gives the iterations, mesh and values the function gave before', () => {
    const sine = runAlgorithm(solveBvpAlgorithm(sinFun as never, sinBc as never, sinMesh, sinGuess, { tol: 1e-6 }), {
      maxSteps: 50,
    })
    expect(sine.state.iteration).toBe(OLD.sin.iterations)
    expect(sine.state.x).toHaveLength(OLD.sin.nodes)
    expect(bvpSolution(sine.state, 2).sol(0.37).data[0]).toBeCloseTo(OLD.sin.value, 11)
    expect(sine.status).toBe('converged')
    const bratu = runAlgorithm(make(), { maxSteps: 50 })
    const r = bvpSolution(bratu.state, 2)
    expect([bratu.state.iteration, bratu.state.x.length, r.status]).toEqual([
      OLD.bratu.iterations,
      OLD.bratu.nodes,
      OLD.bratu.status,
    ])
    expect(r.sol(0.37).data[0]).toBeCloseTo(OLD.bratu.at037, 11)
    expect(r.sol(0.5).data[0]).toBeCloseTo(OLD.bratu.at05, 11)
    const fn = solveBvp(bratuFun as never, bratuBc as never, bratuMesh, bratuGuess, { tol: 1e-6 })
    expect(fn.x).toEqual(r.x)
    expect(fn.niter).toBe(r.niter)
    expect(fn.sol(0.5).data[0]).toBe(r.sol(0.5).data[0])
    // The node budget stops the run before it refines, with status 1.
    const capped = runAlgorithm(make(1e-9, 12), { maxSteps: 50 })
    expect([capped.state.iteration, capped.state.x.length, capped.state.status]).toEqual([
      OLD.capped.iterations,
      OLD.capped.nodes,
      OLD.capped.status,
    ])
    expect(capped.status).toBe('failed')
  })

  it('has a state that is JSON, and a run resumes from a checkpoint to the same end', () => {
    const alg = make()
    const whole = runAlgorithm(alg, { maxSteps: 50 })
    const half = runAlgorithm(alg, { maxSteps: 2 })
    expect(half.state.status).toBeNull()
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half.state)))
    const rest = runAlgorithm(alg, { maxSteps: 48, start: { state: stored, step: 2, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
  })

  it('logs the residual, the mesh and the solution by path; the mesh only grows and the residual ends below the tolerance', () => {
    const rec = recorder()
    const run = runAlgorithm(make(), {
      maxSteps: 50,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    const sizes = metric('bvp/mesh-size')
    expect(sizes).toHaveLength(run.steps)
    expect(sizes.every((v, i) => i === 0 || v >= (sizes[i - 1] ?? 0))).toBe(true)
    expect(metric('bvp/bc-residual').at(-1)).toBeLessThan(1e-6)
    expect(Math.max(...run.state.rms)).toBeLessThan(1e-6)
    expect(rec.events.flatMap((e) => (e.type === 'tensor' && e.name === 'bvp/solution' ? [e.shape[1]] : []))).toEqual(
      sizes,
    )
    expect(Object.keys(bvpContract)).toHaveLength(5)
  })
})
