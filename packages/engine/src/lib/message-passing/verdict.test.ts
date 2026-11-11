// What a message-passing run reports when it stops: `converged` only at a real fixed point. A run whose beliefs stopped moving
// because some updates cannot be computed is `stopped` with the reason `stalled`, a fixed point with a belief that has no density
// is `failed` as `degenerate`, an automatic damped restart can recover a stall, and the tape says which and why.
import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import * as M from './index.ts'
import { ImproperMessageError } from './operators/args.ts'
import type { FactorOperator } from './types.ts'

/** An operator that cannot compute its message for its first `failures` calls (an improper cavity), then sends a message about `mean`. */
function reluctant(failures: number): FactorOperator {
  let calls = 0
  return {
    name: 'reluctant',
    description: 'fails for its first calls, then observes x at 3 with precision 2',
    ports: [{ name: 'x', accepts: ['gaussian'] }],
    support: { ep: 'exact' },
    ep: () => {
      calls++
      if (calls <= failures) throw new ImproperMessageError('the cavity has no density yet')
      return { message: [6, 2] }
    },
  }
}

const graphWith = (op: FactorOperator): M.CompiledGraph =>
  M.compileGraph(
    {
      variables: [{ name: 'x', family: { kind: 'gaussian' } }],
      factors: [M.gaussianPrior('x', 0, 1), { operator: 'reluctant', args: ['x'] }],
    },
    { ...M.defaultOperators(), reluctant: op },
  )

describe('a run that cannot compute an update does not converge', () => {
  it('is stopped as stalled, with the reason and the unresolved count, when its beliefs stop moving only for want of updates', () => {
    const g = graphWith(reluctant(Number.POSITIVE_INFINITY))
    const r = runAlgorithm(M.epAlgorithm(g, { restarts: 0 }), { maxSteps: 50 })
    expect(r.status).toBe('stopped')
    expect(r.state.unresolved).toBe(1)
    const verdict = M.mpVerdict(g, r.state, {
      tol: 1e-8,
      maxSweeps: 200,
      schedule: 'serial',
      projection: M.momentProjection,
    })
    expect(verdict.status).toBe('stopped')
    expect(verdict.reason).toMatch(/^stalled: 1 message update could not be computed/)
  })

  it('says so on the tape: the unresolved metric and a verdict message', () => {
    const g = graphWith(reluctant(Number.POSITIVE_INFINITY))
    const rec = recorder()
    runAlgorithm(M.epAlgorithm(g, { restarts: 0 }), { maxSteps: 50, tracer: rec, policy: policies.teaching })
    const events = rec.run().events
    const unresolved = events.filter((e) => e.type === 'metric' && e.name === 'unresolved').at(-1)
    expect(unresolved).toMatchObject({ value: 1 })
    const verdict = events.filter((e) => e.type === 'message' && e.kind === 'verdict').at(-1)
    expect(JSON.stringify(verdict)).toMatch(/stalled/)
    expect(JSON.stringify(verdict)).toMatch(/"stopped"/)
  })

  it('a restart from the start messages recovers a stall that was only for a while, and is counted on the tape', () => {
    // The operator fails twice: sweeps one and two compute nothing, the beliefs stop moving, and the run is restarted damped.
    const g = graphWith(reluctant(2))
    const stalled = runAlgorithm(M.epAlgorithm(g, { restarts: 0 }), { maxSteps: 50 })
    expect(stalled.status).toBe('stopped')
    const g2 = graphWith(reluctant(2))
    const r = runAlgorithm(M.epAlgorithm(g2, { restarts: 1, tol: 1e-10 }), { maxSteps: 200 })
    expect(r.status).toBe('converged')
    expect(r.state.restarts).toBe(1)
    expect(r.state.damping).toBe(0.5)
    expect(r.state.unresolved).toBe(0)
    // The prior N(0, 1) and the message [6, 2] (a Gaussian at 3 with precision 2) give mean 6 / 3 = 2.
    expect(Number(M.marginalsOf(g2, r.state)['x']?.['mean'])).toBeCloseTo(2, 6)
  })

  it('after a damped restart it is stuck on the same update, which is a structural skip: converged, with the count said', () => {
    const g = graphWith(reluctant(Number.POSITIVE_INFINITY))
    const r = runAlgorithm(M.epAlgorithm(g, { restarts: 2, tol: 1e-10 }), { maxSteps: 200 })
    expect(r.state.restarts).toBe(2)
    expect(r.state.unresolved).toBe(1)
    const verdict = M.mpVerdict(g, r.state, {
      tol: 1e-10,
      maxSweeps: 200,
      schedule: 'serial',
      projection: M.momentProjection,
    })
    expect(verdict.status).toBe('converged')
    expect(verdict.reason).toMatch(
      /converged after 2 damped restarts, but 1 message update could not be computed even then/,
    )
  })

  it('the sweep cap counts from the last restart', () => {
    const g = graphWith(reluctant(Number.POSITIVE_INFINITY))
    const r = runAlgorithm(M.epAlgorithm(g, { restarts: 1, tol: 1e-12, maxSweeps: 3 }), { maxSteps: 200 })
    // Two sweeps stall the first attempt, the restart gets three of its own.
    expect(r.state.restartAt).toBe(2)
    expect(r.state.sweeps).toBeGreaterThan(3)
  })

  it('EP restarts once by default, the other rules never', () => {
    expect(M.epDefaults.restarts).toBe(1)
    expect(M.vmpDefaults.restarts).toBe(0)
    expect(M.bpDefaults.restarts).toBe(0)
  })

  it('a run with every update computed is converged and says no more', () => {
    const g = graphWith(reluctant(0))
    const r = runAlgorithm(M.epAlgorithm(g), { maxSteps: 50 })
    expect(r.status).toBe('converged')
    expect(r.state.unresolved).toBe(0)
    expect(r.state.restarts).toBe(0)
  })

  it('a fixed point with a belief that has no density is failed as degenerate', () => {
    const g = M.compileGraph(
      {
        variables: [{ name: 'x', family: { kind: 'gaussian' } }],
        factors: [{ operator: 'prior', args: ['x'], params: { eta: [0, 0] } }],
      },
      M.defaultOperators(),
    )
    const r = runAlgorithm(M.epAlgorithm(g), { maxSteps: 50 })
    expect(r.status).toBe('failed')
    expect(
      M.mpVerdict(g, r.state, { tol: 1e-8, maxSweeps: 200, schedule: 'serial', projection: M.momentProjection }).reason,
    ).toMatch(/^degenerate: the beliefs of x have no density/)
  })

  it('a state saved by an older run, without the new fields, resumes with their starting values', () => {
    const g = graphWith(reluctant(0))
    const alg = M.epAlgorithm(g)
    const saved = JSON.parse(JSON.stringify(alg.serialise?.(runAlgorithm(alg, { maxSteps: 1 }).state))) as Record<
      string,
      unknown
    >
    for (const k of ['unresolved', 'pending', 'damping', 'restarts', 'restartAt']) delete saved[k]
    const restored = alg.deserialise?.(saved as never) as M.MpState
    expect([restored.unresolved, restored.pending, restored.damping, restored.restarts, restored.restartAt]).toEqual([
      0, 0, 0, 0, 0,
    ])
  })
})
