import { describe, expect, it } from 'vitest'
import { toArray, toFlat } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../../trace/index.ts'
import { place, placeAlgorithm, placeContract, placementOf } from './placement.ts'
import { placementSchemes } from './placement-options.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const a = [
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
  [1, -1, 2, 0.5],
]
const b = [
  [0, 0],
  [1, 0],
  [0, 0],
  [0, 1],
]
const poles = [-1, -2, -3, -4]

describe('placeAlgorithm', () => {
  it('sweeps the pole pairs one sweep per step and reaches the poles of place', () => {
    const run = runAlgorithm(placeAlgorithm(a, b, poles), { maxSteps: 40 })
    expect(['converged', 'completed']).toContain(run.status)
    expect(run.steps).toBe(run.state.iterations)
    const result = placementOf(a, b, poles, run.state)
    expect(toArray(place(a, b, poles).gain)).toEqual(toArray(result.gain))
    const got = result.computedPoles.map((p) => p.re).sort((x, y) => x - y)
    expect(got.length).toBe(4)
    for (const [i, p] of [-4, -3, -2, -1].entries()) expect(got[i]).toBeCloseTo(p, 6)
  })

  it('does not worsen the conditioning of the eigenvector matrix as it sweeps', () => {
    const rec = recorder()
    runAlgorithm(placeAlgorithm(a, b, poles), {
      maxSteps: 40,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, detail: 'full' },
    })
    const cond = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'place/conditioning' ? [e.value] : []))
    expect(cond.length).toBeGreaterThan(0)
    expect(cond.at(-1) ?? Infinity).toBeLessThanOrEqual((cond[0] ?? 0) * (1 + 1e-6))
    expect(Object.keys(placeContract)).toHaveLength(4)
  })

  it('needs no sweeps for a single input or a full-rank B, and resumes from JSON', () => {
    const single = runAlgorithm(
      placeAlgorithm(
        [
          [0, 1],
          [0, 0],
        ],
        [[0], [1]],
        [-1, -2],
      ),
      { maxSteps: 5 },
    )
    expect([single.status, single.steps]).toEqual(['converged', 0])
    const full = runAlgorithm(
      placeAlgorithm(
        [
          [0, 1],
          [0, 0],
        ],
        [
          [1, 0],
          [0, 1],
        ],
        [-1, -2],
      ),
      { maxSteps: 5 },
    )
    expect(full.steps).toBe(0)
    expect(
      toFlat(
        placementOf(
          [
            [0, 1],
            [0, 0],
          ],
          [
            [1, 0],
            [0, 1],
          ],
          [-1, -2],
          full.state,
        ).gain,
      ),
    ).toHaveLength(4)
    const alg = placeAlgorithm(a, b, poles, { method: 'KNV0', maxIterations: 6 })
    const whole = runAlgorithm(alg, { maxSteps: 10 })
    const part = runAlgorithm(alg, { maxSteps: 1 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 10,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 1, serialised: true },
    })
    expect(Array.from(rest.state.X)).toEqual(Array.from(whole.state.X))
    expect(rest.state.iterations).toBe(whole.state.iterations)
    expect(() => placeAlgorithm(a, b, [-1, -2, -3])).toThrow(/give 4 poles/)
    expect(placementSchemes.make({ kind: 'knv0' }).method).toBe('KNV0')
  })
})
