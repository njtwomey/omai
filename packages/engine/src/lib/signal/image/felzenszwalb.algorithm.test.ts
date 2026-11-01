import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../../trace/index.ts'
import { felzenszwalb, felzenszwalbAlgorithm, felzenszwalbContract, felzenszwalbLabels } from './segmentation.ts'
import { felzenszwalbCoarseness } from './segmentation-options.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const image = Array.from({ length: 8 }, (_, i) =>
  Array.from({ length: 8 }, (_, j) => (j < 4 ? 0 : 1) + (i === 3 && j === 1 ? 0.02 : 0)),
)

describe('felzenszwalbAlgorithm', () => {
  it('merges one edge per step, loses a component per merge and ends on the labels of felzenszwalb', () => {
    const run = runAlgorithm(felzenszwalbAlgorithm(image, { minSize: 10 }), { maxSteps: 2000 })
    expect(run.status).toBe('converged')
    // 8-connected edges, two passes.
    const edges = 8 * 7 + 7 * 8 + 7 * 7 + 7 * 7
    expect(run.steps).toBe(2 * edges)
    expect(run.state.components).toBe(new Set(toFlat(felzenszwalbLabels(run.state))).size)
    expect(toFlat(felzenszwalb(image, { minSize: 10 }))).toEqual(toFlat(felzenszwalbLabels(run.state)))
    expect(run.state.components).toBe(2)
  })

  it('logs merges, a falling component count and the thresholds', () => {
    const rec = recorder()
    runAlgorithm(felzenszwalbAlgorithm(image, { minSize: 10 }), {
      maxSteps: 2000,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    const components = metric('seg/components')
    const merges = metric('seg/merge')
    expect(components[0]).toBe(64 - (merges[0] ?? 0))
    for (let i = 1; i < components.length; i++) expect(components[i]).toBe((components[i - 1] ?? 0) - (merges[i] ?? 0))
    expect(metric('seg/threshold')).toHaveLength(components.length)
    expect(Object.keys(felzenszwalbContract)).toHaveLength(4)
  })

  it('serialises mid-run and resumes to the same labels', () => {
    const alg = felzenszwalbAlgorithm(image, { minSize: 10 })
    const whole = runAlgorithm(alg, { maxSteps: 2000 })
    const part = runAlgorithm(alg, { maxSteps: 300 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 2000,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 300, serialised: true },
    })
    expect(toFlat(felzenszwalbLabels(rest.state))).toEqual(toFlat(felzenszwalbLabels(whole.state)))
    expect(rest.state.components).toBe(whole.state.components)
  })

  it('rejects images of the wrong rank', () => {
    expect(() => felzenszwalbAlgorithm([1, 2, 3])).toThrow(/images are/)
  })
})

describe('felzenszwalbCoarseness', () => {
  it('names the levels and builds options', () => {
    expect(felzenszwalbCoarseness.names).toEqual(['fine', 'medium', 'coarse'])
    expect(felzenszwalbCoarseness.make({ kind: 'coarse' }).scale).toBe(500)
    expect(felzenszwalbCoarseness.make({ kind: 'fine', minSize: 3 }).minSize).toBe(3)
  })
})
