import type { DecisionSurface } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { smoothSolvers } from '../../lib/optim/index.ts'
import { defaults } from '../../lib/space/index.ts'
import { type Config, metricSeries, session } from '../../lib/trace/index.ts'
import { linearClassifier } from './linearClassifier.ts'

// Two separable-ish clouds on the plane.
const x = [
  [-1.2, -0.8],
  [-0.9, -1.1],
  [-1.5, -0.3],
  [-0.6, -0.9],
  [0.9, 1.1],
  [1.2, 0.7],
  [0.6, 1.3],
  [1.4, 1.0],
]
const y = [0, 0, 0, 0, 1, 1, 1, 1]

describe('linearClassifier', () => {
  it('fits the same objective with every solver, each logging its own way, and draws the surface at any step', () => {
    const method = linearClassifier({ x, y })
    for (const name of smoothSolvers.names) {
      const solver = { ...(defaults(smoothSolvers.space.options[name] as never) as object), kind: name }
      const s = session(method, { mode: 'eager', steps: 60, config: { solver } as Partial<Config> })
      const objective = metricSeries(s.run(), 'solver/objective').values
      expect(objective.length, name).toBeGreaterThan(0)
      expect(objective.at(-1) as number, name).toBeLessThan(objective[0] as number)
      expect(metricSeries(s.run(), 'model/accuracy').values.at(-1), name).toBe(1)
      // Only the stochastic mode has a learning rate; its path is in the contract (it used to be dropped silently).
      expect(metricSeries(s.run(), 'solver/lr').values.length > 0, name).toBe(name === 'gradient')
      const surface = s.probe(Math.min(2, s.t), 'model/decision-surface') as DecisionSurface
      expect(surface.rows?.points).toHaveLength(8)
      expect(surface.probabilities).toHaveLength(2)
    }
  })

  it('probes the margin of every row and the rows on the wrong side, at any step', () => {
    const s = session(linearClassifier({ x, y }), { mode: 'eager', steps: 60 })
    const margins = s.probe(s.t, 'model/margins') as number[]
    const errors = s.probe(s.t, 'model/errors') as { sets: { misclassified: number[] } }
    expect(margins).toHaveLength(8)
    expect(errors.sets.misclassified).toEqual(margins.flatMap((m, i) => (m <= 0 ? [i] : [])))
    // At the start every score is zero, so no row is on its class's side.
    expect((s.probe(0, 'model/margins') as number[]).every((m) => m === 0)).toBe(true)
  })

  it('trains with every smooth loss it offers, each lowering its own objective', () => {
    const method = linearClassifier({ x, y })
    const options =
      (method.config.spaces as unknown as Record<string, { options: Record<string, unknown> }>)['loss']?.options ?? {}
    for (const name of Object.keys(options)) {
      const loss = { ...(defaults(options[name] as never) as object), kind: name }
      const s = session(method, { mode: 'eager', steps: 80, config: { loss } as Partial<Config> })
      const objective = metricSeries(s.run(), 'solver/objective').values
      expect(objective.length, name).toBeGreaterThan(0)
      expect(objective.at(-1) as number, name).toBeLessThan(objective[0] as number)
      expect(Number.isFinite(objective.at(-1)), name).toBe(true)
    }
  })

  it('offers only the smooth losses and penalties, and the loss is a knob of the objective', () => {
    const space = linearClassifier({ x, y }).config.spaces as Record<string, { options: object }>
    expect(Object.keys(space['loss']?.options ?? {})).toEqual([
      'logistic',
      'squaredHinge',
      'exponential',
      'modifiedHuber',
      'focal',
      'smoothedLogistic',
    ])
    expect(Object.keys(space['penalty']?.options ?? {})).toEqual(['none', 'l2'])
  })
})
