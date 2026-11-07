import { describe, expect, it } from 'vitest'
import { zdt1 } from '../../testing/optim-problems.ts'
import { grad } from '../autodiff/index.ts'
import { key } from '../random/index.ts'
import { item, square, stack, sub, sum, type TensorLike, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import { metricSeries, recorder, tensorSeries, tracePolicy } from '../trace/index.ts'
import { nelderMead } from './minimize/derivative-free.ts'
import {
  augmentedChebyshev,
  chebyshev,
  crowdingDistance,
  dominates,
  epsilonConstraint,
  hypervolume,
  hypervolumeContributions,
  nonDominatedSort,
  paretoFront,
  weightedSum,
} from './multiobjective.ts'
import { nsga2, nsga2Algorithm, polynomialMutation, sbxCrossover } from './nsga2.ts'

const v = (x: unknown): Value => x as Value
// Schaffer's problem: f1 = x^2, f2 = (x - 2)^2; Pareto set [0, 2], front f2 = (sqrt f1 - 2)^2
const schaffer = (x: unknown): Value => stack([sum(square(v(x))), sum(square(sub(v(x), 2)))])

describe('dominance, fronts and crowding', () => {
  it('handles equal points, single objectives and empty sets', () => {
    expect(dominates([1, 1], [1, 1])).toBe(false)
    expect(dominates([0], [1])).toBe(true)
    expect(
      paretoFront([
        [1, 1],
        [1, 1],
        [2, 0],
      ]),
    ).toEqual([0, 1, 2])
    expect(
      nonDominatedSort([
        [1, 1],
        [1, 1],
        [2, 2],
      ]),
    ).toEqual([[0, 1], [2]])
    expect(
      toFlat(
        crowdingDistance([
          [0, 1],
          [1, 0],
        ]),
      ),
    ).toEqual([Infinity, Infinity])
    expect(
      toFlat(
        crowdingDistance([
          [1, 1],
          [1, 1],
          [1, 1],
        ]),
      ),
    ).toEqual([Infinity, 0, Infinity])
    expect(() => nonDominatedSort([1, 2])).toThrow()
  })

  it('computes hypervolumes consistently across dimensions', () => {
    // one point: a box
    expect(hypervolume([[1, 2, 3]], [2, 4, 6])).toBe(1 * 2 * 3)
    // a dominated point and a point outside the reference add nothing
    expect(
      hypervolume(
        [
          [1, 3],
          [2, 2],
          [3, 1],
          [3, 3],
          [5, 0],
        ],
        [4, 4],
      ),
    ).toBe(6)
    // two-dimensional sweep and the WFG recursion agree on a front lifted to 3-D with a constant objective
    const pts = [
      [0.1, 0.9],
      [0.4, 0.5],
      [0.7, 0.2],
      [0.9, 0.05],
    ]
    const lifted = pts.map((p) => [...p, 0.5])
    expect(hypervolume(lifted, [1, 1, 1])).toBeCloseTo(0.5 * hypervolume(pts, [1, 1]), 14)
    expect(hypervolume([], [1, 1])).toBe(0)
    expect(() => hypervolume([[1, 2]], [3])).toThrow()
    const c = hypervolumeContributions(
      [
        [1, 3],
        [2, 2],
        [3, 1],
        [3, 3],
      ],
      [4, 4],
    )
    expect(toFlat(c)).toEqual([1, 1, 1, 0])
  })
})

describe('scalarisations', () => {
  it('are differentiable and reach points of the front', () => {
    const ws = weightedSum(schaffer, [0.25, 0.75])
    // d/dx [0.25 x^2 + 0.75 (x - 2)^2] at 0 = -3
    expect(item(grad(ws)(tensor([0])) as Value)).toBe(-3)
    const x = toFlat(nelderMead(ws, [0], { xatol: 1e-10, fatol: 1e-14 }).x as Value)[0] ?? 0
    expect(x).toBeCloseTo(1.5, 6)
    // Chebyshev with equal weights from the ideal point (0, 0): x^2 = (x - 2)^2, x = 1
    const ch = chebyshev(schaffer, [1, 1], [0, 0])
    expect(toFlat(nelderMead(ch, [0.3], { xatol: 1e-10, fatol: 1e-14 }).x as Value)[0]).toBeCloseTo(1, 6)
    const aug = augmentedChebyshev(schaffer, [1, 1], [0, 0], 0.01)
    expect(item(aug([1]))).toBeCloseTo(1 + 0.01 * 2, 14)
  })

  it('epsilon-constraint returns the objective and c(x) >= 0 constraints', () => {
    const { objective, constraints } = epsilonConstraint(schaffer, 1, [0.25, 0])
    expect(constraints).toHaveLength(1)
    expect(item(objective([1]))).toBe(1)
    // f1 = x^2 <= 0.25 at x = 0.4: 0.25 - 0.16 = 0.09 >= 0
    expect(item((constraints[0] as (x: unknown) => Value)([0.4]))).toBeCloseTo(0.09, 14)
  })
})

describe('genetic operators', () => {
  it('SBX keeps the parents mean away from bounds and stays in the box', () => {
    for (let s = 0; s < 20; s++) {
      const [a, b] = sbxCrossover(
        key(s),
        [0.4, 0.5],
        [0.6, 0.45],
        [
          [-10, 10],
          [-10, 10],
        ],
        { probability: 1, variableProbability: 1, eta: 5 },
      )
      for (let i = 0; i < 2; i++) expect((a.data[i] ?? 0) + (b.data[i] ?? 0)).toBeCloseTo([1, 0.95][i] ?? 0, 9)
    }
    const [c, d] = sbxCrossover(
      key(1),
      [0, 1],
      [1, 0],
      [
        [0, 1],
        [0, 1],
      ],
      { probability: 0 },
    )
    expect([toFlat(c), toFlat(d)]).toEqual([
      [0, 1],
      [1, 0],
    ])
  })

  it('polynomial mutation respects its probability and the bounds', () => {
    expect(
      toFlat(
        polynomialMutation(
          key(0),
          [0.3, 0.7],
          [
            [0, 1],
            [0, 1],
          ],
          { probability: 0 },
        ),
      ),
    ).toEqual([0.3, 0.7])
    for (let s = 0; s < 50; s++) {
      const y = polynomialMutation(
        key(s),
        [0, 1],
        [
          [0, 1],
          [0, 1],
        ],
        { probability: 1, eta: 1 },
      )
      for (const vi of toFlat(y)) expect(vi >= 0 && vi <= 1).toBe(true)
    }
  })
})

describe('NSGA-II', () => {
  it('approximates the ZDT1 front and records its hypervolume', () => {
    const rec = recorder()
    const r = nsga2((x) => zdt1(v(x)), new Array(5).fill(0.5), {
      bounds: new Array(5).fill([0, 1]),
      popSize: 40,
      maxIterations: 60,
      seed: 0,
      reference: [1.1, 1.1],
      tracer: rec,
    })
    const front = toArray(r.paretoFront) as number[][]
    // every front point is near f2 = 1 - sqrt(f1)
    for (const [f1 = 0, f2 = 0] of front) expect(f2 - (1 - Math.sqrt(f1))).toBeLessThan(0.05)
    const hv = metricSeries(rec.run(), 'hypervolume').values
    expect((hv.at(-1) ?? 0) > (hv[0] ?? 1)).toBe(true)
    expect(hv.at(-1)).toBeGreaterThan(0.8)
    expect(r.population.shape).toEqual([40, 5])
    expect(r.objectives.shape).toEqual([40, 2])
    expect(r.nfev).toBe(40 * 61)
  })

  it('handles constraints by constrained domination', () => {
    // Schaffer with x >= 1: the Pareto set shrinks to [1, 2]
    const r = nsga2(schaffer, [0], {
      bounds: [[-5, 5]],
      popSize: 24,
      maxIterations: 40,
      seed: 0,
      constraints: [(x) => sub(sum(v(x)), 1)],
    })
    for (const x of r.paretoSet) {
      const xv = (toArray(x as Value) as number[])[0] ?? 0
      expect(xv).toBeGreaterThanOrEqual(1 - 1e-12)
      expect(xv).toBeLessThanOrEqual(2.01)
    }
    expect(r.front.length).toBeGreaterThan(10)
  })

  it('traces objectives and population, and validates its options', () => {
    const rec = recorder()
    nsga2(schaffer, [0], {
      bounds: [[-5, 5]],
      popSize: 7,
      maxIterations: 3,
      seed: 0,
      tracer: rec,
      policy: tracePolicy({
        metrics: true,
        tensors: true,
        roles: ['state', 'param'],
        messages: false,
        frames: false,
        checkpoints: false,
        schedule: { every: 1 },
        detail: 'full',
        maxTensorSize: 4096,
        maxEvents: 1e6,
      }),
    })
    expect(tensorSeries(rec.run(), 'objectives')[0]?.shape).toEqual([8, 2])
    expect(() => nsga2Algorithm(schaffer, [0], { bounds: [[-5, 5]], popSize: 2 })).toThrow()
    expect(() => nsga2Algorithm(schaffer, [0], { bounds: [[-5, Infinity]] })).toThrow()
  })
})

describe('tensor-like inputs (migration, D-009)', () => {
  it('the reference, the bounds and the individuals take a tensor, a typed array or a list; a matrix is a shape error; a traced one is not differentiable', () => {
    const front = [
      [1, 3],
      [2, 2],
      [3, 1],
    ]
    const box: [number, number][] = [
      [0, 1],
      [0, 1],
    ]
    const rules: [string, (v: TensorLike) => unknown][] = [
      ['hypervolume', (v) => hypervolume(front, v)],
      ['hypervolumeContributions', (v) => toFlat(hypervolumeContributions(front, v))],
      ['epsilonConstraint', (v) => epsilonConstraint(schaffer, 0, v).constraints.length],
      ['sbxCrossover', (v) => sbxCrossover(key(0), v, [0.5, 0.5], box, { probability: 1 }).map(toFlat)],
      ['polynomialMutation', (v) => toFlat(polynomialMutation(key(0), v, box, { probability: 1 }))],
    ]
    for (const [name, f] of rules) {
      const values = name === 'hypervolume' || name === 'hypervolumeContributions' ? [4, 4] : [0.25, 0.75]
      const expected = f(values)
      expect(f(Float64Array.from(values)), name).toEqual(expected)
      expect(f(tensor(values)), name).toEqual(expected)
      expect(() => f(tensor([values, values])), name).toThrow(expect.objectContaining({ code: 'shape' }))
      expect(
        () =>
          grad((a: Value) => {
            f(a as TensorLike)
            return sum(a)
          })(tensor(values)),
        name,
      ).toThrow(expect.objectContaining({ code: 'not-differentiable' }))
    }
    expect(crowdingDistance(front).shape).toEqual([3])
    expect(hypervolumeContributions(front, [4, 4]).shape).toEqual([3])
  })
})
