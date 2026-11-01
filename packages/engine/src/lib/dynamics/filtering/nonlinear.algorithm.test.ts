import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { add, item, mul, sin, sliceAxis, stack, sum, toFlat, type Value } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import { kalmanFilter } from './kalman.ts'
import {
  ensembleFilterContract,
  ensembleKalmanAlgorithm,
  ensembleKalmanFilter,
  extendedKalmanAlgorithm,
  extendedKalmanFilter,
  julierSigmaPoints,
  type NonlinearModel,
  nonlinearFilterContract,
  unscentedFilterContract,
  unscentedKalmanAlgorithm,
  unscentedKalmanFilter,
} from './nonlinear.ts'
import { nonlinearFilters, simulateStateSpace, stateSpaceModels } from './nonlinear-options.ts'

const at = (x: Value, i: number): Value => sum(sliceAxis(x, 0, i, i + 1))
const DT = 0.1
const pendulum: NonlinearModel = {
  f: (x) => stack([add(at(x, 0), mul(at(x, 1), DT)), add(at(x, 1), mul(sin(at(x, 0)), -DT * 9.81))]),
  h: (x) => stack([sin(at(x, 0))]),
  Q: [
    [0.001, 0],
    [0, 0.002],
  ],
  R: [[0.05]],
  x0: [0.5, 0],
  P0: [
    [0.1, 0],
    [0, 0.1],
  ],
}
const zs = [[0.5], [0.52], [0.4], [0.3], [0.1], [-0.1]]

// The numbers the functions returned before they became runs of the algorithms (13 significant digits).
const OLD = {
  ekf: {
    means: [
      0.5143260429284, -0.4810035197809, 0.4951007891994, -1.000974362579, 0.3991863120406, -1.472678898885,
      0.263142444118, -1.858207589054, 0.08247676806213, -2.107601150697, -0.1208612143896, -2.17487359816,
    ],
    logLikelihood: 2.160584156668,
  },
  ukf: {
    means: [
      0.5307786222761, -0.4703789072564, 0.5098266570511, -0.9907543855747, 0.4120668086596, -1.465535947198,
      0.2742178023313, -1.859822153525, 0.09102188863874, -2.121058429817, -0.1159376081338, -2.200517485209,
    ],
    logLikelihood: 2.180675603168,
  },
  ukfJulier: {
    means: [
      0.5309467233913, -0.4691820408643, 0.5111827420064, -0.9893734978324, 0.4132842553241, -1.464817766878,
      0.2755203643488, -1.860098562533, 0.09206884578446, -2.122854092562, -0.115246092459, -2.203604312836,
    ],
    logLikelihood: 2.194539582097,
  },
  enkf: {
    means: [
      0.4931286063986, -0.4726851699349, 0.478879107765, -0.9996339413934, 0.3757691447566, -1.446393434643,
      0.2345573397919, -1.818242375882, 0.06642609675907, -1.994979630342, -0.1339474157143, -2.048306437593,
    ],
    logLikelihood: 2.088158643844,
  },
}

const close = (a: ArrayLike<number>, b: readonly number[]): void => {
  expect(a.length).toBe(b.length)
  b.forEach((v, i) => {
    expect(Math.abs((a[i] ?? 0) - v)).toBeLessThan(1e-11 * Math.max(1, Math.abs(v)))
  })
}

describe('nonlinear Kalman algorithms', () => {
  it('are the filters: the algorithm run to the end gives the numbers the functions gave before', () => {
    const cases = [
      ['ekf', extendedKalmanAlgorithm(pendulum, zs), extendedKalmanFilter(pendulum, zs)],
      ['ukf', unscentedKalmanAlgorithm(pendulum, zs), unscentedKalmanFilter(pendulum, zs)],
      [
        'ukfJulier',
        unscentedKalmanAlgorithm(pendulum, zs, { points: julierSigmaPoints(2), redraw: true }),
        unscentedKalmanFilter(pendulum, zs, { points: julierSigmaPoints(2), redraw: true }),
      ],
      [
        'enkf',
        ensembleKalmanAlgorithm(pendulum, zs, { size: 30, key: key(3) }),
        ensembleKalmanFilter(pendulum, zs, { size: 30, key: key(3) }),
      ],
    ] as const
    for (const [name, alg, fn] of cases) {
      const s = runAlgorithm(alg, { maxSteps: zs.length }).state
      expect(s.t).toBe(zs.length)
      close(toFlat(stack([...s.means])), OLD[name].means)
      close(toFlat(fn.means), OLD[name].means)
      expect(item(s.logLik)).toBeCloseTo(OLD[name].logLikelihood, 10)
      expect(item(fn.logLikelihood)).toBe(item(s.logLik))
    }
  })

  it('have a state that is JSON, and a run resumes from a checkpoint to the same end', () => {
    const algs = [
      extendedKalmanAlgorithm(pendulum, zs),
      unscentedKalmanAlgorithm(pendulum, zs),
      ensembleKalmanAlgorithm(pendulum, zs, { size: 20, key: key(5) }),
    ]
    for (const alg of algs) {
      const whole = runAlgorithm(alg, { maxSteps: zs.length }).state
      const half = runAlgorithm(alg, { maxSteps: 3 }).state
      expect(half.t).toBe(3)
      const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
      const rest = runAlgorithm(alg, { maxSteps: 3, start: { state: stored, step: 3, serialised: true } })
      expect(rest.steps).toBe(zs.length)
      expect(stateToJson(rest.state)).toEqual(stateToJson(whole))
    }
  })

  it('log their paths with the right shapes', () => {
    const cases = [
      [extendedKalmanAlgorithm(pendulum, zs), nonlinearFilterContract],
      [unscentedKalmanAlgorithm(pendulum, zs), unscentedFilterContract],
      [ensembleKalmanAlgorithm(pendulum, zs, { size: 12, key: key(1) }), ensembleFilterContract],
    ] as const
    for (const [alg, contract] of cases) {
      const rec = recorder()
      runAlgorithm(alg as never, {
        maxSteps: 4,
        tracer: rec,
        policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['state', 'output'], detail: 'full' },
      })
      const shapeOf = (name: string): number[][] =>
        rec.events.flatMap((e) => (e.type === 'tensor' && e.name === name ? [[...e.shape]] : []))
      expect(Object.keys(contract)).toContain('filter/innovation')
      expect(shapeOf('filter/mean')).toEqual([[2], [2], [2], [2]])
      expect(shapeOf('filter/cov')).toEqual([
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
      ])
      expect(shapeOf('filter/predicted-mean')).toHaveLength(4)
      expect(shapeOf('filter/innovation')).toEqual([[1], [1], [1], [1]])
      const ll = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'filter/log-likelihood' ? [e.value] : []))
      expect(ll).toHaveLength(4)
      if ('filter/sigma-points' in contract) expect(shapeOf('filter/sigma-points')).toEqual(Array(4).fill([5, 2]))
      if ('filter/ensemble' in contract) expect(shapeOf('filter/ensemble')).toEqual(Array(4).fill([12, 2]))
    }
  })

  it('reduce to the Kalman filter on a linear model (the extended filter)', () => {
    const model = stateSpaceModels.make({ kind: 'range', offset: 0.0001 })
    const lin = {
      f: model.f,
      h: (x: Value) => stack([at(x, 0)]),
      Q: model.Q,
      R: model.R,
      x0: model.x0,
      P0: model.P0,
    }
    const F = [
      [1, 0.5],
      [0, 1],
    ]
    const z = [[0.4], [1.2], [1.4], [2.1]]
    const kf = toFlat(kalmanFilter({ F, H: [[1, 0]], Q: lin.Q, R: lin.R, x0: lin.x0, P0: lin.P0 }, z).means)
    const ekf = toFlat(runAlgorithm(extendedKalmanAlgorithm(lin, z), { maxSteps: 4 }).state.means.at(-1) as Value)
    expect(Math.abs((ekf[0] ?? 0) - (kf[kf.length - 2] ?? 0))).toBeLessThan(1e-9)
  })

  it('inflation widens the analysis ensemble about its mean and changes nothing at one', () => {
    const run = (inflation?: number) =>
      runAlgorithm(
        ensembleKalmanAlgorithm(pendulum, zs, {
          size: 40,
          key: key(2),
          ...(inflation === undefined ? {} : { inflation }),
        }),
        { maxSteps: 1 },
      )
    const base = run().state
    expect(stateToJson(run(1).state)).toEqual(stateToJson(base))
    // The first analysis is the same ensemble scaled about its mean: the mean is unchanged, the spread larger.
    const wide = run(1.5).state
    close(toFlat(wide.x), Array.from(toFlat(base.x)))
    expect(item(sum(wide.P))).toBeGreaterThan(item(sum(base.P)))
  })

  it('run every option of the filter table', () => {
    const model = stateSpaceModels.make({ kind: 'pendulum' })
    const { observations } = simulateStateSpace(model, 5, key(7))
    for (const name of nonlinearFilters.names) {
      const alg = nonlinearFilters.make({ kind: name })(model, observations, key(1))
      const s = runAlgorithm(alg, { maxSteps: 5 }).state
      expect(s.t).toBe(5)
      expect(Number.isFinite(item(s.logLik))).toBe(true)
    }
    expect(nonlinearFilters.where((p) => p.stochastic)).toEqual(['ensemble'])
  })
})
