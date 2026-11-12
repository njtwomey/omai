import { describe, expect, it } from 'vitest'
import { bruteTilted } from '../../../testing/message-passing.ts'
import { ctx, gaussian, gp, msg, near } from '../../../testing/operator-checks.ts'
import { operatorMessage } from '../engine/dispatch.ts'
import { momentProjection } from '../engine/projections.ts'
import * as M from '../index.ts'

describe('variational operators equal the expected log factor', () => {
  /** The expected log factor as a function of the target's statistics, up to a constant, against the message's pairing. */
  it('gaussian with a gamma precision: messages to x, mean and precision', () => {
    const bx = M.gaussianEta(0.5, 2)
    const bm = M.gaussianEta(-0.3, 1.5)
    const bl = M.gammaEta(3, 2)
    const args = [msg(gaussian, bx), msg(gaussian, bm), msg(M.gamma, bl)]
    // E over (mu, lambda) of log N(x; mu, 1 / lambda) as a function of x: differences against h x - J x^2 / 2.
    const lambdaMean = 1.5
    const muMean = -0.3
    const toX = M.gaussianGammaPrecision.vmp?.(ctx(0, args, {}, 'vmp')).message as M.Eta
    const g = (x: number): number => -0.5 * lambdaMean * ((x - muMean) ** 2 + 1.5)
    const pair = (e: M.Eta, x: number): number => (e[0] as number) * x - 0.5 * (e[1] as number) * x * x
    expect(g(1.2) - g(0.1)).toBeCloseTo(pair(toX, 1.2) - pair(toX, 0.1), 12)
    const toL = M.gaussianGammaPrecision.vmp?.(ctx(2, args, {}, 'vmp')).message as M.Eta
    // E_{x, mu} log N(x; mu, 1/l) as a function of l: 1/2 log l - l/2 E (x - mu)^2.
    const e2 = 2 + 1.5 + (0.5 + 0.3) ** 2
    const gl = (l: number): number => 0.5 * Math.log(l) - 0.5 * l * e2
    const pl = (l: number): number => (toL[0] as number) * Math.log(l) + (toL[1] as number) * l
    expect(gl(2.5) - gl(0.7)).toBeCloseTo(pl(2.5) - pl(0.7), 12)
    const lp = M.gaussianGammaPrecision.expectedLog?.(ctx(0, args, {}, 'vmp')) as number
    // E log f = 1/2 (E log l - log 2 pi) - 1/2 E l E (x - mu)^2.
    const mo = M.gamma.moments(bl)
    expect(lp).toBeCloseTo(0.5 * ((mo[0] as number) - Math.log(2 * Math.PI)) - 0.5 * lambdaMean * e2, 12)
  })
  it('gaussian-gamma-precision starts from a flat cavity of a latent value: a proper message, the prior predictive', () => {
    // x ~ N(0, 1 / lambda), lambda ~ Gamma(2, 2), nothing yet known of x: the message to x is the matched Student-t (variance E[1/lambda] = rate / (shape - 1) = 2).
    const args = [msg(gaussian, gaussian.flat), M.valueArg(gaussian, 0), msg(M.gamma, M.gammaEta(2, 2))]
    // The operator declares that it needs a density; the engine shows a flat cavity wide (flat-cavity.test.ts).
    expect(M.gaussianGammaPrecision.needsDensity).toBe(true)
    const p = operatorMessage(
      { operator: M.gaussianGammaPrecision, params: {}, power: 1, ports: [] } as never,
      momentProjection,
      0,
      args,
    )
    const message = p.kind === 'message' ? p.eta : []
    expect(message[0]).toBeCloseTo(0, 6)
    expect(1 / (message[1] as number)).toBeCloseTo(2, 2)
  })
  it('gaussian-normal-gamma and the mixture: messages to the normal-gamma and to the assignment', () => {
    const ng = M.normalGammaEta(0.4, 2, 3, 1.5)
    const x = 1.3
    const toTheta = M.gaussianNormalGamma.vmp?.(ctx(1, [M.valueArg(gaussian, x), msg(M.normalGamma, ng)], {}, 'vmp'))
      .message as M.Eta
    // E_x log N(x | mu, 1/l) for the observed x is 1/2 log l - l (x - mu)^2 / 2: a function of the statistics (l mu, l mu^2, l, log l).
    const stat = (mu: number, l: number): number[] => [l * mu, l * mu * mu, l, Math.log(l)]
    const dot = (a: readonly number[], b: readonly number[]): number =>
      a.reduce((s, v, i) => s + v * (b[i] as number), 0)
    const gx = (mu: number, l: number): number => 0.5 * Math.log(l) - 0.5 * l * (x - mu) ** 2
    const d1 = gx(0.7, 2.1) - gx(-0.4, 0.9)
    expect(d1).toBeCloseTo(dot(toTheta, stat(0.7, 2.1)) - dot(toTheta, stat(-0.4, 0.9)), 12)
    // The mixture: responsibility times that message, and E log N under each component to z.
    const dz = M.discrete(2)
    const t0 = M.normalGammaEta(0, 1, 2, 1)
    const t1 = M.normalGammaEta(5, 2, 4, 3)
    const args = [
      M.valueArg(gaussian, x),
      msg(dz, M.discreteEta([0.25, 0.75])),
      msg(M.normalGamma, t0),
      msg(M.normalGamma, t1),
    ]
    const toT1 = M.gaussianMixture.vmp?.(ctx(3, args, { components: 2 }, 'vmp')).message as M.Eta
    near(
      toT1,
      toTheta.map(
        (v, i) => 0.75 * (i === 0 || i === 3 ? (i === 0 ? x : 0.5) : i === 1 ? -0.5 : -0.5 * x * x) * (v === 0 ? 1 : 1),
      ),
      1e-12,
    )
    const toZ = M.gaussianMixture.vmp?.(ctx(1, args, { components: 2 }, 'vmp')).message as M.Eta
    const expectLog = (t: M.Eta): number => {
      const mo = M.normalGamma.moments(t)
      return (
        0.5 * ((mo[3] as number) - Math.log(2 * Math.PI)) -
        0.5 * (x * x * (mo[2] as number) - 2 * x * (mo[0] as number) + (mo[1] as number))
      )
    }
    near(toZ, [expectLog(t0), expectLog(t1)], 1e-12)
  })
  it('gaussian: the expected log density and the messages', () => {
    const bx = M.gaussianEta(0.5, 2)
    const bm = M.gaussianEta(-0.3, 1.5)
    const toX = M.gaussianOperator.vmp?.(ctx(0, [msg(gaussian, bx), msg(gaussian, bm)], { precision: 3 }, 'vmp'))
      .message as M.Eta
    expect(toX[0]).toBeCloseTo(-0.9, 12)
    expect(toX[1]).toBe(3)
    const lp = M.gaussianOperator.expectedLog?.(
      ctx(0, [msg(gaussian, bx), msg(gaussian, bm)], { precision: 3 }, 'vmp'),
    ) as number
    const e = 2 + 1.5 + (0.5 + 0.3) ** 2
    expect(lp).toBeCloseTo(0.5 * Math.log(3 / (2 * Math.PI)) - 1.5 * e, 12)
  })
})

describe('prior, equal and the numeric operator', () => {
  it('the prior is its own message and its mass is the ratio of partitions', () => {
    const eta = M.gaussianEta(1, 0.5)
    const cav = M.gaussianEta(0.2, 2)
    expect(M.prior.ep?.(ctx(0, [msg(gaussian, cav)], { eta })).message).toEqual(eta)
    const brute = bruteTilted(
      [gp(cav)],
      ([x]) => (-0.5 * ((x ?? 0) - 1) ** 2) / 0.5 - 0.5 * Math.log(2 * Math.PI * 0.5),
      0,
    )
    expect(M.prior.logEvidence?.(ctx(0, [msg(gaussian, cav)], { eta }))).toBeCloseTo(brute.logZ, 7)
    expect(M.prior.check?.({ eta }, [gaussian])).toBeUndefined()
    expect(M.prior.check?.({ eta: [1] }, [gaussian])).toBeDefined()
  })
  it('equal passes the other side along', () => {
    const e = M.gammaEta(2, 3)
    expect(M.equal.ep?.(ctx(0, [msg(M.gamma, M.gamma.flat), msg(M.gamma, e)])).message).toEqual(e)
  })
  it('numericOperator: the tilted moments of the clutter factor equal the integral', () => {
    const w = 0.25
    const y = 1.4
    const logf = ([x]: readonly number[]): number =>
      Math.log(
        ((1 - w) * Math.exp(-0.5 * (y - (x ?? 0)) ** 2)) / Math.sqrt(2 * Math.PI) +
          (w * Math.exp((-0.5 * (y * y)) / 10)) / Math.sqrt(20 * Math.PI),
      )
    const op = M.numericOperator({ name: 'clutter', arity: 1, logDensity: logf })
    const ex = M.gaussianEta(0.3, 4)
    const t = op.tilted?.(ctx(0, [msg(gaussian, ex)])) as M.TiltedMoments
    const brute = bruteTilted([gp(ex)], logf, 0)
    near(t.moments, brute.moments, 2e-5)
    expect(t.logZ).toBeCloseTo(brute.logZ, 5)
    const two = M.numericOperator({ name: 'pair', arity: 2, logDensity: ([a, b]) => -0.5 * ((a ?? 0) - (b ?? 0)) ** 2 })
    const e2 = [msg(gaussian, M.gaussianEta(0, 2)), msg(gaussian, M.gaussianEta(1, 3))]
    const tp = two.tilted?.(ctx(1, e2)) as M.TiltedMoments
    near(
      tp.moments,
      bruteTilted([gp(M.gaussianEta(0, 2)), gp(M.gaussianEta(1, 3))], ([a, b]) => -0.5 * ((a ?? 0) - (b ?? 0)) ** 2, 1)
        .moments,
      1e-6,
    )
  })
  it('the registry lists the operators by name and checks its ports', () => {
    const ops = M.defaultOperators()
    for (const name of ['prior', 'gaussian', 'probit', 'plus', 'table', 'select']) expect(ops[name]?.name).toBe(name)
    expect(Object.keys(ops).length).toBeGreaterThanOrEqual(26)
  })
})
