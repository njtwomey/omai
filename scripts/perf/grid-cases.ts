/**
 * The typical integrands the performance harness (`scripts/ep-perf.ts`) and the tensorisation study (`scripts/perf/`) run through the
 * shared grid integrator (`message-passing/numerics/_grid.ts`), one per profile: a Laplace likelihood with a random log scale (the
 * `pair` profile, with the kink of the absolute value), a bivariate normal with a random precision and correlation (`axes`, three
 * axes), a Student-t with a random scale and degrees of freedom (`triple`) and a softmax-like sum over eight classes along one
 * coordinate (`line`). The integrands are the scalar closures the operators supply today; each case counts how often the grid calls
 * it, so the cost can be said per node. Not part of the library: a measurement fixture.
 */

import {
  betaAxis,
  gammaAxis,
  gaussianAxis,
  integrateAxes,
  integrateLine,
  integratePair,
  integrateTriple,
} from '../../packages/engine/src/lib/message-passing/numerics/_grid.ts'
import { lgammaOf } from '../../packages/engine/src/lib/message-passing/numerics/_scalar.ts'

/** The observed outcome of the pair, axes and triple integrands. */
export const Y = 0.7

/** One case: a name, the profile it runs in, a function that runs it once and the number of integrand calls of that run. */
export interface GridCase {
  readonly name: 'pair' | 'axes' | 'triple' | 'line'
  readonly describe: string
  /** Runs the integral once; returns its log mass, the nodes of the final grid and how many times the integrand was called. */
  readonly run: () => { readonly logMass: number; readonly nodes: number; readonly evals: number }
}

/** The log of a bivariate normal at `(y1, y2)` with mean `(m, m)`, precision `exp(t)` and correlation `2u - 1`, `s = logit(u)`. */
export function binormalLog(m: number, t: number, s: number): number {
  const rho = 2 / (1 + Math.exp(-s)) - 1
  const d1 = 0.4 - m
  const d2 = -0.3 - m
  const q = (d1 * d1 - 2 * rho * d1 * d2 + d2 * d2) / (1 - rho * rho)
  return t - Math.log(2 * Math.PI) - 0.5 * Math.log(1 - rho * rho) - 0.5 * Math.exp(t) * q
}

/** The log of a Student-t density at `Y` with location `x0`, log scale `t` and log degrees of freedom `u`. */
export function studentLog(x0: number, t: number, u: number): number {
  const nu = Math.exp(u)
  const z = (Y - x0) * Math.exp(-t)
  return (
    lgammaOf((nu + 1) / 2) -
    lgammaOf(nu / 2) -
    0.5 * Math.log(nu * Math.PI) -
    t -
    ((nu + 1) / 2) * Math.log1p((z * z) / nu)
  )
}

/** The log of the normaliser of eight softmax classes at the auxiliary coordinate `x`: `log sum_k exp(a_k x + b_k)`. */
export function softmaxLog(x: number): number {
  const e = (k: number): number => (0.3 * k - 1) * x - 0.1 * k * k
  let top = Number.NEGATIVE_INFINITY
  for (let k = 0; k < 8; k++) top = Math.max(top, e(k))
  let s = 0
  for (let k = 0; k < 8; k++) s += Math.exp(e(k) - top)
  return top + Math.log(s) - x
}

/**
 * The cases, each timed through the library's own entry point.
 *
 * @returns The four cases.
 */
export function gridCases(): readonly GridCase[] {
  return [
    {
      name: 'pair',
      describe: 'Laplace likelihood, Gaussian location (kink at the outcome) x gamma scale (log coordinate)',
      run: () => {
        let evals = 0
        const r = integratePair([gaussianAxis(0.4, 1.0, [Y]), gammaAxis(4, 3)], (t) => {
          // What depends on the second coordinate alone is computed once per node of it, as the operators' column form does.
          const s = Math.exp(-t)
          return (x0) => {
            evals += 1
            return -t - s * Math.abs(Y - x0)
          }
        })
        return { logMass: r.logMass, nodes: r.weights.length, evals }
      },
    },
    {
      name: 'axes',
      describe: 'bivariate normal, Gaussian mean x gamma precision x beta correlation (three axes, equal panels)',
      run: () => {
        let evals = 0
        const r = integrateAxes([gaussianAxis(0, 1), gammaAxis(3, 2), betaAxis(2, 2)], (t) => {
          evals += 1
          return binormalLog(t[0] as number, t[1] as number, t[2] as number)
        })
        return { logMass: r.logMass, nodes: r.weights.length, evals }
      },
    },
    {
      name: 'triple',
      describe: 'Student-t, Gaussian location x gamma scale x gamma degrees of freedom (three axes, column form)',
      run: () => {
        let evals = 0
        const r = integrateTriple([gaussianAxis(0.4, 1.0), gammaAxis(4, 3), gammaAxis(3, 0.5)], (t, u) => (x0) => {
          evals += 1
          return studentLog(x0, t, u)
        })
        return { logMass: r.logMass, nodes: 0, evals }
      },
    },
    {
      name: 'line',
      describe: 'softmax normaliser over eight classes along one Gaussian coordinate',
      run: () => {
        let evals = 0
        const r = integrateLine(gaussianAxis(0, 4), (x) => {
          evals += 1
          return softmaxLog(x)
        })
        return { logMass: r.logMass, nodes: r.weights.length, evals }
      },
    },
  ]
}
