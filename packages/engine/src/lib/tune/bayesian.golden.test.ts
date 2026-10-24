import { describe, expect, it } from 'vitest'
import { decode, expectMatches, type GoldenCase, loadFixture } from '../../testing/golden.ts'
import { asTensor, type Tensor, toFlat } from '../tensor/index.ts'
import {
  expectedImprovement,
  fitGpSurrogate,
  gpPosterior,
  logExpectedImprovement,
  logProbabilityOfImprovement,
  lowerConfidenceBound,
  probabilityOfImprovement,
  upperConfidenceBound,
} from './index.ts'

const fixture = loadFixture('tune/bayesian')

function rows(x: unknown): number[][] {
  const t = decode(x) as Tensor
  const [n = 0, d = 1] = t.shape.length === 1 ? [t.shape[0], 1] : t.shape
  const flat = toFlat(t)
  return Array.from({ length: n }, (_, i) => flat.slice(i * d, (i + 1) * d))
}

function vector(x: unknown): number[] {
  return toFlat(decode(x) as Tensor)
}

const hyper = (c: GoldenCase) => ({
  lengthScales: c.params['lengthScales'] as number[],
  kernelScale: Number(c.params['kernelScale']),
  noise: Number(c.params['noise']),
})

describe('bayesian optimisation against scikit-learn, scipy and optuna', () => {
  for (const c of fixture.cases) {
    it(c.name, () => {
      const best = Number(c.params['best'])
      switch (c.fn) {
        case 'posterior':
        case 'posteriorCategorical': {
          const [x, y, xs] = c.inputs
          const categorical = c.fn === 'posteriorCategorical' ? [false, true] : undefined
          const p = gpPosterior(rows(x), vector(y), rows(xs), hyper(c), {
            ...(categorical === undefined ? {} : { categorical, sizes: [1, 3] }),
          })
          const out = c.output as { mean: unknown; std: unknown; lml?: number }
          expectMatches(asTensor(p.mean), out.mean, c.rtol, c.atol, `${c.name} mean`)
          expectMatches(asTensor(p.std), out.std, c.rtol, c.atol, `${c.name} std`)
          return
        }
        case 'ei':
        case 'pi':
        case 'logPi':
        case 'logEi':
        case 'lcb':
        case 'ucb': {
          const [m, s] = [vector(c.inputs[0]), vector(c.inputs[1])]
          const got = {
            ei: () => expectedImprovement(m, s, best),
            pi: () => probabilityOfImprovement(m, s, best),
            logPi: () => logProbabilityOfImprovement(m, s, best),
            logEi: () => logExpectedImprovement(m, s, best),
            lcb: () => lowerConfidenceBound(m, s, 2),
            ucb: () => upperConfidenceBound(m, s, Number(c.params['beta'])),
          }[c.fn]()
          expectMatches(got, c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'fit': {
          const x = rows(c.inputs[0])
          const categorical = c.params['categorical'] as boolean[]
          const gp = fitGpSurrogate(x, vector(c.inputs[1]), { categorical, sizes: categorical.map((k) => (k ? 3 : 1)) })
          // The MAP objective at our optimum is no worse than at Optuna's (which stops at gtol 1e-2).
          const optuna = c.output as number
          expect(-gp.logPosterior).toBeLessThanOrEqual(optuna + c.rtol * Math.abs(optuna) + c.atol)
          return
        }
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})
