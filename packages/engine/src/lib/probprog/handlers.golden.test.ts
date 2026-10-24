import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { MODELS } from '../../testing/probprog.ts'
import { bernoulli, beta, categorical, dirichlet, exponential, halfCauchy, normal, poisson } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { item, ones, type Tensor, tensor, zeros } from '../tensor/index.ts'
import {
  block,
  condition,
  doIntervention,
  enumerate,
  logJoint,
  type Model,
  mask,
  observe,
  plate,
  sample,
  scale,
  seed,
  substitute,
  traceOf,
} from './index.ts'

const SIGMA = [15, 10, 16, 11]
const SCHOOLS = [28, 8, -3, 7]
const schools = MODELS['schools'] as Model

function values(x: unknown): Record<string, Tensor> {
  return Object.fromEntries(Object.entries(x as Record<string, unknown>).map(([k, v]) => [k, decode(v) as Tensor]))
}

const shapesModel: Model = () => {
  sample('a', normal(0, 1), { sampleShape: [3] })
  plate('outer', 2, () => {
    sample('b', normal(0, 1))
    plate('inner', 4, () => sample('c', normal(zeros([1]), 1)))
  })
  plate('rows', 5, () => sample('d', dirichlet(ones([3]))), { dim: -2 })
  plate('big', 10, () => sample('e', exponential(1)), { subsampleSize: 4 })
}

const mixed: Model = () => {
  const p = sample('p', beta(2, 2))
  plate('n', 3, () => {
    sample('c', bernoulli({ probs: p }))
    sample('k', poisson(2))
  })
  sample('d', categorical({ probs: [0.2, 0.8] }))
}

function maskedSchools(m: readonly boolean[]): Model {
  return () => {
    const mu = sample('mu', normal(0, 5))
    const tau = sample('tau', halfCauchy(5))
    plate('J', 4, () =>
      mask(() => {
        const theta = sample('theta', normal(mu, tau))
        observe('y', normal(theta, SIGMA), SCHOOLS)
      }, m)(undefined),
    )
  }
}

describe('probprog/handlers against NumPyro', () => {
  for (const c of loadFixture('probprog/handlers').cases) {
    it(c.name, () => {
      const input = c.inputs[0] === undefined ? {} : values(c.inputs[0])
      const lj = (m: Model, v: Record<string, Tensor>): number => item(logJoint(m, undefined, v).logJoint)
      switch (c.fn) {
        case 'condition':
          expectMatches(lj(condition(schools, { tau: 2.5 }), input), c.output, c.rtol, c.atol, c.name)
          return
        case 'observedFlags': {
          const t = traceOf(seed(condition(schools, { tau: 2.5 }), key(0)), undefined)
          const flags = ['mu', 'tau', 'theta', 'y'].map((n) => {
            const s = t.sites.get(n)
            return s?.type === 'sample' && s.isObserved ? 1 : 0
          })
          expectMatches(tensor(flags), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'substitute':
          expectMatches(lj(substitute(schools, { mu: 1.5 }), input), c.output, c.rtol, c.atol, c.name)
          return
        case 'scale':
          expectMatches(lj(scale(schools, c.params['factor'] as number), input), c.output, c.rtol, c.atol, c.name)
          return
        case 'mask':
          expectMatches(lj(maskedSchools(c.params['mask'] as boolean[]), input), c.output, c.rtol, c.atol, c.name)
          return
        case 'maskAll':
          expectMatches(lj(mask(schools, false), input), c.output, c.rtol, c.atol, c.name)
          return
        case 'do':
          expectMatches(lj(doIntervention(schools, { mu: 4 }), input), c.output, c.rtol, c.atol, c.name)
          return
        case 'blockHide':
        case 'blockExpose': {
          const blocked =
            c.fn === 'blockHide'
              ? block(seed(schools, key(0)), { hide: ['mu', 'theta'] })
              : block(seed(schools, key(0)), { expose: ['tau'] })
          const names = [...traceOf(blocked, undefined).sites.keys()].join(',')
          if (names !== c.params['sites']) throw new Error(`${c.name}: ${names} != ${String(c.params['sites'])}`)
          return
        }
        case 'shape': {
          const t = traceOf(seed(shapesModel, key(0)), undefined)
          const s = t.sites.get(c.params['site'] as string)
          if (s?.type !== 'sample') throw new Error(`${c.name}: missing site`)
          expectMatches(tensor(s.value.shape), c.output, 0, 0, c.name)
          const dims = s.plates
            .map((f) => f.dim)
            .sort((a, b) => a - b)
            .join(',')
          if (dims !== c.params['dims']) throw new Error(`${c.name}: dims ${dims} != ${String(c.params['dims'])}`)
          if (Math.abs(s.scale - (c.params['scale'] as number)) > 1e-12) throw new Error(`${c.name}: scale ${s.scale}`)
          return
        }
        case 'enumerateMarks': {
          const t = traceOf(seed(enumerate(mixed), key(0)), undefined)
          const marked = [...t.sites.values()]
            .filter((s) => s.type === 'sample' && s.infer.enumerate === 'parallel')
            .map((s) => s.name)
            .join(',')
          if (marked !== c.params['sites']) throw new Error(`${c.name}: ${marked}`)
          return
        }
        default:
          throw new Error(`no function ${c.fn}`)
      }
    })
  }
})
