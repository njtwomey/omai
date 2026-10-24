import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { DISCRETE, MODELS } from '../../testing/probprog.ts'
import { concat, item, reshape, type Tensor, tensor, type Value } from '../tensor/index.ts'
import {
  condition,
  constrainSites,
  enumerate,
  enumerationPosterior,
  logJoint,
  type Model,
  potentialEnergy,
  substitute,
  unconstrainSites,
} from './index.ts'

type Values = Record<string, Tensor>

const SUBSAMPLE = [1, 4, 7]

function program(name: string): Model {
  const m = MODELS[name] ?? DISCRETE[name]
  if (m === undefined) throw new Error(`no model ${name}`)
  return name === 'subsampled' ? substitute(m, { data: SUBSAMPLE }) : m
}

function values(x: unknown): Values {
  return Object.fromEntries(Object.entries(x as Record<string, unknown>).map(([k, v]) => [k, decode(v) as Tensor]))
}

function matchRecord(
  got: Readonly<Record<string, Value>>,
  want: unknown,
  rtol: number,
  atol: number,
  label: string,
): void {
  for (const [k, v] of Object.entries(want as Record<string, unknown>)) {
    const g = got[k]
    if (g === undefined) throw new Error(`${label}: missing ${k}`)
    expectMatches(reshape(g, (decode(v) as Tensor).shape), v, rtol, atol, `${label}.${k}`)
  }
}

describe('probprog/density against NumPyro', () => {
  for (const c of loadFixture('probprog/density').cases) {
    it(c.name, () => {
      const name = c.params['model'] as string
      const model = program(name)
      const input = values(c.inputs[0])
      switch (c.fn) {
        case 'logDensity':
          expectMatches(item(logJoint(model, undefined, input).logJoint), c.output, c.rtol, c.atol, c.name)
          return
        case 'unconstrain':
          matchRecord(unconstrainSites(model, undefined, input), c.output, c.rtol, c.atol, c.name)
          return
        case 'potential': {
          const pe = potentialEnergy(model, undefined)
          const u = concat(pe.layout.map((l) => reshape(input[l.name] as Tensor, [l.size])))
          expectMatches(pe.potential(Array.from((u as Tensor).data)), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'constrain':
          matchRecord(constrainSites(model, undefined, input), c.output, c.rtol, c.atol, c.name)
          return
        case 'enumerated':
          expectMatches(item(logJoint(enumerate(model), undefined, input).logJoint), c.output, c.rtol, c.atol, c.name)
          return
        case 'marginals': {
          const post = enumerationPosterior(condition(model, input), undefined)
          expectMatches(post.mean(c.params['site'] as string), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'chainMarginals': {
          const post = enumerationPosterior(model, undefined)
          expectMatches(tensor(post.sites.map((s) => item(post.mean(s)))), c.output, c.rtol, c.atol, c.name)
          return
        }
        default:
          throw new Error(`no function ${c.fn}`)
      }
    })
  }
  it('covers every fixture function', () => {
    expect(new Set(loadFixture('probprog/density').cases.map((c) => c.fn)).size).toBe(7)
  })
})
