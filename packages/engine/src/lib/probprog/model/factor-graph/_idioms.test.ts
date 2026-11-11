// The pure recognisers of the model compiler: the precision idiom of a scale, the known multiple of a gamma variable, what an
// expression depends on.
import { describe, expect, it } from 'vitest'
import { toFlat } from '../../../tensor/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, ops } from '../index.ts'
import { defaultOperators } from '../operators.ts'
import type { ExprSpec } from '../spec.ts'
import {
  dependsOnLatent,
  type IdiomEnv,
  isRecip,
  latentScope,
  multipleOf,
  precisionExpr,
  precisionMultiple,
  sharedGamma,
  unwrap,
} from './_idioms.ts'
import { evaluateKnown, isKnownExpr, knownConstants } from './_known.ts'

const model = defineModel('idioms', (m) => {
  const tau = m.latent('tau', d.gamma(2, 1))
  const rate = m.latent('rate', d.exponential(1))
  const other = m.latent('other', d.gamma(2, 1))
  const k = m.data('k', { dims: [] })
  m.derived('prec', ops.mul(2, tau))
  m.derived('both', ops.add(tau, other))
  m.derived('kk', ops.mul(k, 3))
  m.observe('y', d.normal(0, ops.div(1, ops.sqrt(tau))), m.data('y', { dims: ['i'] }))
  void rate
})
const bound = model.bind({ k: 4, y: [1, 2] })
const modelOps = defaultOperators()
const { known } = knownConstants(bound, modelOps)
const nodes = new Map(bound.spec.nodes.filter((n) => n.kind !== 'observed').map((n) => [n.name, n]))
const env: IdiomEnv = {
  nodes,
  isConst: (e) => isKnownExpr(e, known, bound.customs),
  valueOf: (e) => evaluateKnown(e, { modelOps, customs: bound.customs, ref: (name) => known.get(name) as never }),
  isRecipNode: (name) => name === 'sigma2',
}
const ref = (name: string): ExprSpec => ({ kind: 'ref', name })
const num = (value: number): ExprSpec => ({ kind: 'const', value })
const op = (name: string, ...args: ExprSpec[]): ExprSpec => ({ kind: 'op', op: name, args })

describe('the precision idiom of a scale', () => {
  it('reads 1 / sqrt(tau), tau ** -0.5 and sqrt(1 / tau) as tau', () => {
    expect(precisionExpr(env, op('div', num(1), op('sqrt', ref('tau'))))).toEqual(ref('tau'))
    expect(precisionExpr(env, op('pow', ref('tau'), num(-0.5)))).toEqual(ref('tau'))
    expect(precisionExpr(env, op('sqrt', op('div', num(1), ref('tau'))))).toEqual(ref('tau'))
  })

  it('does not read any other shape', () => {
    expect(precisionExpr(env, ref('tau'))).toBeUndefined()
    expect(precisionExpr(env, op('div', num(2), op('sqrt', ref('tau'))))).toBeUndefined()
    expect(precisionExpr(env, op('pow', ref('tau'), num(-1)))).toBeUndefined()
  })

  it('reads the square root of an inverse-gamma variance as the reciprocal of that variance', () => {
    expect(precisionExpr(env, op('sqrt', ref('sigma2')))).toEqual(op('div', num(1), ref('sigma2')))
    expect(isRecip(env, ref('sigma2'))).toBe(true)
    expect(isRecip(env, op('div', ref('sigma2'), num(2)))).toBe(true)
    expect(isRecip(env, op('div', ref('sigma2'), ref('tau')))).toBe(false)
  })
})

describe('the known multiple of a gamma variable', () => {
  it('finds c in c * tau, tau / c and their nesting, and 1 for the node itself', () => {
    expect(multipleOf(env, ref('tau'), 'tau')).toBe(1)
    expect(multipleOf(env, op('mul', num(3), ref('tau')), 'tau')).toBe(3)
    expect(multipleOf(env, op('div', ref('tau'), num(4)), 'tau')).toBe(0.25)
    expect(multipleOf(env, op('mul', ref('kk'), op('div', ref('tau'), num(2))), 'tau')).toBe(6)
  })

  it('finds nothing for a random multiple or another variable', () => {
    expect(multipleOf(env, op('mul', ref('other'), ref('tau')), 'tau')).toBeUndefined()
    expect(multipleOf(env, op('mul', num(3), ref('other')), 'tau')).toBeUndefined()
  })

  it('is the precision a scale stands for: 1 / sqrt(c tau) is c, for an inverse gamma sqrt(c sigma2) is 1 / c', () => {
    expect(precisionMultiple(env, op('div', num(1), op('sqrt', op('mul', num(4), ref('tau')))), 'tau', false)).toBe(4)
    expect(precisionMultiple(env, op('sqrt', op('mul', num(4), ref('sigma2'))), 'sigma2', true)).toBe(0.25)
    expect(precisionMultiple(env, ref('tau'), 'tau', false)).toBeUndefined()
  })
})

describe('unwrap, dependsOnLatent and sharedGamma', () => {
  const scope = latentScope(bound.spec)

  it('follows a reference through derived nodes to the expression behind it', () => {
    expect(unwrap(nodes, ref('prec'))).toEqual(op('mul', num(2), ref('tau')))
    expect(unwrap(nodes, ref('tau'))).toEqual(ref('tau'))
  })

  it('says whether an expression reads a latent, through derived nodes', () => {
    expect(dependsOnLatent(scope, ref('prec'))).toBe(true)
    expect(dependsOnLatent(scope, ref('kk'))).toBe(false)
    expect(dependsOnLatent(scope, op('add', ref('k'), num(1)))).toBe(false)
  })

  it('names the one scalar gamma latent an expression reads, and nothing for two or for a data value', () => {
    expect(sharedGamma(scope, ref('prec'))).toBe('tau')
    expect(sharedGamma(scope, ref('both'))).toBeUndefined()
    expect(sharedGamma(scope, ref('kk'))).toBeUndefined()
  })

  it('reads a known value by the registry', () => {
    expect(Array.from(toFlat(env.valueOf(ref('kk')).value))).toEqual([12])
  })
})
