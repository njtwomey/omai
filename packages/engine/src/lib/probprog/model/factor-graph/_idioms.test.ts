// The pure recognisers of the model compiler: the precision idiom of a scale, the known multiple of a gamma variable, what an
// expression depends on.
import { describe, expect, it } from 'vitest'
import { toFlat } from '../../../tensor/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, ops } from '../index.ts'
import { defaultOperators } from '../operators.ts'
import type { ExprSpec } from '../spec.ts'
import type { Cell } from './_cells.ts'
import {
  dependsOnLatent,
  expArgument,
  type IdiomEnv,
  isRecip,
  latentScope,
  logVariableOf,
  logVariableOfCell,
  mentionsDiscrete,
  multipleOf,
  precisionExpr,
  precisionMultiple,
  randomTransitionRow,
  scaledExp,
  sharedGamma,
  stripClip,
  transitionRow,
  unwrap,
} from './_idioms.ts'
import { evaluateKnown, isKnownExpr, knownConstants } from './_known.ts'

const model = defineModel('idioms', (m) => {
  const tau = m.latent('tau', d.gamma(2, 1))
  const rate = m.latent('rate', d.exponential(1))
  const other = m.latent('other', d.gamma(2, 1))
  const flag = m.latent('flag', d.bernoulli({ probs: 0.3 }))
  m.derived('flagged', ops.mul(flag, 2))
  const k = m.data('k', { dims: [] })
  m.derived('prec', ops.mul(2, tau))
  m.derived('both', ops.add(tau, other))
  m.derived('kk', ops.mul(k, 3))
  m.latent('sigma', d.halfNormal(1))
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
  hasLogVariable: (name) => name === 'sigma',
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

describe('the log-link recognisers', () => {
  it('reads the argument of exp, and nothing of any other expression', () => {
    expect(expArgument(op('exp', ref('tau')))).toEqual(ref('tau'))
    expect(expArgument(op('square', ref('tau')))).toBeUndefined()
    expect(expArgument(ref('tau'))).toBeUndefined()
  })

  it('reads exp(L) and k * exp(L) as the log expression and the known factor, through derived nodes', () => {
    expect(scaledExp(env, op('exp', ref('tau')))).toEqual({ inner: ref('tau'), scale: undefined })
    expect(scaledExp(env, op('mul', num(3), op('exp', ref('tau'))))).toEqual({ inner: ref('tau'), scale: num(3) })
    expect(scaledExp(env, op('mul', op('exp', ref('tau')), ref('k')))).toEqual({ inner: ref('tau'), scale: ref('k') })
    expect(scaledExp(env, op('mul', ref('other'), op('exp', ref('tau'))))).toBeUndefined()
    expect(scaledExp(env, op('mul', ref('other'), op('exp', ref('tau'))), true)).toEqual({
      inner: ref('tau'),
      scale: ref('other'),
    })
    expect(scaledExp(env, op('add', ref('tau'), num(1)))).toBeUndefined()
  })

  it('reads a positive latent carried on its log as exp of that log', () => {
    expect(scaledExp(env, ref('sigma'))).toEqual({ inner: ref('sigma#log'), scale: undefined })
  })

  it('finds the log variable behind a cell that is exactly one value variable', () => {
    const atoms = new Map([['v:tau', { kind: 'var', name: 'tau' } as const]])
    const logScale = new Map([['tau', 'tau#log']])
    const one = (coefficient: number, offset: number): Cell => ({
      kind: 'aff',
      aff: { terms: new Map([['v:tau', coefficient]]), offset },
    })
    expect(logVariableOf(atoms, logScale, 'v:tau')).toBe('tau#log')
    expect(logVariableOf(atoms, logScale, 'v:other')).toBeUndefined()
    expect(logVariableOfCell(one(1, 0), atoms, logScale)).toBe('tau#log')
    expect(logVariableOfCell(one(2, 0), atoms, logScale)).toBeUndefined()
    expect(logVariableOfCell(one(1, 1), atoms, logScale)).toBeUndefined()
  })
})

describe('stripClip, transitionRow and mentionsDiscrete', () => {
  const scope = latentScope(bound.spec)

  it('reads a clip with known bounds as the value clipped', () => {
    expect(stripClip(env, op('clip', ref('tau'), num(0.1), num(10)))).toEqual(ref('tau'))
    expect(stripClip(env, op('clip', ref('tau'), ref('other'), num(10)))).toEqual(
      op('clip', ref('tau'), ref('other'), num(10)),
    )
  })

  it('reads at(A, z) with a known matrix as a transition row, and with a random matrix as a random one', () => {
    expect(transitionRow(env, op('at', ref('kk'), ref('flag')))).toEqual({ matrix: ref('kk'), index: ref('flag') })
    expect(transitionRow(env, op('at', ref('tau'), ref('flag')))).toBeUndefined()
    expect(randomTransitionRow(env, op('at', ref('tau'), ref('flag')))).toEqual({
      matrix: ref('tau'),
      index: ref('flag'),
    })
    expect(randomTransitionRow(env, op('at', ref('kk'), ref('flag')))).toBeUndefined()
  })

  it('says whether an expression reads a Bernoulli or categorical latent, through derived nodes', () => {
    expect(mentionsDiscrete(scope, ref('flagged'))).toBe(true)
    expect(mentionsDiscrete(scope, ref('prec'))).toBe(false)
  })
})
