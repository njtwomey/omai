// The parameters of every operator of the registry: declared, parsed at compile, and never mistaken for a numerical failure.
// The table is driven by the registry itself, so a new operator that declares no parameters (or declares them wrongly) fails here.
import { describe, expect, it } from 'vitest'
import { type Json, OmaiError } from '../../base/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as M from '../index.ts'

const operators = M.defaultOperators()
const names = Object.keys(operators)

/** The declared parameters of an operator at its example values, completed by what a dependent spec needs. */
const specOf = (name: string, params: M.Params = {}): M.ParamSpec => {
  const spec = (operators[name] as M.FactorOperator).params as M.ParamSpecOf
  return typeof spec === 'function' ? spec(params) : spec
}

/** Valid parameters of an operator: its example values, and for a registry operator the entry its first example names. */
function validParams(name: string): Record<string, Json> {
  const op = operators[name] as M.FactorOperator
  let params = M.exampleParams(op.params as M.ParamSpecOf)
  if (typeof op.params === 'function') {
    // A registry operator: name one entry, then ask again for that entry's own parameters.
    const choices = (op.params({}) as M.ParamSpec)['likelihood']?.choices
    if (choices !== undefined)
      params = { likelihood: choices[0] as string, ...M.exampleParams(op.params({ likelihood: choices[0] as string })) }
  }
  return params
}

const wrongOf = (rule: M.ParamRule): Json => {
  switch (rule.kind) {
    case 'vector':
      return 'not an array'
    case 'text':
      return 17
    case 'flag':
      return 'yes'
    case 'any':
      return undefined as never
    default:
      return 'not a number'
  }
}

describe('every operator declares its parameters', () => {
  it('all 69 do', () => {
    expect(names.filter((n) => operators[n]?.params === undefined)).toEqual([])
  })

  it.each(names)('%s: its example parameters are valid', (name) => {
    expect(() =>
      M.validateParams(name, (operators[name] as M.FactorOperator).params as M.ParamSpecOf, validParams(name)),
    ).not.toThrow()
  })

  it.each(names)('%s: an unknown parameter is refused, naming the operator and the parameter', (name) => {
    const op = operators[name] as M.FactorOperator
    expect(() =>
      M.validateParams(name, op.params as M.ParamSpecOf, { ...validParams(name), notAParameter: 1 }),
    ).toThrow(new RegExp(`^${name}: unknown parameter 'notAParameter'`))
  })

  it.each(names)('%s: a missing parameter is refused, naming the operator and the parameter', (name) => {
    const op = operators[name] as M.FactorOperator
    const valid = validParams(name)
    const required = Object.entries(specOf(name, valid)).filter(([, r]) => !r.optional && r.kind !== 'any')
    for (const [param] of required) {
      const { [param]: _gone, ...rest } = valid
      expect(() => M.validateParams(name, op.params as M.ParamSpecOf, rest), `${name}.${param}`).toThrow(
        new RegExp(`^${name}: .*'${param}'`),
      )
    }
  })

  it.each(names)('%s: a parameter of the wrong type is refused, naming the operator and the parameter', (name) => {
    const op = operators[name] as M.FactorOperator
    const valid = validParams(name)
    for (const [param, rule] of Object.entries(specOf(name, valid))) {
      if (rule.kind === 'any') continue
      expect(
        () => M.validateParams(name, op.params as M.ParamSpecOf, { ...valid, [param]: wrongOf(rule) }),
        `${name}.${param}`,
      ).toThrow(new RegExp(`^${name}: the parameter '${param}'`))
    }
  })
})

describe('a graph with a wrong parameter fails at compile, as a ParameterError', () => {
  const graph = (operator: string, params?: M.Params): M.FactorGraphSpec => ({
    variables: [
      { name: 'x', family: { kind: 'gaussian' } },
      { name: 'y', family: { kind: 'gaussian' } },
    ],
    factors: [
      {
        operator,
        args: operator === 'times-constant' ? ['y', 'x'] : ['y', 'x'],
        ...(params === undefined ? {} : { params }),
      },
    ],
  })

  it('a gaussian factor given a variance (it takes a precision), which ran to `converged` before', () => {
    const run = (): unknown => M.compileGraph(graph('gaussian', { variance: -2 }), operators)
    expect(run).toThrow(M.ParameterError)
    expect(run).toThrow(/f0 \(gaussian\): unknown parameter 'variance' \(it takes: precision\)/)
  })

  it('a gaussian factor given a negative precision', () => {
    expect(() => M.compileGraph(graph('gaussian', { precision: -2 }), operators)).toThrow(
      /f0 \(gaussian\): the parameter 'precision' must be a number greater than 0, got -2/,
    )
  })

  it('times-constant without its factor, which compiled and ended `failed` with improper events before', () => {
    expect(() => M.compileGraph(graph('times-constant'), operators)).toThrow(
      /f0 \(times-constant\): the parameter 'factor' is required \(a number\)/,
    )
    expect(() => M.compileGraph(graph('times-constant', { a: 2 }), operators)).toThrow(
      /unknown parameter 'a' \(it takes: factor\)/,
    )
  })

  it('a valid graph still compiles', () => {
    expect(() => M.compileGraph(graph('times-constant', { factor: 2 }), operators)).not.toThrow()
    expect(() => M.compileGraph(graph('gaussian', { precision: 2 }), operators)).not.toThrow()
  })

  it('an operator of your own, with no declared parameters, is not checked', () => {
    const { params: _declared, ...undeclared } = operators['times-constant'] as M.FactorOperator
    const mine: M.FactorOperator = { ...undeclared, name: 'mine' }
    expect(() => M.compileGraph(graph('mine', { anything: 1 }), { ...operators, mine })).not.toThrow()
  })

  it('the gate checks the parameters of its branches', () => {
    const gate = (params: M.Params): M.FactorGraphSpec => ({
      variables: [
        { name: 'z', family: { kind: 'discrete', size: 2 } },
        { name: 'x', family: { kind: 'gaussian' } },
      ],
      factors: [
        {
          operator: 'gate',
          args: ['z', 'x'],
          params: {
            arity: 1,
            branches: [
              { operator: 'gaussian', args: [0, { const: 0 }], params },
              { operator: 'gaussian', args: [0, { const: 1 }], params: { precision: 1 } },
            ],
          },
        },
      ],
    })
    expect(() => M.compileGraph(gate({ precision: 1 }), operators)).not.toThrow()
    expect(() => M.compileGraph(gate({ variance: 1 }), operators)).toThrow(
      /branch 0 \(gaussian\): unknown parameter 'variance'/,
    )
  })
})

describe('the engine repairs an improper message and never swallows a parameter error', () => {
  const run = (operator: M.FactorOperator): unknown => {
    const g = M.compileGraph(
      { variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [{ operator: 'probe', args: ['x'] }] },
      { probe: operator },
    )
    const settings: M.EngineSettings = {
      projection: M.momentProjection,
      schedule: 'serial',
      granularity: 'sweep',
      damping: 0,
      improper: 'skip',
      tol: 1e-10,
      maxSweeps: 3,
    }
    return runAlgorithm(M.messagePassingAlgorithm(g, settings), { maxSteps: 20 }).state
  }
  const probe = (error: () => Error): M.FactorOperator => ({
    name: 'probe',
    description: 'throws',
    ports: [{ name: 'x', accepts: ['gaussian'] }],
    support: { ep: 'numeric' },
    ep: () => {
      throw error()
    },
    logEvidence: () => 0,
  })

  it('counts an ImproperMessageError and goes on', () => {
    const state = run(probe(() => new M.ImproperMessageError('no density'))) as M.MpState
    expect(state.improper.events).toBeGreaterThan(0)
  })

  it('lets a ParameterError out of the run', () => {
    expect(() => run(probe(() => new M.ParameterError('bad parameter')))).toThrow(M.ParameterError)
  })

  it('does not take a bare OmaiError of code `value` for an improper message either', () => {
    expect(() => run(probe(() => new OmaiError('value', 'something else')))).toThrow(/something else/)
  })
})
