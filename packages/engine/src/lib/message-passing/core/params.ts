/**
 * The declared parameters of a factor operator and their validation: each operator lists the JSON parameters it reads
 * (`param.number()`, `param.positive()`, `param.vector()`, ...) with whether each is optional, and `compileGraph` parses a
 * factor's parameters against that list before anything else runs, so an unknown, missing or ill-typed parameter ends at
 * compile time with a `ParameterError` that names the operator and the parameter. A spec can be a function of the
 * parameters for an operator whose parameters depend on one of them (a registry of likelihoods chosen by name). The
 * structural checks that need the families of the ports stay in the operator's `check`.
 */

import type { Json } from '../../base/index.ts'
import { ParameterError } from './errors.ts'

/** What one declared parameter must be. */
export interface ParamRule {
  /** `number` (finite unless `infinite`), `integer`, `vector` (an array of finite numbers), `text`, `flag` (a boolean), `any` (opaque; the operator's `check` reads it). */
  readonly kind: 'number' | 'integer' | 'vector' | 'text' | 'flag' | 'any'
  /** Whether the parameter may be left out (the operator then reads its own default). */
  readonly optional: boolean
  /** A number's lower bound, inclusive. */
  readonly min?: number
  /** A number's lower bound, exclusive. */
  readonly above?: number
  /** A number's upper bound, inclusive. */
  readonly max?: number
  /** Whether a number may be infinite (a bound of a support). */
  readonly infinite?: boolean
  /** The strings a `text` parameter may be. */
  readonly choices?: readonly string[]
  /** A valid value, used to build example parameters (the tests, the docs). */
  readonly example: Json
}

/** The parameters an operator reads, by name. */
export type ParamSpec = Readonly<Record<string, ParamRule>>

/** A spec, or a function of the parameters when the declared names depend on one of them. */
export type ParamSpecOf = ParamSpec | ((params: Readonly<Record<string, Json>>) => ParamSpec)

/** Options common to every rule. */
export interface ParamOptions {
  readonly optional?: boolean
  /** A valid value for the examples (default: the kind's own). */
  readonly example?: Json
}

/** Options of a number rule. */
export interface NumberOptions extends ParamOptions {
  readonly min?: number
  readonly above?: number
  readonly max?: number
  readonly infinite?: boolean
}

/** Options of a text rule. */
export interface TextOptions extends ParamOptions {
  readonly choices?: readonly string[]
}

const make = (kind: ParamRule['kind'], o: NumberOptions & TextOptions, example: Json): ParamRule => ({
  kind,
  optional: o.optional === true,
  ...(o.min === undefined ? {} : { min: o.min }),
  ...(o.above === undefined ? {} : { above: o.above }),
  ...(o.max === undefined ? {} : { max: o.max }),
  ...(o.infinite === true ? { infinite: true } : {}),
  ...(o.choices === undefined ? {} : { choices: o.choices }),
  example: o.example ?? example,
})

/** A number near `start` that satisfies the bounds of the options. */
function inRange(o: NumberOptions, start: number): number {
  if (o.above !== undefined && start <= o.above) return o.above + 1
  if (o.min !== undefined && start < o.min) return o.min
  if (o.max !== undefined && start > o.max) return o.max
  return start
}

/**
 * The rule builders: one call per kind of parameter.
 *
 * @example
 * ```ts
 * import { param, validateParams } from 'omai-engine/message-passing'
 * const spec = { precision: param.positive(), offset: param.number({ optional: true }) }
 * validateParams('demo', spec, { precision: 2 }) // => undefined
 * ```
 */
export const param: {
  readonly number: (o?: NumberOptions) => ParamRule
  readonly positive: (o?: NumberOptions) => ParamRule
  readonly nonNegative: (o?: NumberOptions) => ParamRule
  readonly integer: (o?: NumberOptions) => ParamRule
  readonly vector: (o?: ParamOptions) => ParamRule
  readonly text: (o?: TextOptions) => ParamRule
  readonly flag: (o?: ParamOptions) => ParamRule
  readonly any: (o?: ParamOptions) => ParamRule
} = {
  number: (o = {}) => make('number', o, inRange(o, 1)),
  positive: (o = {}) => make('number', { ...o, above: o.above ?? 0 }, inRange({ above: o.above ?? 0, ...o }, 1)),
  nonNegative: (o = {}) => make('number', { ...o, min: o.min ?? 0 }, inRange({ min: o.min ?? 0, ...o }, 1)),
  integer: (o = {}) => make('integer', o, inRange(o, 2)),
  vector: (o = {}) => make('vector', o, [1, 2]),
  text: (o = {}) => make('text', o, o.choices?.[0] ?? 'a'),
  flag: (o = {}) => make('flag', o, true),
  any: (o = {}) => make('any', o, null),
}

function describe(r: ParamRule): string {
  switch (r.kind) {
    case 'number': {
      if (r.above !== undefined) return `a number greater than ${r.above}`
      if (r.min !== undefined) return `a number of at least ${r.min}`
      if (r.max !== undefined) return `a number of at most ${r.max}`
      return 'a number'
    }
    case 'integer':
      return r.min === undefined ? 'an integer' : `an integer of at least ${r.min}`
    case 'vector':
      return 'an array of numbers'
    case 'text':
      return r.choices === undefined ? 'a string' : `one of ${r.choices.map((c) => `'${c}'`).join(', ')}`
    case 'flag':
      return 'a boolean'
    case 'any':
      return 'a value'
  }
}

function show(v: unknown): string {
  const text = typeof v === 'number' ? String(v) : (JSON.stringify(v) ?? String(v))
  return text.length > 40 ? `${text.slice(0, 37)}...` : text
}

/** Whether a value is what a rule wants. */
function accepts(r: ParamRule, v: unknown): boolean {
  switch (r.kind) {
    case 'any':
      return true
    case 'flag':
      return typeof v === 'boolean'
    case 'text':
      return typeof v === 'string' && (r.choices === undefined || r.choices.includes(v))
    case 'vector':
      return Array.isArray(v) && v.every((x) => typeof x === 'number' && Number.isFinite(x))
    case 'integer':
    case 'number': {
      if (typeof v !== 'number' || Number.isNaN(v)) return false
      if (r.infinite !== true && !Number.isFinite(v)) return false
      if (r.kind === 'integer' && !Number.isInteger(v)) return false
      if (r.above !== undefined && !(v > r.above)) return false
      if (r.min !== undefined && !(v >= r.min)) return false
      return r.max === undefined || v <= r.max
    }
  }
}

/**
 * Parse the parameters of a factor against an operator's declared ones: every required parameter present, every present
 * parameter of the right kind and range, and no parameter the operator does not read.
 *
 * @param who - What to name in the message: the operator, or the factor and its operator.
 * @param spec - The declared parameters.
 * @param params - The factor's parameters.
 * @throws ParameterError naming the operator and the parameter.
 * @example
 * ```ts
 * import { param, validateParams } from 'omai-engine/message-passing'
 * const check = (p: Record<string, number>) => {
 *   try {
 *     validateParams('gaussian', { precision: param.positive() }, p)
 *     return 'ok'
 *   } catch (e) {
 *     return (e as Error).message
 *   }
 * }
 * check({ variance: 2 }) // => "gaussian: unknown parameter 'variance' (it takes: precision)"
 * ```
 */
export function validateParams(who: string, spec: ParamSpecOf, params: Readonly<Record<string, Json>>): void {
  const rules = typeof spec === 'function' ? spec(params) : spec
  const names = Object.keys(rules)
  for (const name of Object.keys(params)) {
    if (rules[name] === undefined) {
      const takes = names.length === 0 ? 'it takes no parameters' : `it takes: ${names.join(', ')}`
      throw new ParameterError(`${who}: unknown parameter '${name}' (${takes})`)
    }
  }
  for (const [name, r] of Object.entries(rules)) {
    const v = params[name]
    if (v === undefined) {
      if (!r.optional) throw new ParameterError(`${who}: the parameter '${name}' is required (${describe(r)})`)
    } else if (!accepts(r, v)) {
      throw new ParameterError(`${who}: the parameter '${name}' must be ${describe(r)}, got ${show(v)}`)
    }
  }
}

/**
 * The parameters of an operator built on another (a regression row lifts a link: its own parameters beside the link's): the
 * declared ones merged with the other operator's, or nothing when the other operator declares none (then the lift is not
 * checked either).
 *
 * @param own - The lift's own parameters.
 * @param inner - The inner operator's `params`.
 * @returns The merged spec, or `undefined` when the inner operator declares no parameters.
 * @example
 * ```ts
 * import { mergeParams, param } from 'omai-engine/message-passing'
 * const merged = mergeParams({ a: param.vector() }, { noise: param.positive({ optional: true }) })
 * Object.keys(merged ?? {}) // => ['a', 'noise']
 * ```
 */
export function mergeParams(own: ParamSpec, inner: ParamSpecOf | undefined): ParamSpecOf | undefined {
  if (inner === undefined) return undefined
  if (typeof inner === 'function') return (params) => ({ ...own, ...inner(params) })
  return { ...own, ...inner }
}

/**
 * A spec from a list of names, for a registry whose entries list the parameters they read: each a number, except the names
 * given as vectors, and the names given as optional may be left out.
 *
 * @param names - The parameter names.
 * @param options - Which of them are optional, and which are vectors.
 * @returns The spec.
 * @example
 * ```ts
 * import { namedParams } from 'omai-engine/message-passing'
 * const spec = namedParams(['shape', 'censored'], { optional: ['censored'] })
 * [spec['shape']?.optional, spec['censored']?.optional] // => [false, true]
 * ```
 */
export function namedParams(
  names: readonly string[],
  options: { readonly optional?: readonly string[]; readonly vectors?: readonly string[] } = {},
): ParamSpec {
  const out: Record<string, ParamRule> = {}
  for (const name of names) {
    const optional = options.optional?.includes(name) === true
    out[name] = options.vectors?.includes(name) === true ? param.vector({ optional }) : param.number({ optional })
  }
  return out
}

/**
 * Valid parameters built from a spec: every required parameter at the rule's example value (and the optional ones too when
 * `withOptional`). For tests and documentation.
 *
 * @param spec - The declared parameters (a function of the parameters is called with `{}`).
 * @param withOptional - Whether to include the optional parameters.
 * @returns The parameters.
 * @example
 * ```ts
 * import { exampleParams, param } from 'omai-engine/message-passing'
 * exampleParams({ precision: param.positive(), offset: param.number({ optional: true }) }) // => { precision: 1 }
 * ```
 */
export function exampleParams(spec: ParamSpecOf, withOptional = false): Record<string, Json> {
  const rules = typeof spec === 'function' ? spec({}) : spec
  const out: Record<string, Json> = {}
  for (const [name, r] of Object.entries(rules)) if (withOptional || !r.optional) out[name] = r.example
  return out
}
