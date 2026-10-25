/**
 * The `Context`: everything a computation needs besides its data, namely a seed with named key derivation, the tracer
 * and trace policy, the backend and the default float dtype. `createContext` makes one, and `derive` makes a child
 * whose keys are namespaced so independent parts of a program never share randomness.
 */

import { type Key, keyFromPath } from '../random/index.ts'
import { type Backend, cpu, type DType } from '../tensor/index.ts'
import { nullTracer, policies, type TracePolicy, type Tracer } from '../trace/index.ts'

/**
 * Everything a computation needs besides its data: the seed (with named key derivation), where traces go and what is
 * recorded, the backend and the default float dtype. Passing one context gives a whole program reproducible
 * randomness and consistent tracing.
 *
 * @concept context
 */
export interface Context {
  readonly kind: 'context'
  readonly seed: number
  readonly path: readonly (string | number)[]
  readonly tracer: Tracer
  readonly policy: TracePolicy
  readonly backend: Backend
  readonly dtype: Extract<DType, 'float64' | 'float32'>
  /** A key for a named purpose: the same names always give the same key. */
  key(...names: readonly (string | number)[]): Key
}

/** Options for {@link createContext}. */
export interface CreateContextOptions {
  readonly seed?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
  readonly backend?: Backend
  readonly dtype?: Extract<DType, 'float64' | 'float32'>
}

function make(
  seed: number,
  path: readonly (string | number)[],
  options: Required<Omit<CreateContextOptions, 'seed'>>,
): Context {
  return {
    kind: 'context',
    seed,
    path,
    tracer: options.tracer,
    policy: options.policy,
    backend: options.backend,
    dtype: options.dtype,
    key: (...names) => keyFromPath(seed, ...path, ...names),
  }
}

/**
 * Create a context.
 *
 * @param options - Seed (default 0), tracer (default none), policy (default minimal), backend (default CPU), dtype.
 * @returns The context.
 * @concept context
 * @example
 * ```ts
 * import { createContext } from 'whitebox-engine/context'
 * const ctx = createContext({ seed: 42 })
 * ctx.key('init').words[0] === createContext({ seed: 42 }).key('init').words[0] // => true
 * ```
 */
export function createContext(options: CreateContextOptions = {}): Context {
  return make(options.seed ?? 0, [], {
    tracer: options.tracer ?? nullTracer,
    policy: options.policy ?? policies.minimal,
    backend: options.backend ?? cpu,
    dtype: options.dtype ?? 'float64',
  })
}

/**
 * A child context whose keys are namespaced under `name` (so independent parts of a program never share randomness),
 * optionally with other settings changed.
 *
 * @param ctx - The parent.
 * @param name - The namespace.
 * @param overrides - Settings to change.
 * @returns The child context.
 * @example
 * ```ts
 * import { createContext, derive } from 'whitebox-engine/context'
 * const ctx = createContext({ seed: 1 })
 * derive(ctx, 'data').key('x').words[0] === ctx.key('data', 'x').words[0] // => true
 * ```
 */
export function derive(
  ctx: Context,
  name: string | number,
  overrides: Omit<CreateContextOptions, 'seed'> = {},
): Context {
  return make(ctx.seed, [...ctx.path, name], {
    tracer: overrides.tracer ?? ctx.tracer,
    policy: overrides.policy ?? ctx.policy,
    backend: overrides.backend ?? ctx.backend,
    dtype: overrides.dtype ?? ctx.dtype,
  })
}

/**
 * Whether a value is a context.
 *
 * @param value - Any value.
 * @returns `true` for contexts.
 * @example
 * ```ts
 * import { createContext, isContext } from 'whitebox-engine/context'
 * isContext(createContext()) // => true
 * ```
 */
export function isContext(value: unknown): value is Context {
  return typeof value === 'object' && value !== null && (value as { kind?: unknown }).kind === 'context'
}
