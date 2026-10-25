/**
 * Running examples: parse and transform the code, load the modules it imports through the module map, execute it, and
 * report each `// =>` expectation with the value it gave. The same code runs in Node (the example checker) and in the
 * browser (live examples).
 */
import type { ModuleLoader } from '../modules/index.ts'
import type { Example, ExampleError, ExampleResult, ExpectationResult } from '../schema/examples.ts'
import { compareValues } from './compare.ts'
import { errorLine, execute, type Hooks } from './execute.ts'
import { formatValue } from './format.ts'
import { type TransformedExample, transformExample } from './transform.ts'

/** Where examples find their imports, and how long they may take. */
export interface RunOptions {
  /** Lazy loaders by specifier, normally `moduleLoaders` from `mdml-meta/modules`. */
  readonly loaders?: Readonly<Record<string, ModuleLoader>>
  /** Modules already in hand, by specifier; they take precedence over the loaders. */
  readonly modules?: Readonly<Record<string, unknown>>
  /** The longest an example may wait on promises, in milliseconds (default 30 000). */
  readonly timeoutMs?: number
  /** Whether a value is a figure, for `// => figure` (default: `isScene` from `mdml-engine/scene`). */
  readonly isFigure?: (value: unknown) => boolean
  /** Whether a value is a trace, for `// => trace` (default: `isRun` from `mdml-engine/trace`). */
  readonly isTrace?: (value: unknown) => boolean
}

/** One example with its result. */
export interface ExampleRun {
  readonly example: Example
  readonly result: ExampleResult
}

/** What running a list of examples reports. */
export interface ExamplesReport {
  readonly total: number
  readonly passed: number
  readonly failed: number
  readonly runs: readonly ExampleRun[]
  readonly ms: number
}

function message(error: unknown): string {
  if (error instanceof Error) {
    const code = (error as { code?: unknown }).code
    return `${error.name}${typeof code === 'string' ? ` (${code})` : ''}: ${error.message}`
  }
  return String(error)
}

function now(): number {
  return performance.now() // allow: timing an example run, not an algorithm
}

async function loadModules(specifiers: readonly string[], options: RunOptions): Promise<Map<string, unknown>> {
  const out = new Map<string, unknown>()
  for (const spec of specifiers) {
    const given = options.modules?.[spec]
    if (given !== undefined) out.set(spec, given)
    else {
      const load = options.loaders?.[spec]
      if (load === undefined) throw new Error(`no module '${spec}' to import`)
      out.set(spec, await load())
    }
  }
  return out
}

async function recogniser(
  given: ((value: unknown) => boolean) | undefined,
  spec: string,
  name: string,
  options: RunOptions,
): Promise<(value: unknown) => boolean> {
  if (given !== undefined) return given
  try {
    const mod = (await loadModules([spec], options)).get(spec) as Record<string, unknown>
    const f = mod[name]
    if (typeof f === 'function') return f as (value: unknown) => boolean
  } catch {
    // Without the engine, fall back to "an object".
  }
  return (value) => typeof value === 'object' && value !== null
}

/**
 * Run one example and report every expectation: `pass` or `fail` with the value it gave, `error` when evaluating the
 * expected value threw, `skipped` when the example stopped before reaching it.
 *
 * @param example - The example, or its code.
 * @param options - The module map and limits.
 * @returns The result; it never rejects.
 */
export async function runExample(example: Example | string, options: RunOptions = {}): Promise<ExampleResult> {
  const code = typeof example === 'string' ? example : example.code
  const t0 = now()
  const logs: string[] = []
  let transformed: TransformedExample
  try {
    transformed = transformExample(code)
  } catch (e) {
    const at = /\((\d+):\d+\)/.exec(e instanceof Error ? e.message : '')
    const error: ExampleError = { message: message(e), ...(at === null ? {} : { line: Number(at[1]) - 1 }) }
    return { ok: false, expectations: [], logs, error, ms: now() - t0 }
  }
  const { parsed, js } = transformed
  const results = new Map<number, ExpectationResult>()
  const record = (k: number, status: ExpectationResult['status'], actual: string, why: string): void => {
    const e = parsed.expectations[k]
    const had = results.get(k)
    if (e === undefined || (had !== undefined && had.status !== 'pass')) return
    const { prefix: _prefix, markerLine: _marker, ...expectation } = e
    results.set(k, { ...expectation, status, actual, message: why })
  }
  const kinds = new Set(parsed.expectations.map((e) => e.kind))
  const isFigure = kinds.has('figure')
    ? await recogniser(options.isFigure, 'mdml-engine/scene', 'isScene', options)
    : () => false
  const isTrace = kinds.has('trace')
    ? await recogniser(options.isTrace, 'mdml-engine/trace', 'isRun', options)
    : () => false
  let error: ExampleError | undefined
  try {
    const modules = await loadModules(parsed.imports, options)
    const hooks: Hooks = {
      require: (spec) => {
        if (!modules.has(spec)) throw new Error(`no module '${spec}' to import`)
        return modules.get(spec)
      },
      __at: (k, value, expected) => {
        const e = parsed.expectations[k]
        const shown = formatValue(value)
        if (e === undefined) return value
        if (e.kind === 'figure' || e.kind === 'trace') {
          const ok = (e.kind === 'figure' ? isFigure : isTrace)(value)
          record(k, ok ? 'pass' : 'fail', shown, ok ? '' : `expected a ${e.kind}`)
          return value
        }
        let want: unknown
        try {
          want = expected?.()
        } catch (err) {
          record(k, 'error', shown, `the expected value threw: ${message(err)}`)
          return value
        }
        const c = compareValues(value, want, e.expected)
        record(k, c.pass ? 'pass' : 'fail', shown, c.message)
        return value
      },
      __throws: async (k, run) => {
        const e = parsed.expectations[k]
        try {
          const value = await run()
          record(k, 'fail', formatValue(value), 'expected it to throw')
        } catch (err) {
          const code = (err as { code?: unknown }).code
          const name = err instanceof Error ? err.name : ''
          const want = e?.expected ?? ''
          const ok = want === '' || code === want || name === want
          record(k, ok ? 'pass' : 'fail', message(err), ok ? '' : `expected ${want}, got ${code ?? name}`)
        }
      },
      console: Object.fromEntries(
        (['log', 'info', 'warn', 'error', 'debug'] as const).map((level) => [
          level,
          (...args: unknown[]) => {
            logs.push(args.map((a) => (typeof a === 'string' ? a : formatValue(a))).join(' '))
          },
        ]),
      ) as Hooks['console'],
    }
    await execute(js, hooks, options.timeoutMs ?? 30_000)
  } catch (e) {
    const line = errorLine(e)
    error = { message: message(e), ...(line === undefined ? {} : { line }) }
  }
  const expectations = parsed.expectations.map((e, k): ExpectationResult => {
    const r = results.get(k)
    if (r !== undefined) return r
    const { prefix: _prefix, markerLine: _marker, ...expectation } = e
    return { ...expectation, status: 'skipped', actual: '', message: 'not reached' }
  })
  const ok = error === undefined && expectations.every((e) => e.status === 'pass')
  return { ok, expectations, logs, ...(error === undefined ? {} : { error }), ms: now() - t0 }
}

/**
 * Run examples one after another and count the outcomes.
 *
 * @param examples - The examples.
 * @param options - The module map and limits.
 * @returns Each example's result and the totals.
 */
export async function runExamples(examples: readonly Example[], options: RunOptions = {}): Promise<ExamplesReport> {
  const t0 = now()
  const runs: ExampleRun[] = []
  for (const example of examples) runs.push({ example, result: await runExample(example, options) })
  const passed = runs.filter((r) => r.result.ok).length
  return { total: runs.length, passed, failed: runs.length - passed, runs, ms: now() - t0 }
}
