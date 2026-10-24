/**
 * Executing a transformed example: its JavaScript becomes the body of an async function whose parameters are the
 * runner's hooks, so the code sees nothing else of the page. Asynchronous waits are bounded by a time limit; a
 * synchronous loop is not (run examples in a worker to bound those).
 */
import { HOOKS } from './transform.ts'

/** The functions the instrumented code calls, by the names in {@link HOOKS}. */
export interface Hooks {
  readonly require: (specifier: string) => unknown
  readonly __at: (index: number, value: unknown, expected?: () => unknown) => unknown
  readonly __throws: (index: number, run: () => Promise<unknown>) => Promise<void>
  readonly console: Readonly<Record<'log' | 'info' | 'warn' | 'error' | 'debug', (...args: unknown[]) => void>>
}

// The Function constructor puts two lines (the header and the opening brace) before the body.
const BODY_OFFSET = 2

type AsyncBody = (...args: unknown[]) => Promise<unknown>
const AsyncFunction = (async () => {}).constructor as new (...args: string[]) => AsyncBody

/** An error raised by a time limit. */
export class TimeoutError extends Error {
  constructor(ms: number) {
    super(`timed out after ${ms} ms`)
    this.name = 'TimeoutError'
  }
}

/**
 * The 0-based example line an error was thrown from, read from its stack (`<anonymous>:line:column`).
 *
 * @param error - A caught value.
 * @returns The line, or `undefined` when the stack does not say.
 */
export function errorLine(error: unknown): number | undefined {
  const stack = error instanceof Error ? (error.stack ?? '') : ''
  const m = /<anonymous>:(\d+):\d+/.exec(stack) ?? /eval[^\n]*?:(\d+):\d+\)?\s*$/m.exec(stack)
  if (m === null) return undefined
  const line = Number(m[1]) - 1 - BODY_OFFSET
  return line >= 0 ? line : undefined
}

/**
 * Run a transformed example's JavaScript with the given hooks.
 *
 * @param js - The body, from `transformExample`.
 * @param hooks - The functions it calls.
 * @param timeoutMs - The longest to wait for it to settle (`Infinity` for no limit).
 * @returns A promise that settles when the example does.
 * @throws Whatever the example throws, or a {@link TimeoutError}.
 */
export async function execute(js: string, hooks: Hooks, timeoutMs: number = Number.POSITIVE_INFINITY): Promise<void> {
  const body = new AsyncFunction(...HOOKS, js)
  const run = body(...HOOKS.map((name) => hooks[name]))
  if (!Number.isFinite(timeoutMs)) {
    await run
    return
  }
  let timer: ReturnType<typeof setTimeout> | undefined
  const limit = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new TimeoutError(timeoutMs)), timeoutMs)
  })
  try {
    await Promise.race([run, limit])
  } finally {
    clearTimeout(timer)
  }
}
