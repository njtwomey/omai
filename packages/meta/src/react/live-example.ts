/**
 * The state of a live example, for any React page that shows one: the code being edited, the latest result (with the
 * code it belongs to), running on demand and once on first view. Drawing it is the page's business (the docs site
 * draws the code with its highlighter and the outcomes beside each line).
 */
import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'
import type { ModuleLoader } from '../modules/index.ts'
import { runExample } from '../runner/run.ts'
import type { Example, ExampleResult } from '../schema/examples.ts'

/** A result with the code it was run on, so a page can tell whether it still describes what is in the editor. */
export interface RunOutcome {
  readonly code: string
  readonly result: ExampleResult
}

/** What {@link useLiveExample} gives a page. */
export interface LiveExampleState {
  readonly code: string
  readonly setCode: (code: string) => void
  /** Whether the code differs from the example's. */
  readonly edited: boolean
  /** The latest result, or `null` before the first run. */
  readonly outcome: RunOutcome | null
  readonly running: boolean
  /** Run the code in the editor (or the given code). Later runs win over earlier ones still in flight. */
  readonly run: (code?: string) => Promise<void>
  /** Put the example's own code back and run it. */
  readonly reset: () => void
  /** Attach to the element whose first appearance on screen should run the example. */
  readonly ref: RefObject<HTMLDivElement | null>
}

/** Options of {@link useLiveExample}. */
export interface LiveExampleOptions {
  /** The module map for the example's imports (normally `moduleLoaders` from `omai-meta/modules`). */
  readonly loaders: Readonly<Record<string, ModuleLoader>>
  /** The longest an example may wait on promises (default 10 s). */
  readonly timeoutMs?: number
  /** Run once when the element under `ref` first scrolls into view (default true). */
  readonly runOnView?: boolean
}

/**
 * Hold a live example: its editable code and the result of running it with omai-meta's runner.
 *
 * @param example - The example.
 * @param options - The module map, the time limit, and whether to run on first view.
 * @returns The state and its actions.
 */
export function useLiveExample(example: Example, options: LiveExampleOptions): LiveExampleState {
  const { loaders, timeoutMs = 10_000, runOnView = true } = options
  const [code, setCode] = useState(example.code)
  const [outcome, setOutcome] = useState<RunOutcome | null>(null)
  const [running, setRunning] = useState(false)
  const ref = useRef<HTMLDivElement | null>(null)
  const latest = useRef(0)
  const codeNow = useRef(code)
  codeNow.current = code
  const run = useCallback(
    async (text?: string): Promise<void> => {
      const source = text ?? codeNow.current
      const ticket = ++latest.current
      setRunning(true)
      // Let the page repaint (the button says it is running) before a long synchronous example takes the thread.
      await new Promise((resolve) => setTimeout(resolve, 0))
      const result = await runExample(source, { loaders, timeoutMs })
      if (ticket !== latest.current) return
      setOutcome({ code: source, result })
      setRunning(false)
    },
    [loaders, timeoutMs],
  )
  const reset = useCallback((): void => {
    setCode(example.code)
    void run(example.code)
  }, [example.code, run])
  useEffect(() => {
    const el = ref.current
    if (!runOnView || el === null) return
    if (typeof IntersectionObserver === 'undefined') {
      void run(example.code)
      return
    }
    const io = new IntersectionObserver((entries) => {
      if (!entries.some((e) => e.isIntersecting)) return
      io.disconnect()
      void run(example.code)
    })
    io.observe(el)
    return () => io.disconnect()
  }, [runOnView, run, example.code])
  return { code, setCode, edited: code !== example.code, outcome, running, run, reset, ref }
}
