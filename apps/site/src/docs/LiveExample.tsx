/**
 * An executable example, as aifn-engine's docs cells: the code in an editable box with a Run button, and under it the
 * outcome of every `// =>` line (the expected value beside the value the code gave, mismatches marked), what the code
 * printed, and any error with its example line. It runs once on first view, through whitebox-meta's runner with the
 * module map for its imports, on the page's thread (asynchronous waits are bounded; a synchronous loop is not).
 */
import { type ReactNode, type RefObject, useCallback, useEffect, useRef, useState } from 'react'
import { moduleLoaders } from 'whitebox-meta/modules'
import { runExample } from 'whitebox-meta/runner'
import type { Example, ExampleResult, ExpectationResult } from 'whitebox-meta/schema'

const STATUS: Record<ExpectationResult['status'], { readonly mark: string; readonly className: string }> = {
  pass: { mark: '✓', className: 'text-emerald-600 dark:text-emerald-400' },
  fail: { mark: '✗', className: 'text-red-600 dark:text-red-400' },
  error: { mark: '!', className: 'text-red-600 dark:text-red-400' },
  skipped: { mark: '–', className: 'text-muted' },
}

function oneLine(text: string, max = 80): string {
  const s = text.replace(/\s+/g, ' ').trim()
  return s.length > max ? `${s.slice(0, max - 1)}…` : s
}

function Outcome(props: { readonly e: ExpectationResult }): ReactNode {
  const { e } = props
  const s = STATUS[e.status]
  const want = e.kind === 'value' ? e.expected : e.kind === 'throws' ? `throws ${e.expected}`.trim() : e.kind
  return (
    <li className="grid grid-cols-[1.25rem_3rem_1fr] gap-x-2 py-0.5">
      <span className={`font-bold ${s.className}`} title={e.status}>
        {s.mark}
      </span>
      <span className="text-muted">line {e.line + 1}</span>
      <span className="min-w-0">
        <span className="text-fg-secondary">{oneLine(e.expression)}</span>
        {e.status === 'pass' ? (
          <span className="text-muted"> → {oneLine(e.actual, 200)}</span>
        ) : e.status === 'skipped' ? (
          <span className="text-muted"> (not reached)</span>
        ) : (
          <span className="block pl-3">
            <span className="text-muted">expected </span>
            <span>{oneLine(want, 200)}</span>
            <span className="text-muted">, got </span>
            <span className={s.className}>{oneLine(e.actual, 200) || '(nothing)'}</span>
            {e.message === '' ? null : <span className="block text-muted">{e.message}</span>}
          </span>
        )}
      </span>
    </li>
  )
}

/** Run the example once it scrolls into view (pages hold many; running them all at load would stall the page). */
function useFirstView(ref: RefObject<HTMLElement | null>, onView: () => void): void {
  useEffect(() => {
    const el = ref.current
    if (el === null) return
    if (typeof IntersectionObserver === 'undefined') {
      onView()
      return
    }
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        io.disconnect()
        onView()
      }
    })
    io.observe(el)
    return () => io.disconnect()
  }, [ref, onView])
}

/** One executable example: editable code, a Run button, and the outcome of each expectation. */
export function LiveExample(props: { readonly example: Example }): ReactNode {
  const { example } = props
  const [code, setCode] = useState(example.code)
  const [result, setResult] = useState<ExampleResult | null>(null)
  const [running, setRunning] = useState(false)
  const box = useRef<HTMLDivElement | null>(null)
  const latest = useRef(0)
  const run = useCallback(async (text: string): Promise<void> => {
    const ticket = ++latest.current
    setRunning(true)
    // Let the button repaint before a long synchronous example takes the thread.
    await new Promise((r) => setTimeout(r, 0))
    const r = await runExample(text, { loaders: moduleLoaders, timeoutMs: 10_000 })
    if (ticket !== latest.current) return
    setResult(r)
    setRunning(false)
  }, [])
  const first = useCallback(() => void run(example.code), [run, example.code])
  useFirstView(box, first)
  const edited = code !== example.code
  const lines = code.split('\n').length
  const ok = result?.ok === true
  return (
    <div ref={box} className="site-code overflow-hidden rounded-lg border border-line bg-surface">
      <div className="flex items-center justify-between gap-2 border-b border-line px-3 py-1.5">
        <span className="truncate text-xs font-medium text-muted">
          {example.title || 'Example'}
          {result === null ? null : (
            <span className={ok ? STATUS.pass.className : STATUS.fail.className}>
              {' · '}
              {ok
                ? `${result.expectations.length} checked`
                : `${result.expectations.filter((e) => e.status !== 'pass').length || 1} problem(s)`}
              {` · ${Math.round(result.ms)} ms`}
            </span>
          )}
        </span>
        <span className="flex items-center gap-1">
          {edited ? (
            <button
              type="button"
              onClick={() => {
                setCode(example.code)
                void run(example.code)
              }}
              className="rounded px-2 py-0.5 text-xs text-muted hover:bg-bg hover:text-fg"
            >
              Reset
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => void run(code)}
            disabled={running}
            title="Run (Ctrl+Enter)"
            className="rounded border border-line bg-bg px-2.5 py-0.5 text-xs font-medium hover:border-accent hover:text-accent disabled:opacity-50"
          >
            {running ? 'Running…' : '▶ Run'}
          </button>
        </span>
      </div>
      <textarea
        value={code}
        onChange={(e) => setCode(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault()
            void run(code)
          }
        }}
        spellCheck={false}
        wrap="off"
        rows={lines}
        aria-label={`${example.title || 'Example'} code`}
        className="block w-full resize-y bg-transparent p-3 font-mono text-[12.5px] leading-relaxed whitespace-pre text-fg outline-none"
      />
      <div aria-live="polite" className="border-t border-line px-3 py-2 font-mono text-xs">
        {result === null ? (
          <span className="text-muted">{running ? 'running…' : 'not run yet'}</span>
        ) : (
          <>
            {result.expectations.length === 0 && result.error === undefined && result.logs.length === 0 ? (
              <span className="text-muted">{'Ran; no `// =>` lines to check.'}</span>
            ) : null}
            <ul className="m-0 list-none p-0">
              {result.expectations.map((e) => (
                <Outcome key={`${e.startLine}:${e.line}`} e={e} />
              ))}
            </ul>
            {result.logs.length === 0 ? null : (
              <pre className="m-0 mt-1 border-t border-line pt-1 whitespace-pre-wrap text-fg-secondary">
                {result.logs.join('\n')}
              </pre>
            )}
            {result.error === undefined ? null : (
              <p className="m-0 mt-1 text-red-600 dark:text-red-400">
                {result.error.line === undefined ? '' : `line ${result.error.line + 1}: `}
                {result.error.message}
              </p>
            )}
          </>
        )}
      </div>
    </div>
  )
}
