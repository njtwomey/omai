/**
 * An executable example, as aifn-engine's docs cells: the code in an editable box with a Run button, and under it the
 * outcome of every `// =>` line (the expected value beside the value the code gave, mismatches marked), what the code
 * printed, and any error with its example line. It runs once on first view, through whitebox-meta's runner with the
 * module map for its imports (`useLiveExample` from whitebox-meta/react), on the page's thread (asynchronous waits are
 * bounded; a synchronous loop is not).
 */
import { type ReactNode, useMemo } from 'react'
import { moduleLoaders } from 'whitebox-meta/modules'
import { useLiveExample } from 'whitebox-meta/react'
import type { Example, ExampleResult, ExpectationResult } from 'whitebox-meta/schema'
import { highlight } from '../shell/highlight.ts'

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

/**
 * The code as an editor: the site's highlighter draws the code, and a transparent textarea over it takes the typing
 * (caret and selection stay visible). `notes` puts a mark at the end of given lines: the outcome beside each `// =>`.
 */
function Editor(props: {
  readonly code: string
  readonly onChange: (code: string) => void
  readonly onRun: () => void
  readonly label: string
  readonly notes: ReadonlyMap<number, ReactNode>
}): ReactNode {
  const { code, onChange, onRun, label, notes } = props
  const lines = code.split('\n')
  return (
    <div className="overflow-x-auto">
      <div className="relative inline-block min-w-full">
        <pre
          aria-hidden="true"
          className="pointer-events-none m-0 p-3 font-mono text-[12.5px] leading-relaxed whitespace-pre"
        >
          {lines.map((line, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: lines are positions in the code
            <div key={i}>
              {line === ''
                ? ' '
                : highlight(line).map((t, j) =>
                    t.kind === undefined ? (
                      t.text
                    ) : (
                      // biome-ignore lint/suspicious/noArrayIndexKey: tokens are positions in the line
                      <span key={j} className={`tok-${t.kind}`}>
                        {t.text}
                      </span>
                    ),
                  )}
              {notes.get(i)}
            </div>
          ))}
        </pre>
        <textarea
          value={code}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
              e.preventDefault()
              onRun()
            }
          }}
          spellCheck={false}
          wrap="off"
          aria-label={label}
          className="absolute inset-0 m-0 h-full w-full resize-none overflow-hidden bg-transparent p-3 font-mono text-[12.5px] leading-relaxed whitespace-pre text-transparent caret-fg outline-none selection:bg-accent/30"
        />
      </div>
    </div>
  )
}

/** The marks beside the code for a result: ✓ or ✗ with the value at each `// =>` line, the error at its line. */
function notesFor(result: ExampleResult): Map<number, ReactNode> {
  const notes = new Map<number, ReactNode>()
  for (const e of result.expectations) {
    if (e.status === 'skipped') continue
    const s = STATUS[e.status]
    notes.set(
      e.line,
      <span className={`select-none ${s.className}`}>
        {'  '}
        {s.mark}
        {e.status === 'pass' ? null : <span>{` got ${oneLine(e.actual, 60) || '(nothing)'}`}</span>}
      </span>,
    )
  }
  if (result.error?.line !== undefined)
    notes.set(
      result.error.line,
      <span className="select-none text-red-600 dark:text-red-400">{`  ← ${oneLine(result.error.message, 80)}`}</span>,
    )
  return notes
}

/** One executable example: editable code, a Run button, and the outcome of each expectation. */
export function LiveExample(props: { readonly example: Example }): ReactNode {
  const { example } = props
  const {
    code,
    setCode,
    edited,
    outcome: result,
    running,
    run,
    reset,
    ref: box,
  } = useLiveExample(example, {
    loaders: moduleLoaders,
  })
  const shown = result?.result ?? null
  const ok = shown?.ok === true
  const notes = useMemo(
    () => (result === null || result.code !== code ? new Map<number, ReactNode>() : notesFor(result.result)),
    [result, code],
  )
  return (
    <div ref={box} className="site-code overflow-hidden rounded-lg border border-line bg-surface">
      <div className="flex items-center justify-between gap-2 border-b border-line px-3 py-1.5">
        <span className="truncate text-xs font-medium text-muted">
          {example.title || 'Example'}
          {shown === null ? null : (
            <span className={ok ? STATUS.pass.className : STATUS.fail.className}>
              {' · '}
              {ok
                ? `${shown.expectations.length} checked`
                : `${shown.expectations.filter((e) => e.status !== 'pass').length || 1} problem(s)`}
              {` · ${Math.round(shown.ms)} ms`}
            </span>
          )}
        </span>
        <span className="flex items-center gap-1">
          {edited ? (
            <button
              type="button"
              onClick={reset}
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
      <Editor
        code={code}
        onChange={setCode}
        onRun={() => void run(code)}
        label={`${example.title || 'Example'} code`}
        notes={notes}
      />
      <div aria-live="polite" className="border-t border-line px-3 py-2 font-mono text-xs">
        {shown === null ? (
          <span className="text-muted">{running ? 'running…' : 'not run yet'}</span>
        ) : (
          <>
            {shown.expectations.length === 0 && shown.error === undefined && shown.logs.length === 0 ? (
              <span className="text-muted">{'Ran; no `// =>` lines to check.'}</span>
            ) : null}
            <ul className="m-0 list-none p-0">
              {shown.expectations.map((e) => (
                <Outcome key={`${e.startLine}:${e.line}`} e={e} />
              ))}
            </ul>
            {shown.logs.length === 0 ? null : (
              <pre className="m-0 mt-1 border-t border-line pt-1 whitespace-pre-wrap text-fg-secondary">
                {shown.logs.join('\n')}
              </pre>
            )}
            {shown.error === undefined ? null : (
              <p className="m-0 mt-1 text-red-600 dark:text-red-400">
                {shown.error.line === undefined ? '' : `line ${shown.error.line + 1}: `}
                {shown.error.message}
              </p>
            )}
          </>
        )}
      </div>
    </div>
  )
}
