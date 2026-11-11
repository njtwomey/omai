import type { ModelRun } from 'omai-engine/exposition'
import type { toStructure } from 'omai-engine/probprog/model'
import { ControlPanel, FigurePanel, ModelRunPlayer, StructureView, usePanels } from 'omai-render/react'
import { modelGraph } from 'omai-render/structures'
import { type ReactNode, useMemo } from 'react'
import { CodeBlock } from '../../../../site/src/shell/CodeBlock.tsx'
import { snippetOf } from '../../../../site/src/shell/snippet.ts'
import { type ProgramCheck, passes } from './program-check.ts'

const tone = (ok: boolean): string => (ok ? 'var(--wb-success)' : 'var(--wb-danger)')

function Badge(props: { readonly children: ReactNode; readonly color: string }): ReactNode {
  return (
    <span
      className="inline-block rounded border px-2 py-0.5 text-xs font-semibold uppercase tracking-wide"
      style={{ color: props.color, borderColor: props.color }}
    >
      {props.children}
    </span>
  )
}

/**
 * The page of an EP program: what the model is, the controls of the simulated data, the structure of the model drawn as a
 * diagram, what EP recovered (the children: the truth beside the posterior), the messages of the run, the recovery check with
 * its PASS or FAIL badge and the tolerance of each line, and at the bottom the program's own source, the code that ran
 * (`source` is the file imported with `?raw`, shown between its `// region` markers).
 *
 * @param props - The pieces of the page.
 * @returns The page.
 */
export function ProgramPage(props: {
  readonly id: string
  readonly equations: readonly string[]
  readonly controls: ReactNode
  /** The structure of the bound model (for its diagram), when it could be made. */
  readonly structure: (() => ReturnType<typeof toStructure> | undefined) | undefined
  /** The messages of EP's run as a record, made when the panel is open. */
  readonly record: () => ModelRun | undefined
  /** How the run ended, in words (or why there is none). */
  readonly status: string
  readonly check: ProgramCheck | undefined
  readonly source: string
  readonly file: string
  readonly children: ReactNode
}): ReactNode {
  const { id, equations, controls, structure, record, status, check, source, file, children } = props
  const panels = usePanels(`ep-program-${id}`, {
    graph: 'The model as a diagram',
    messages: 'EP’s messages, sweep by sweep',
  })
  const graph = useMemo(() => {
    const s = structure?.()
    return s === undefined ? undefined : modelGraph(s)
  }, [structure])
  const run = useMemo(() => (panels.active('messages') ? record() : undefined), [panels, record])
  const pass = check === undefined ? undefined : passes(check)
  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-2 rounded border border-line bg-surface px-3 py-2">
        {check === undefined ? (
          <Badge color="var(--wb-warning)">computing</Badge>
        ) : (
          <Badge color={tone(pass === true)}>{pass === true ? 'recovery check passes' : 'recovery check fails'}</Badge>
        )}
        <span className="text-xs text-muted" role="status">
          {status}
        </span>
      </div>
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">The model and the data</h2>
        <pre className="max-w-4xl overflow-x-auto rounded border border-line bg-surface px-3 py-2 font-mono text-xs leading-relaxed">
          {equations.join('\n')}
        </pre>
        <ControlPanel title="change the simulated data" columns={4}>
          {controls}
        </ControlPanel>
        <FigurePanel panels={panels} name="graph">
          {graph === undefined ? (
            <p className="text-sm text-muted">No diagram.</p>
          ) : (
            <StructureView structure={graph} frame={{ filename: `ep-program-${id}-model` }} />
          )}
        </FigurePanel>
      </section>
      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-semibold">What EP recovered, against the truth</h2>
        {children}
        <FigurePanel panels={panels} name="messages">
          {run === undefined ? (
            <p className="text-sm text-muted">Computing...</p>
          ) : (
            <ModelRunPlayer record={run} filename={`ep-program-${id}-messages`} />
          )}
        </FigurePanel>
      </section>
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">How we know it recovered the truth</h2>
        {check === undefined ? (
          <p className="text-sm text-muted">Computing the reference...</p>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm tabular-nums">
                <thead className="text-xs text-muted">
                  <tr>
                    <th className="py-1 font-normal">check</th>
                    <th className="font-normal">value</th>
                    <th className="font-normal">tolerance</th>
                    <th className="font-normal" />
                  </tr>
                </thead>
                <tbody>
                  {check.rows.map((r) => (
                    <tr key={r.label} className="border-t border-line">
                      <td className="py-1 pr-3">{r.label}</td>
                      <td className="pr-3">{r.value}</td>
                      <td className="pr-3 text-muted">{r.tolerance}</td>
                      <td style={{ color: r.gating === false ? 'var(--wb-muted)' : tone(r.pass) }}>
                        {r.gating === false ? (r.pass ? 'pass (info)' : 'fails (info)') : r.pass ? 'pass' : 'FAIL'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {check.notes.map((n) => (
              <p key={n} className="max-w-4xl text-xs text-muted">
                {n}
              </p>
            ))}
          </>
        )}
      </section>
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">The program</h2>
        <p className="max-w-4xl text-xs text-muted">
          This is the file the page imports: the simulation, the model and the inference that ran above, not a copy.
        </p>
        <CodeBlock code={snippetOf(source)} title={file} />
      </section>
    </div>
  )
}
