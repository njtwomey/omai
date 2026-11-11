import { Choice, ControlPanel, SbcRunView } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { CodeBlock } from '../../../../site/src/shell/CodeBlock.tsx'
import { RunBar } from '../../shared/RunBar.tsx'
import { useAnytimeRuns } from '../../shared/useAnytimeRuns.ts'
import { calibrationRuns, METHODS, REPLICATIONS } from './compute.ts'
import { codeOf } from './shown.ts'

type Id = (typeof METHODS)[number]['id']

export default function CalibrationSbc(): ReactNode {
  const [budget, setBudget] = useState(15)
  const [shown, setShown] = useState<Id>('mean-field')
  const specs = useMemo(() => calibrationRuns(), [])
  const runs = useAnytimeRuns(specs, budget * 1000)
  const of = (id: Id) => runs.runs.find((r) => r.id === id)?.value
  const method = METHODS.find((m) => m.id === shown)
  const record = of(shown)?.record
  return (
    <div className="flex flex-col gap-4">
      <RunBar runs={runs} budget={budget} onBudget={setBudget} budgets={[5, 15, 45]} unit="replications" />
      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="flex min-w-0 flex-col gap-2">
          <ControlPanel columns={1}>
            <Choice
              label="the ranks of"
              value={shown}
              options={METHODS.map((m) => ({ value: m.id, label: m.label }))}
              onChange={setShown}
            />
          </ControlPanel>
          {record === undefined ? (
            <p className="rounded-lg border border-line bg-surface p-6 text-sm text-muted">
              The ranks appear from the tenth replication ({of(shown)?.done ?? 0} so far).
            </p>
          ) : (
            <SbcRunView record={record} filename={`calibration-${shown}`} />
          )}
        </div>
        <div className="flex flex-col gap-3 text-sm">
          <h2 className="font-medium text-fg">The verdicts so far (of {REPLICATIONS} replications)</h2>
          <ul className="m-0 flex list-none flex-col gap-2 p-0">
            {METHODS.map((m) => {
              const r = of(m.id)?.record
              return (
                <li key={m.id} className="flex flex-col gap-0.5 border-t border-line pt-2">
                  <div className="flex justify-between tabular-nums">
                    <span className="text-fg">{m.label}</span>
                    <span className="text-muted">{r === undefined ? 'waiting' : `${r.replications} replications`}</span>
                  </div>
                  <span
                    className={r === undefined ? 'text-muted' : r.passes ? 'text-fg' : 'text-(--wb-danger,#cf222e)'}
                  >
                    {r === undefined
                      ? '…'
                      : r.passes
                        ? 'calibrated, as far as these replications can say'
                        : 'not calibrated'}
                  </span>
                </li>
              )
            })}
          </ul>
          <p className="text-xs text-muted">{method?.note}</p>
          <p className="text-xs text-muted">
            Few replications give wide bands: a curve inside them is no proof of calibration, one outside them is
            evidence against it. More budget, more replications.
          </p>
        </div>
      </div>
      <CodeBlock code={codeOf(method?.id)} title="compute.ts: calibration-sbc" className="max-w-4xl" />
    </div>
  )
}
