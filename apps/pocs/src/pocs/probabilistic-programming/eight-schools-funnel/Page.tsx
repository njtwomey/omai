import { ChainDiagnosticsView, Choice, ControlPanel } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { RunBar } from '../../../shared/RunBar.tsx'
import { useAnytimeRuns } from '../../../shared/useAnytimeRuns.ts'
import { FORMS, type Form, schoolsRuns } from './compute.ts'

const BUDGETS = [10, 30, 90]

export default function EightSchoolsFunnel(): ReactNode {
  const [budget, setBudget] = useState(30)
  const [form, setForm] = useState<Form>('centred')
  const specs = useMemo(() => schoolsRuns(), [])
  const runs = useAnytimeRuns(specs, budget * 1000)
  const of = (f: Form) => runs.runs.find((r) => r.id === f)
  const shown = of(form)?.value?.record
  return (
    <div className="flex flex-col gap-4">
      <RunBar runs={runs} budget={budget} onBudget={setBudget} budgets={BUDGETS} unit="iterations" />
      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <div className="flex min-w-0 flex-col gap-2">
          <ControlPanel columns={1}>
            <Choice label="the chains of" value={form} options={FORMS} onChange={setForm} />
          </ControlPanel>
          {shown === undefined ? (
            <p className="rounded-lg border border-line bg-surface p-6 text-sm text-muted">
              {form} is warming up: the draws appear once its adaptation is over.
            </p>
          ) : (
            <ChainDiagnosticsView record={shown} initial="pairs" filename={`eight-schools-${form}`} />
          )}
        </div>
        <div className="flex flex-col gap-2 text-sm">
          <h2 className="font-medium text-fg">The three forms, so far</h2>
          <table className="w-full text-left tabular-nums">
            <thead className="text-xs text-muted">
              <tr>
                <th className="py-1 font-normal">form</th>
                <th className="font-normal">draws</th>
                <th className="font-normal">divergent</th>
                <th className="font-normal">ESS of tau</th>
              </tr>
            </thead>
            <tbody>
              {FORMS.map((f) => {
                const k = of(f)?.value
                return (
                  <tr key={f} className="border-t border-line">
                    <td className="py-1">{f}</td>
                    <td>{k === undefined ? '…' : k.draws}</td>
                    <td className={k !== undefined && k.divergences > 0 ? 'text-(--wb-danger,#cf222e)' : ''}>
                      {k === undefined ? '…' : k.divergences}
                    </td>
                    <td>{k === undefined || Number.isNaN(k.essTau) ? '…' : Math.round(k.essTau)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <p className="text-xs text-muted">
            Three chains each. In the centred form the pairs chart, of the first school’s effect against log tau, is the
            funnel: the divergent transitions (marked) collect in its neck, where the effects are tied to the mean. The
            other two forms sample the same posterior of mu and tau without one.
          </p>
        </div>
      </div>
    </div>
  )
}
