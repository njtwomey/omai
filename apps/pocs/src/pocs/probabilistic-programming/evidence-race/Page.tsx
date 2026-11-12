import { Choice, ControlPanel, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { CurveChart } from '../../../shared/CurveChart.tsx'
import { RunBar } from '../../../shared/RunBar.tsx'
import { useAnytimeRuns } from '../../../shared/useAnytimeRuns.ts'
import { ESTIMATORS, evidenceRuns, exactLogEvidence, MODELS, type ModelId } from './compute.ts'

/** Half the width of the window the estimates are drawn in, in nats around the exact value. */
const WINDOW = 0.5

export default function EvidenceRace(): ReactNode {
  const [budget, setBudget] = useState(15)
  const [model, setModel] = useState<ModelId>('normal-normal')
  const [seed, setSeed] = useState(1)
  const exact = useMemo(() => exactLogEvidence(model), [model])
  const specs = useMemo(() => evidenceRuns(model, seed), [model, seed])
  const runs = useAnytimeRuns(specs, budget * 1000)
  // Each estimate as a bar of two standard errors either side and a dot, one row per estimator.
  const picture = useMemo(() => {
    const rows = ESTIMATORS.map(([id, label], i) => {
      const e = runs.runs.find((r) => r.id === id)?.value
      return { id, label, i, e }
    })
    return {
      curves: [
        {
          name: 'exact',
          colour: 'fg',
          width: 2,
          dash: 'dashed' as const,
          x: [exact, exact],
          y: [-0.5, ESTIMATORS.length - 0.5],
        },
        ...rows.flatMap((r) =>
          r.e === undefined || r.e.standardError === undefined
            ? []
            : [
                {
                  colour: `tab${r.i}`,
                  width: 3,
                  x: [r.e.value - 2 * r.e.standardError, r.e.value + 2 * r.e.standardError],
                  y: [r.i, r.i],
                },
              ],
        ),
      ],
      dots: rows.flatMap((r) =>
        r.e === undefined ? [] : [{ x: [r.e.value], y: [r.i], name: r.label, colour: `tab${r.i}`, size: 9 }],
      ),
    }
  }, [runs.runs, exact])
  // The error of each estimate as the run goes on.
  const errors = useMemo(
    () =>
      ESTIMATORS.flatMap(([id, label], i) => {
        const h = runs.runs.find((r) => r.id === id)?.history ?? []
        const kept = h.filter((p) => p.value !== undefined)
        return kept.length === 0
          ? []
          : [
              {
                name: label,
                colour: `tab${i}`,
                width: 2,
                markers: true,
                x: kept.map((p) => p.elapsedMs / 1000),
                y: kept.map((p) => Math.max(-1, Math.min(1, (p.value?.value ?? Number.NaN) - exact))),
              },
            ]
      }),
    [runs.runs, exact],
  )
  const xAxis = useMemo(
    () => ({ label: 'log evidence', limits: [exact - WINDOW, exact + WINDOW] as [number, number] }),
    [exact],
  )
  const yAxis = useMemo(() => ({ show: false, limits: [-0.6, ESTIMATORS.length - 0.4] as [number, number] }), [])
  const timeAxis = useMemo(() => ({ label: 'seconds' }), [])
  const errAxis = useMemo(
    () => ({ label: 'estimate minus exact (clipped at 1)', limits: [-1.05, 1.05] as [number, number] }),
    [],
  )
  return (
    <div className="flex flex-col gap-4">
      <RunBar runs={runs} budget={budget} onBudget={setBudget} budgets={[5, 15, 45]} unit="steps" />
      <ControlPanel columns={2}>
        <Choice
          label="model"
          value={model}
          options={MODELS.map((m) => ({ value: m.id, label: m.label }))}
          onChange={setModel}
        />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-sm font-medium text-fg">
            The estimates, with two standard errors, against the exact value
          </h2>
          <CurveChart curves={picture.curves} dots={picture.dots} x={xAxis} y={yAxis} />
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-sm font-medium text-fg">How far each is from the exact value, as time goes on</h2>
          <CurveChart curves={errors} x={timeAxis} y={errAxis} />
        </div>
      </div>
      <table className="w-full max-w-2xl text-left text-sm tabular-nums">
        <thead className="text-xs text-muted">
          <tr>
            <th className="py-1 font-normal">estimator</th>
            <th className="font-normal">log evidence</th>
            <th className="font-normal">standard error</th>
            <th className="font-normal">error</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-t border-line text-muted">
            <td className="py-1">exact</td>
            <td>{exact.toFixed(3)}</td>
            <td />
            <td />
          </tr>
          {ESTIMATORS.map(([id, label]) => {
            const e = runs.runs.find((r) => r.id === id)?.value
            return (
              <tr key={id} className="border-t border-line">
                <td className="py-1">{label}</td>
                <td>{e?.value.toFixed(3) ?? '…'}</td>
                <td>{e?.standardError === undefined ? '' : e.standardError.toFixed(3)}</td>
                <td>{e === undefined ? '' : (e.value - exact).toFixed(3)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
