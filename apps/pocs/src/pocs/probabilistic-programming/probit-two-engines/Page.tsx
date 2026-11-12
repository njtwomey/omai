import { modelRunFromTape } from 'omai-engine/exposition'
import { Choice, ControlPanel, ModelRunPlayer, PosteriorComparisonView } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { RunBar } from '../../../shared/RunBar.tsx'
import { useAnytimeRuns } from '../../../shared/useAnytimeRuns.ts'
import { useDeferred } from '../../../shared/useDeferred.ts'
import { compare, exactAnswer, PICTURE, probitRuns } from './compute.ts'

type Diagram = 'EP' | 'NUTS'

export default function ProbitTwoEngines(): ReactNode {
  const [budget, setBudget] = useState(10)
  const [diagram, setDiagram] = useState<Diagram>('EP')
  // The exact answer on a grid, made once the page is up (about a second and a half).
  const truth = useDeferred(exactAnswer)
  const specs = useMemo(() => (truth === undefined ? undefined : probitRuns()), [truth])
  const runs = useAnytimeRuns(specs, budget * 1000)
  const ep = runs.runs.find((r) => r.id === 'ep')
  const nuts = runs.runs.find((r) => r.id === 'nuts')
  const laplace = runs.runs.find((r) => r.id === 'laplace')
  const comparison = useMemo(
    () =>
      truth === undefined
        ? undefined
        : compare(
            [
              { name: 'EP', kept: ep?.value },
              { name: 'NUTS', kept: nuts?.value },
              { name: 'Laplace', kept: laplace?.value },
            ],
            truth.marginals,
          ),
    [truth, ep?.value, nuts?.value, laplace?.value],
  )
  const nutsHistory = nuts?.history
  const record = useMemo(() => {
    if (diagram === 'EP') return ep?.value?.record
    if (nutsHistory === undefined || nutsHistory.length === 0) return undefined
    return modelRunFromTape(
      PICTURE,
      {
        snapshots: nutsHistory.flatMap((h) =>
          h.value.posterior.sites.length === 0
            ? []
            : [{ step: h.t, snapshot: { summary: h.value.posterior.summary() } }],
        ),
      },
      { method: 'nuts' },
    )
  }, [diagram, ep?.value, nutsHistory])
  return (
    <div className="flex flex-col gap-4">
      {truth === undefined ? (
        <p className="text-sm text-muted">Computing the exact posterior on a grid…</p>
      ) : (
        <RunBar runs={runs} budget={budget} onBudget={setBudget} unit="iterations" />
      )}
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-2">
          <ControlPanel columns={1}>
            <Choice label="the diagram shows" value={diagram} options={['EP', 'NUTS'] as const} onChange={setDiagram} />
          </ControlPanel>
          {record === undefined ? (
            <p className="text-sm text-muted">Nothing from {diagram} yet.</p>
          ) : (
            <ModelRunPlayer record={record} filename="probit-model-run" />
          )}
        </div>
        <div className="min-w-0">
          {comparison === undefined ? (
            <p className="text-sm text-muted">The marginals appear with the first snapshot.</p>
          ) : (
            <PosteriorComparisonView record={comparison} filename="probit-comparison" />
          )}
        </div>
      </div>
    </div>
  )
}
