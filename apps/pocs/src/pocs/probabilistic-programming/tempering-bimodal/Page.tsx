import { ChainDiagnosticsView, Choice, ControlPanel } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { RunBar } from '../../../shared/RunBar.tsx'
import { useAnytimeRuns } from '../../../shared/useAnytimeRuns.ts'
import { type Kept, modeRuns } from './compute.ts'

type Which = 'NUTS' | 'parallel tempering'

/** The share of each chain in the upper mode, as bars with the truth (one half) marked. */
function ModeWeights(props: { readonly name: string; readonly kept: Kept | undefined }): ReactNode {
  const { name, kept } = props
  return (
    <div className="flex flex-col gap-1.5">
      <p className="text-xs text-muted">{name}</p>
      {(kept?.upper.length ?? 0) === 0 ? (
        <p className="text-xs text-muted">waiting for draws…</p>
      ) : (
        kept?.upper.map((u, c) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: chains are identified by their position
          <div key={c} className="flex items-center gap-2 text-xs tabular-nums">
            <span className="w-14 text-muted">chain {c + 1}</span>
            <div className="relative h-3 flex-1 overflow-hidden rounded-sm bg-bg">
              <div className="h-full bg-(--wb-accent)" style={{ width: `${100 * u}%` }} />
              <div className="absolute inset-y-0 left-1/2 w-px bg-fg" />
            </div>
            <span className="w-10 text-right">{Math.round(100 * u)}%</span>
          </div>
        ))
      )}
    </div>
  )
}

export default function TemperingBimodal(): ReactNode {
  const [budget, setBudget] = useState(15)
  const [which, setWhich] = useState<Which>('parallel tempering')
  const specs = useMemo(() => modeRuns(), [])
  const runs = useAnytimeRuns(specs, budget * 1000)
  const nuts = runs.runs.find((r) => r.id === 'nuts')?.value
  const pt = runs.runs.find((r) => r.id === 'tempering')?.value
  const shown = which === 'NUTS' ? nuts : pt
  return (
    <div className="flex flex-col gap-4">
      <RunBar runs={runs} budget={budget} onBudget={setBudget} budgets={[5, 15, 45]} unit="iterations" />
      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="flex min-w-0 flex-col gap-2">
          <ControlPanel columns={1}>
            <Choice
              label="the chains of"
              value={which}
              options={['NUTS', 'parallel tempering'] as const}
              onChange={setWhich}
            />
          </ControlPanel>
          {shown?.record === undefined ? (
            <p className="rounded-lg border border-line bg-surface p-6 text-sm text-muted">
              Waiting for the first draws.
            </p>
          ) : (
            <ChainDiagnosticsView record={shown.record} initial="trace" filename={`bimodal-${which}`} />
          )}
        </div>
        <div className="flex flex-col gap-4 text-sm">
          <h2 className="font-medium text-fg">How much of each chain is in the upper mode</h2>
          <ModeWeights name="NUTS" kept={nuts} />
          <ModeWeights name="parallel tempering" kept={pt} />
          <p className="text-xs text-muted">The line is the truth: half of the mass in each mode.</p>
          <h2 className="font-medium text-fg">Swaps between temperatures</h2>
          {pt?.swaps === undefined ? (
            <p className="text-xs text-muted">waiting…</p>
          ) : (
            <>
              <ul className="m-0 flex list-none flex-col gap-1 p-0 text-xs tabular-nums">
                {pt.swaps.map((r, j) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: pairs are identified by their position
                  <li key={j} className="flex justify-between">
                    <span className="text-muted">
                      replicas {j} and {j + 1}
                    </span>
                    <span>{Math.round(100 * r)}% accepted</span>
                  </li>
                ))}
              </ul>
              <p className="text-xs text-muted">
                Round trips between the hottest and the coldest replica so far: {pt.roundTrips ?? 0}.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
