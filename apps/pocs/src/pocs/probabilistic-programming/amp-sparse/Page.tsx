import { ControlPanel, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { CurveChart } from '../../../shared/CurveChart.tsx'
import { RunBar } from '../../../shared/RunBar.tsx'
import { useAnytimeRuns } from '../../../shared/useAnytimeRuns.ts'
import { useDeferred } from '../../../shared/useDeferred.ts'
import { problemOf, type Setting } from './compute.ts'

export default function AmpSparse(): ReactNode {
  const [budget, setBudget] = useState(10)
  const [seed, setSeed] = useState(1)
  const [delta, setDelta] = useState(0.5)
  const [sparsity, setSparsity] = useState(0.1)
  const [noise, setNoise] = useState(0.01)
  const setting = useMemo<Setting>(
    () => ({ seed, columns: 3000, delta, sparsity, noiseVariance: noise }),
    [seed, delta, sparsity, noise],
  )
  // The matrix and the measurements take a moment to make: made once the page is up (and again for a new setting).
  const made = useDeferred(useMemo(() => () => problemOf(setting), [setting]))
  const specs = useMemo(() => (made === undefined ? undefined : [made.run]), [made])
  const runs = useAnytimeRuns(specs, budget * 1000)
  const kept = runs.runs[0]?.value
  const curves = useMemo(() => {
    if (made === undefined) return []
    const n = Math.max(kept?.tau2.length ?? 0, 1)
    const steps = (len: number): number[] => Array.from({ length: len }, (_, i) => i + 1)
    return [
      {
        name: 'state evolution (predicted)',
        colour: 'fg',
        width: 2,
        dash: 'dashed' as const,
        x: steps(n),
        y: made.predicted.slice(0, n).map((v) => Math.log10(v)),
      },
      {
        name: 'AMP (measured)',
        colour: 'tab0',
        width: 2.4,
        x: steps(kept?.tau2.length ?? 0),
        y: (kept?.tau2 ?? []).map((v) => Math.log10(v)),
      },
      {
        name: 'error of the estimate',
        colour: 'tab1',
        width: 2.4,
        x: steps(kept?.mse.length ?? 0),
        y: (kept?.mse ?? []).map((v) => Math.log10(v)),
      },
    ]
  }, [made, kept])
  const dots = useMemo(
    () =>
      made === undefined || kept === undefined
        ? []
        : [{ x: [...made.truth], y: [...kept.estimate], name: 'entries', colour: 'tab0', size: 3, opacity: 0.5 }],
    [made, kept],
  )
  return (
    <div className="flex flex-col gap-4">
      {made === undefined ? (
        <p className="text-sm text-muted">Making the measurements…</p>
      ) : (
        <RunBar runs={runs} budget={budget} onBudget={setBudget} budgets={[3, 10, 30]} unit="iterations" />
      )}
      <ControlPanel columns={4}>
        <NumberField label="measurements per unknown" value={delta} onChange={setDelta} min={0.2} max={1} step={0.05} />
        <NumberField label="sparsity" value={sparsity} onChange={setSparsity} min={0.02} max={0.4} step={0.01} />
        <NumberField label="noise variance" value={noise} onChange={setNoise} min={0.0001} max={0.5} log />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-sm font-medium text-fg">
            The noise level of each iteration, predicted and measured (log10)
          </h2>
          <CurveChart curves={curves} x={{ label: 'iteration' }} y={{ label: 'log10 of the variance' }} />
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-sm font-medium text-fg">The estimate against the signal, entry by entry</h2>
          <CurveChart curves={[]} dots={dots} x={{ label: 'true entry' }} y={{ label: 'estimate' }} aspect={1.5} />
        </div>
      </div>
    </div>
  )
}
