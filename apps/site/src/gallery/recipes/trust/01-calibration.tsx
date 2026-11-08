import { calibrationRecord, type ReliabilityCurve } from 'omai-engine/exposition'
import { calibrationCurve } from 'omai-engine/metrics'
import { toFlat } from 'omai-engine/tensor'
import { Choice, ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { sigmoid, uniforms } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Calibration',
  question: 'How do I show whether a model’s probabilities can be taken at face value, and for whom?',
  explain:
    'A `calibration` record holds one reliability curve per model or group: the mean forecast and the observed frequency in each bin, the bin counts and the summary errors (ECE, MCE, Brier). `CalibrationView` draws the curves against the diagonal, points sized by count, with the errors in the legend. The toy forecaster reports a probability through a temperature, and one group’s forecasts are sharper than the other’s, so a single average curve would hide that one is over-confident. Switch to quantile bins and the sparse tails get as many points as the middle.',
  shows: ['calibration', 'CalibrationView'],
}

const logit = (p: number): number => Math.log(p / (1 - p))

export default function CalibrationRecipe(): ReactNode {
  const [temperature, setTemperature] = useState(0.6)
  const [bins, setBins] = useState(10)
  const [strategy, setStrategy] = useState<'uniform' | 'quantile'>('uniform')
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const n = 4000
    const u = uniforms(seed, 2 * n)
    const truth = Array.from({ length: n }, (_, i) => Math.min(0.97, Math.max(0.03, u[i] ?? 0.5)))
    const y = truth.map((q, i) => ((u[n + i] ?? 1) < q ? 1 : 0))
    const group = truth.map((_, i) => i % 2)
    const forecast = (t: (g: number) => number): number[] => truth.map((q, i) => sigmoid(logit(q) / t(group[i] ?? 0)))
    const curve = (name: string, p: readonly number[], keep: (i: number) => boolean): ReliabilityCurve => {
      const idx = p.flatMap((_, i) => (keep(i) ? [i] : []))
      const yy = idx.map((i) => y[i] ?? 0)
      const pp = idx.map((i) => p[i] ?? 0)
      const c = calibrationCurve(yy, pp, { nBins: bins, strategy })
      const [counts, probTrue, probPred] = [c.counts, c.probTrue, c.probPred].map(toFlat)
      const ece = (counts ?? []).reduce(
        (a, k, j) => a + (k * Math.abs((probTrue?.[j] ?? 0) - (probPred?.[j] ?? 0))) / idx.length,
        0,
      )
      const mce = Math.max(...(probTrue ?? []).map((v, j) => Math.abs(v - (probPred?.[j] ?? 0))))
      const brier = pp.reduce((a, v, i) => a + (v - (yy[i] ?? 0)) ** 2, 0) / idx.length
      return { name, meanPredicted: probPred ?? [], observed: probTrue ?? [], counts: counts ?? [], ece, mce, brier }
    }
    const calibrated = forecast(() => 1)
    const model = forecast((g) => (g === 0 ? 1 : temperature))
    return calibrationRecord({
      nBins: bins,
      strategy,
      split: 'held out',
      curves: [
        curve('calibrated', calibrated, () => true),
        curve('model, group A', model, (i) => group[i] === 0),
        curve('model, group B', model, (i) => group[i] === 1),
      ],
    })
  }, [temperature, bins, strategy, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField
          label="group B temperature"
          value={temperature}
          onChange={setTemperature}
          min={0.2}
          max={3}
          step={0.05}
          log
        />
        <NumberField label="bins" value={bins} onChange={setBins} min={3} max={30} integer />
        <Choice label="bins are" value={strategy} options={['uniform', 'quantile'] as const} onChange={setStrategy} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Reliability by group" />
    </div>
  )
}
