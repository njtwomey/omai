import { sbcRun } from 'omai-engine/exposition'
import { sbc } from 'omai-engine/probprog/validation'
import { foldIn, key, normal } from 'omai-engine/random'
import { item } from 'omai-engine/tensor'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Simulation-based calibration',
  question:
    'How do I show whether an inference method’s posteriors are calibrated, with a band that says how much wobble is chance?',
  explain:
    'An `sbcRun` record holds, per quantity, the ranks of the true value among the posterior draws over many simulated datasets: their histogram with the exact binomial band, and the ECDF of the ranks minus the uniform CDF with the exact simultaneous band (Säilynoja et al.), so a curve that stays inside is calibrated and one that leaves it is not, with the shape saying how (a smile is too confident, a frown too wide, a slope biased). The toy method gives three quantities: one exact, one whose posterior is scaled by the width, one shifted by the bias.',
  shows: ['sbcRun', 'SbcRunView'],
}

// A toy conjugate problem per quantity: m ~ N(0, 1), y ~ N(m, 1), whose exact posterior is N(y / 2, 1 / 2).
const NAMES = ['exact', 'width', 'bias'] as const

export default function SimulationCalibration(): ReactNode {
  const [width, setWidth] = useState(0.5)
  const [bias, setBias] = useState(0.5)
  const [replications, setReplications] = useState(200)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const scale = [1, width, 1]
    const shift = [0, 0, bias]
    const result = sbc({
      key: key(seed),
      replications,
      numRanks: 19,
      nullSimulations: 400,
      simulate: (k) => {
        const truth: Record<string, number> = {}
        const y: number[] = []
        for (const [i, n] of NAMES.entries()) {
          const m = item(normal(foldIn(k, 2 * i), []))
          truth[n] = m
          y.push(m + item(normal(foldIn(k, 2 * i + 1), [])))
        }
        return { data: y, truth }
      },
      fit: (y, k) =>
        Object.fromEntries(
          NAMES.map((n, i) => [
            n,
            Array.from(
              normal(foldIn(k, i), [19], {
                loc: (y[i] ?? 0) / 2 + (shift[i] ?? 0),
                scale: (scale[i] ?? 1) * Math.SQRT1_2,
              }).data,
            ),
          ]),
        ),
    })
    return sbcRun(result)
  }, [width, bias, replications, seed])
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField label="posterior width" value={width} onChange={setWidth} min={0.2} max={3} step={0.05} log />
        <NumberField label="bias (in sd)" value={bias} onChange={setBias} min={-1.5} max={1.5} step={0.1} />
        <NumberField
          label="replications"
          value={replications}
          onChange={setReplications}
          min={50}
          max={500}
          step={50}
          integer
        />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView
        record={record}
        title="Ranks of the true value among the draws"
        filename="simulation-calibration"
      />
    </div>
  )
}
