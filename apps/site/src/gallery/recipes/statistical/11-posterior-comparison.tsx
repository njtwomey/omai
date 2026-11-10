import { gamma, normal } from 'omai-engine/dists'
import { posteriorComparison } from 'omai-engine/exposition'
import { agree, distributionOracle } from 'omai-engine/probprog/validation'
import { item } from 'omai-engine/tensor'
import { ControlPanel, NumberField, PosteriorComparisonView, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals, uniforms } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Posterior comparison',
  question: 'How do I show several methods’ posteriors against the exact one, and how far off each is?',
  explain:
    'A `posteriorComparison` record holds, per quantity, the oracle’s CDF on a grid and each method’s distribution (the quantiles of its draws), so the densities, the ECDFs or the difference of each ECDF from the oracle’s CDF go on one set of axes, and the kernel’s agreement table as a marked table: the mean error in posterior sds, the sd ratio, the Kolmogorov and Wasserstein distances, flagged where they pass the tolerance. The toy has three methods on a skewed target and a normal one: an exact sampler, a normal that is too narrow, and a chain that sits to one side.',
  shows: ['posteriorComparison', 'PosteriorComparisonView'],
}

const TRUTH = {
  rate: distributionOracle('rate', gamma(4, 2)),
  mu: distributionOracle('mu', normal(0.6, Math.SQRT1_2)),
}

export default function PosteriorComparison(): ReactNode {
  const [narrow, setNarrow] = useState(0.55)
  const [shift, setShift] = useState(0.35)
  const [draws, setDraws] = useState(1500)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const exact = (name: 'rate' | 'mu', s: number): number[] => {
      const dist = name === 'rate' ? gamma(4, 2) : normal(0.6, Math.SQRT1_2)
      return uniforms(s, draws).map((u) => item(dist.ppf(Math.min(1 - 1e-9, Math.max(1e-9, u)))))
    }
    const scaled = (name: 'rate' | 'mu', s: number, by: number, move: number): number[] => {
      const o = TRUTH[name]
      return normals(s, draws).map((z) => o.mean + move * o.sd + by * o.sd * z)
    }
    const approximations = [
      { name: 'nuts', draws: { rate: exact('rate', seed), mu: exact('mu', seed + 1) } },
      {
        name: 'mean-field normal',
        draws: { rate: scaled('rate', seed + 2, narrow, 0), mu: scaled('mu', seed + 3, narrow, 0) },
      },
      {
        name: 'stuck chain',
        draws: {
          rate: exact('rate', seed + 4).map((v) => v + shift * TRUTH.rate.sd),
          mu: exact('mu', seed + 5).map((v) => v + shift * TRUTH.mu.sd),
        },
      },
    ]
    return posteriorComparison(approximations, {
      truth: TRUTH,
      table: agree(approximations, { oracle: TRUTH }),
      quantilePoints: 150,
    })
  }, [narrow, shift, draws, seed])
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField
          label="normal’s width (of the truth’s sd)"
          value={narrow}
          onChange={setNarrow}
          min={0.2}
          max={2}
          step={0.05}
        />
        <NumberField label="chain’s shift (in sd)" value={shift} onChange={setShift} min={0} max={1} step={0.05} />
        <NumberField label="draws" value={draws} onChange={setDraws} min={200} max={4000} step={200} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <PosteriorComparisonView
        record={record}
        title="Three methods against the exact posterior"
        filename="posterior-comparison"
      />
    </div>
  )
}
