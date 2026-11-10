import { posteriorPairs } from 'omai-engine/exposition'
import { ControlPanel, NumberField, PosteriorPairsPlayer, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Posterior pair plot',
  question: 'How do I overlay several methods’ draws of a posterior as a corner plot, and play them arriving?',
  explain:
    'A `posteriorPairs` record keeps, per method, a few hundred thinned draws with the step each arrived at, and at a modest number of cursor positions the density of each coordinate, the means and the correlations of the draws so far. The lower triangle scatters the draws, the diagonal overlays the marginals, the upper triangle writes each method’s correlation (or draws its ellipses) and the truth is marked in every panel. The player moves the cursor along the steps, or a page can own it (`cursor` and `onCursor`). The toy has an exact sampler, a random walk that starts far away and takes the run to reach the posterior, and a mean-field normal that is too narrow and misses the correlations.',
  shows: ['posteriorPairs', 'PosteriorPairsView', 'PosteriorPairsPlayer'],
}

const D = 6
const NAMES = Array.from({ length: D }, (_, j) => `w${j}`)
const MEAN = [0.5, -1, 2, 0, 1, -0.5]
const SD = [1, 0.6, 1.4, 0.8, 1.1, 0.7]
// Neighbouring coordinates are correlated, with the sign alternating: x_j = r x_(j-1) + sqrt(1 - r^2) z_j.
const R = [0.8, -0.6, 0.5, -0.7, 0.4]

/** Whitened draws `[n][D]` to draws of the posterior: the correlation by the chain above, then the means and sds. */
const posterior = (z: number[][]): Record<string, number[]> => {
  const x = z.map((row) => {
    const out: number[] = []
    row.forEach((v, j) => {
      const r = R[j - 1] ?? 0
      out.push(j === 0 ? v : r * (out[j - 1] as number) + Math.sqrt(1 - r * r) * v)
    })
    return out
  })
  return Object.fromEntries(
    NAMES.map((name, j) => [name, x.map((row) => (MEAN[j] as number) + (SD[j] as number) * (row[j] as number))]),
  )
}

export default function PosteriorPairs(): ReactNode {
  const [draws, setDraws] = useState(1500)
  const [width, setWidth] = useState(0.55)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const rows = (s: number): number[][] => {
      const flat = normals(s, draws * D)
      return Array.from({ length: draws }, (_, i) => flat.slice(i * D, (i + 1) * D))
    }
    // A random walk on the whitened coordinates, slow to forget its start four sds away.
    const walk = rows(seed + 1)
    const phi = 0.985
    let state: number[] = NAMES.map((_, j) => (j % 2 === 0 ? -4 : 4))
    const walked = walk.map((e) => {
      state = state.map((v, k) => phi * v + Math.sqrt(1 - phi * phi) * (e[k] as number))
      return state
    })
    const iid = rows(seed + 2).map((r) => r.map((v) => v * width))
    return posteriorPairs(
      [
        { name: 'nuts', draws: posterior(rows(seed)) },
        { name: 'random walk', draws: posterior(walked) },
        {
          name: 'mean-field',
          draws: Object.fromEntries(
            NAMES.map((n, j) => [n, iid.map((r) => (MEAN[j] as number) + (SD[j] as number) * (r[j] as number))]),
          ),
        },
      ],
      { frames: 30, maxPoints: 300, truth: Object.fromEntries(NAMES.map((n, j) => [n, MEAN[j] as number])) },
    )
  }, [draws, width, seed])
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="draws" value={draws} onChange={setDraws} min={200} max={4000} step={200} integer />
        <NumberField
          label="mean-field width (of the truth’s sd)"
          value={width}
          onChange={setWidth}
          min={0.2}
          max={1.5}
          step={0.05}
        />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <PosteriorPairsPlayer record={record} title="Three methods over one posterior" filename="posterior-pairs" />
    </div>
  )
}
