import { recordsOf } from 'mdml-engine/exposition'
import { maternKernel, rbfKernel, scaleKernel } from 'mdml-engine/kernels'
import { gaussianProcessRegressor } from 'mdml-engine/methods/gp'
import { key, normal, uniform } from 'mdml-engine/random'
import { toFlat } from 'mdml-engine/tensor'
import { Choice, ControlPanel, NumberField, PosteriorView, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Posterior',
  question: 'How do I show a model’s predictive distribution, its uncertainty and draws from it?',
  explain:
    'A Gaussian process on one input returns a `posteriorGrid` record: the mean, variance and `nSamples` keyed draws on a grid, with the observations. `PosteriorView` draws the mean with bands of one and two standard deviations and the draws as thin lines in one colour. Switch to the prior to see functions the kernel believes in before any data (mean zero, the kernel’s variance); the points stay on the figure but are not conditioned on. Back on the posterior, every draw passes near the points and the bands pinch there; shorten the length scale and the draws wiggle more, raise the noise and they stop passing through the points. The seed changes only the draws.',
}

// region
const xs = Array.from(toFlat(uniform(key(4), [40], { low: -4, high: 4 })))
const eps = Array.from(toFlat(normal(key(5), [40])))
const ys = xs.map((x, i) => Math.sin(1.4 * x) + 0.3 * x + 0.15 * (eps[i] ?? 0))
const KERNELS = { rbf: 'RBF', matern32: 'Matérn 3/2', matern12: 'Matérn 1/2' } as const
type KernelName = keyof typeof KERNELS
const LIMITS = { x: [-4.4, 4.4], y: [-3.5, 3.5] } as const

export default function PosteriorRecipe(): ReactNode {
  const [draw, setDraw] = useState<'posterior' | 'prior'>('posterior')
  const [nSamples, setNSamples] = useState(10)
  const [n, setN] = useState(8)
  const [kernelName, setKernelName] = useState<KernelName>('rbf')
  const [lengthScale, setLengthScale] = useState(1)
  const [noise, setNoise] = useState(0.01)
  const [seed, setSeed] = useState(0)
  const grid = useMemo(() => {
    const base =
      kernelName === 'rbf'
        ? rbfKernel({ lengthScale })
        : maternKernel({ lengthScale, nu: kernelName === 'matern32' ? 1.5 : 0.5 })
    const gp = gaussianProcessRegressor({
      kernel: scaleKernel(base, 1),
      alpha: noise,
      optimize: false,
      surfaceResolution: 0,
      nSamples,
      draw,
      gridPoints: 200,
      gridLimits: LIMITS.x,
    })
    const fitted = gp.fit({ x: xs.slice(0, n).map((x) => [x]), y: ys.slice(0, n) }, { key: key(seed) })
    return recordsOf(fitted.exposition, 'posteriorGrid')[0]
  }, [draw, nSamples, n, kernelName, lengthScale, noise, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="draw from" value={draw} onChange={setDraw} options={['posterior', 'prior']} />
        <NumberField label="samples" value={nSamples} onChange={setNSamples} min={1} max={50} integer />
        <Choice
          label="kernel"
          value={kernelName}
          onChange={setKernelName}
          options={Object.entries(KERNELS).map(([value, label]) => ({ value: value as KernelName, label }))}
        />
        <NumberField label="length scale" value={lengthScale} onChange={setLengthScale} min={0.05} max={10} log />
        <NumberField label="noise variance" value={noise} onChange={setNoise} min={1e-6} max={1} log />
        <NumberField label="observations" value={n} onChange={setN} min={1} max={40} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      {grid !== undefined && <PosteriorView record={grid} limits={LIMITS} />}
    </div>
  )
}
// endregion
