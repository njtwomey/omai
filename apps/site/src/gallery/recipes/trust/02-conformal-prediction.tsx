import { conformalRecord } from 'mdml-engine/exposition'
import { Choice, ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Conformal prediction',
  question: 'How do I show prediction intervals or sets that cover the truth at a promised rate?',
  explain:
    'A `conformal` record is a promise (cover the truth with probability 1 − α) and its evidence. For regression it holds a prediction with lower and upper bounds along x and the calibration scores behind the interval’s half-width; for classification, a set of classes per row and the sets’ sizes; for a stream, the running coverage with and without adaptation. `ConformalView` draws each, and plays the stream. Here a made-up regressor is wrong more where the noise is higher: the bands widen to keep coverage near the target. In the stream the noise doubles halfway; adaptive α pulls coverage back up.',
  shows: ['conformal', 'ConformalView'],
}

type Task = 'regression' | 'classification' | 'stream'

const quantile = (v: readonly number[], p: number): number => {
  const s = [...v].sort((a, b) => a - b)
  return s[Math.min(s.length - 1, Math.ceil(p * (s.length + 1)) - 1)] ?? 0
}

export default function ConformalRecipe(): ReactNode {
  const [task, setTask] = useState<Task>('regression')
  const [alpha, setAlpha] = useState(0.1)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    if (task === 'regression') {
      const n = 400
      const x = Array.from({ length: n }, (_, i) => (i / (n - 1)) * 6)
      const e = normals(seed, 2 * n)
      const truth = (v: number): number => Math.sin(v) + 0.3 * v
      const sd = (v: number): number => 0.1 + 0.12 * v
      const y = x.map((v, i) => truth(v) + sd(v) * (e[i] ?? 0))
      // Calibration rows are an independent draw; the score is |y - prediction| / sd(x), so one quantile serves every x.
      const cal = x.map((_, i) => Math.abs(e[n + i] ?? 0))
      const q = quantile(cal, 1 - alpha)
      const prediction = x.map(truth)
      const band = x.map((v) => q * sd(v))
      const inside = y.filter((v, i) => Math.abs(v - (prediction[i] ?? 0)) <= (band[i] ?? 0)).length / n
      return conformalRecord({
        task: 'regression',
        method: 'split conformal',
        alpha,
        coverage: inside,
        scores: cal,
        quantile: q,
        x,
        xLabel: 'x',
        y,
        prediction,
        lower: prediction.map((p, i) => p - (band[i] ?? 0)),
        upper: prediction.map((p, i) => p + (band[i] ?? 0)),
      })
    }
    if (task === 'classification') {
      const n = 300
      const classes = ['cat', 'dog', 'fox', 'owl']
      const g = normals(seed, 2 * n * classes.length)
      const probs = Array.from({ length: 2 * n }, (_, i) => {
        const truth = i % classes.length
        const logits = classes.map((_, c) => 1.4 * (c === truth ? 1 : 0) + 0.9 * (g[i * classes.length + c] ?? 0))
        const z = logits.map(Math.exp)
        const s = z.reduce((a, b) => a + b, 0)
        return { truth, p: z.map((v) => v / s) }
      })
      const cal = probs.slice(0, n).map((r) => 1 - (r.p[r.truth] ?? 0))
      const q = quantile(cal, 1 - alpha)
      const test = probs.slice(n)
      const sets = test.map((r) => r.p.flatMap((p, c) => (1 - p <= q ? [c] : [])))
      const truth = test.map((r) => r.truth)
      const cover = sets.filter((s, i) => s.includes(truth[i] ?? -1)).length / n
      return conformalRecord({
        task: 'classification',
        method: 'least ambiguous sets',
        alpha,
        coverage: cover,
        scores: cal,
        quantile: q,
        classes,
        sets,
        truth,
      })
    }
    // Online: the noise doubles halfway; a fixed interval loses coverage, an adaptive alpha recovers it.
    const T = 400
    const e = normals(seed, T)
    const scale = (t: number): number => (t < T / 2 ? 1 : 2)
    const t = Array.from({ length: T }, (_, i) => i)
    const window = 60
    const scores = e.map((v, i) => Math.abs(v) * scale(i))
    let a = alpha
    const fixedQ = quantile(scores.slice(0, window), 1 - alpha)
    const hit: number[] = []
    const hitFixed: number[] = []
    const alphas: number[] = []
    const widths: number[] = []
    for (let i = 0; i < T; i++) {
      const recent = scores.slice(Math.max(0, i - window), i)
      const w = recent.length < 10 ? fixedQ : quantile(recent, Math.min(0.999, Math.max(0.5, 1 - a)))
      const miss = (scores[i] ?? 0) > w ? 1 : 0
      hit.push(1 - miss)
      hitFixed.push((scores[i] ?? 0) <= fixedQ ? 1 : 0)
      alphas.push(a)
      widths.push(w)
      a += 0.05 * (alpha - miss)
    }
    const running = (h: readonly number[]): number[] => {
      let s = 0
      return h.map((v, i) => {
        s += v
        return s / (i + 1)
      })
    }
    const coverage = running(hit)
    return conformalRecord({
      task: 'regression',
      method: 'adaptive conformal',
      alpha,
      coverage: coverage[T - 1] ?? 0,
      x: t,
      xLabel: 'time',
      y: e.map((v, i) => v * scale(i)),
      prediction: t.map(() => 0),
      lower: widths.map((w) => -w),
      upper: widths,
      scores: scores.slice(0, window),
      quantile: fixedQ,
      stream: { t, coverage, baseline: running(hitFixed), alphas, width: widths },
    })
  }, [task, alpha, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice
          label="task"
          value={task}
          options={['regression', 'classification', 'stream'] as const}
          onChange={setTask}
        />
        <NumberField label="miss rate α" value={alpha} onChange={setAlpha} min={0.02} max={0.5} step={0.02} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Intervals that cover at the promised rate" />
    </div>
  )
}
