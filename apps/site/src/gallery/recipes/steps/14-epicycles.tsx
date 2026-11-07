import { epicycles } from 'mdml-engine/exposition'
import { Choice, ControlPanel, ExpositionView, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Epicycles',
  question: 'How do I show a curve drawn by a chain of rotating arrows, and how many arrows it needs?',
  explain:
    'An `epicycles` record is the Fourier decomposition of a closed curve: the target, the chain of arrows (each a rotation speed with a length and a starting angle), the curve their sum traces, the spectrum with the arrows kept and the share of energy they hold. `EpicyclePlayer` plays the chain drawing the curve, with the spectrum beside it. A heart or a star needs few arrows for its smooth part and many for its corners: keep more and the trace hugs the target, keep fewer and it rounds it off.',
  shows: ['epicycles', 'EpicyclePlayer'],
}

type EpicycleShape = 'heart' | 'star' | 'square'

const curve = (shape: EpicycleShape, n: number): [number, number][] =>
  Array.from({ length: n }, (_, i) => {
    const t = (2 * Math.PI * i) / n
    if (shape === 'heart')
      return [16 * Math.sin(t) ** 3, 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)]
    if (shape === 'star') {
      const r = 1 + 0.45 * Math.cos(5 * t) ** 3
      return [10 * r * Math.cos(t), 10 * r * Math.sin(t)]
    }
    const s = (i / n) * 4
    const side = Math.floor(s)
    const f = s - side
    return [
      [10 - 20 * f, 10],
      [-10, 10 - 20 * f],
      [-10 + 20 * f, -10],
      [10, -10 + 20 * f],
    ][side] as [number, number]
  })

export default function EpicyclesRecipe(): ReactNode {
  const [shape, setShape] = useState<EpicycleShape>('heart')
  const [kept, setKept] = useState(12)
  const record = useMemo(() => {
    const raw = curve(shape, 256)
    const n = raw.length
    const mx = raw.reduce((s, p) => s + p[0], 0) / n
    const my = raw.reduce((s, p) => s + p[1], 0) / n
    // The unit of the plane is the farthest point's distance from the centre, as the view draws it.
    const radius = Math.max(...raw.map((p) => Math.hypot(p[0] - mx, p[1] - my)))
    const pts = raw.map((p): [number, number] => [(p[0] - mx) / radius, (p[1] - my) / radius])
    const cx = 0
    const cy = 0
    // Discrete Fourier transform of the complex curve z = x + iy.
    const ks = Array.from({ length: n }, (_, j) => (j <= n / 2 ? j : j - n))
    const terms = ks.map((k) => {
      let re = 0
      let im = 0
      pts.forEach(([x, y], t) => {
        const a = (-2 * Math.PI * k * t) / n
        re += (x - cx) * Math.cos(a) - (y - cy) * Math.sin(a)
        im += (x - cx) * Math.sin(a) + (y - cy) * Math.cos(a)
      })
      return { k, re: re / n, im: im / n, mag: Math.hypot(re, im) / n }
    })
    const ranked = [...terms].sort((a, b) => b.mag - a.mag)
    const chosen = new Set(ranked.slice(0, kept).map((t) => t.k))
    const chain = ranked.slice(0, kept)
    const trace = pts.map((_, t) => {
      let x = cx
      let y = cy
      for (const c of chain) {
        const a = (2 * Math.PI * c.k * t) / n
        x += c.re * Math.cos(a) - c.im * Math.sin(a)
        y += c.re * Math.sin(a) + c.im * Math.cos(a)
      }
      return [x, y]
    })
    const energy = terms.reduce((s, t) => s + t.mag ** 2, 0)
    return epicycles({
      method: 'discrete Fourier transform',
      source: shape,
      target: { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) },
      centroid: [cx, cy],
      chain: { re: chain.map((c) => c.re), im: chain.map((c) => c.im), k: chain.map((c) => c.k) },
      trace: { x: trace.map((p) => p[0] ?? 0), y: trace.map((p) => p[1] ?? 0) },
      spectrum: {
        k: terms.map((t) => t.k),
        magnitude: terms.map((t) => t.mag),
        kept: terms.map((t) => chosen.has(t.k)),
      },
      energyKept: chain.reduce((s, c) => s + c.mag ** 2, 0) / energy,
      steps: 120,
    })
  }, [shape, kept])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="curve" value={shape} options={['heart', 'star', 'square'] as const} onChange={setShape} />
        <NumberField label="arrows kept" value={kept} onChange={setKept} min={1} max={80} integer />
      </ControlPanel>
      <ExpositionView record={record} title="A curve drawn by rotating arrows" />
    </div>
  )
}
