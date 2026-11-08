/**
 * Example "A drawing from rotating circles": a closed curve (a heart) is sampled at 128 points and its discrete Fourier
 * transform taken with the library's `fft`. Each coefficient is a circle: it turns at its own frequency with a radius and a
 * starting angle that come from the coefficient. Chained tip to tip, the largest few circles trace the curve. The figure
 * is a scene (the library's own drawing primitives) of the chain at one moment and the path its tip has traced so far.
 */
import { fft } from 'omai-engine/numerics/fourier'
import { circle, line, path, type SceneNode, scene } from 'omai-engine/scene'
import { toArray } from 'omai-engine/tensor'
import type { Mode } from '../_shared/colours.ts'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A drawing from rotating circles',
  summary: 'A closed curve as the sum of turning circles: its Fourier coefficients, chained tip to tip, draw it again.',
  readme: false,
  alt: 'A chain of rotating circles whose tip traces a heart-shaped curve',
  home: { label: 'Fourier', fps: 20, seconds: 11 },
}

const POINTS = 128
/** How many circles are used (the largest by radius). */
const CIRCLES = 14

interface Result {
  /** The curve that was sampled. */
  readonly curve: [number, number][]
  /** The circles, largest first: frequency (turns per loop), radius and starting angle. */
  readonly circles: readonly { readonly k: number; readonly radius: number; readonly phase: number }[]
}

export function run(): Outcome<Result> {
  const curve = Array.from({ length: POINTS }, (_, i): [number, number] => {
    const t = (2 * Math.PI * i) / POINTS
    return [16 * Math.sin(t) ** 3, 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)]
  })
  const spectrum = fft({ re: curve.map((p) => p[0]), im: curve.map((p) => p[1]) })
  const re = toArray(spectrum.re) as number[]
  const im = toArray(spectrum.im) as number[]
  const all = re.map((r, i) => ({
    // Frequencies above half the length are the negative ones.
    k: i <= POINTS / 2 ? i : i - POINTS,
    radius: Math.hypot(r, im[i] ?? 0) / POINTS,
    phase: Math.atan2(im[i] ?? 0, r),
  }))
  const circles = [...all].sort((a, b) => b.radius - a.radius).slice(0, CIRCLES)
  const kept = circles.reduce((s, c) => s + c.radius ** 2, 0)
  const total = all.reduce((s, c) => s + c.radius ** 2, 0)
  return {
    result: { curve, circles },
    notes: [`${CIRCLES} of ${POINTS} circles hold ${((100 * kept) / total).toFixed(2)}% of the curve's energy`],
  }
}

/** Played in the browser: 160 frames, one loop of the chain. */
export const frames = (): number => 160

const W = 520
const H = 420
const SCALE = 10.5
const at2 = (x: number, y: number): [number, number] => [W / 2 + SCALE * x, H / 2 - SCALE * y]

/** Where the chain's tip is, and where each circle is centred, at a share of the loop. */
function chain(result: Result, share: number): { centres: [number, number][]; tip: [number, number] } {
  let x = 0
  let y = 0
  const centres: [number, number][] = []
  for (const c of result.circles) {
    centres.push([x, y])
    const angle = 2 * Math.PI * c.k * share + c.phase
    x += c.radius * Math.cos(angle)
    y += c.radius * Math.sin(angle)
  }
  return { centres, tip: [x, y] }
}

export function View({ result, at, mode = 'light' }: { result: Result; at?: number; mode?: Mode }) {
  const done = at === undefined
  const share = done ? 1 : at / frames()
  const nodes: SceneNode[] = []
  // The curve that was sampled, faint, and the path the tip has traced so far, on top of it.
  nodes.push(
    path(
      [...result.curve, result.curve[0] ?? [0, 0]].map(([x, y]) => at2(x, y)),
      { stroke: 'grid', strokeWidth: 1.5 },
    ),
  )
  const steps = Math.max(2, Math.round(share * 200))
  const traced = Array.from({ length: steps + 1 }, (_, i) => at2(...chain(result, (share * i) / steps).tip))
  nodes.push(path(traced, { stroke: 'tab0', strokeWidth: 2.5 }))
  const { centres, tip } = chain(result, share)
  result.circles.forEach((c, j) => {
    const [cx, cy] = at2(...(centres[j] ?? [0, 0]))
    if (c.radius * SCALE > 1.5)
      nodes.push(circle(cx, cy, c.radius * SCALE, { stroke: 'muted', strokeWidth: 1, fill: 'none', opacity: 0.6 }))
    const next = centres[j + 1] ?? tip
    const [nx, ny] = at2(...next)
    nodes.push(line(cx, cy, nx, ny, { stroke: 'fg', strokeWidth: 1.2 }))
  })
  const [tx, ty] = at2(...tip)
  nodes.push(circle(tx, ty, 4, { fill: 'tab1' }))
  return (
    <Figure width={W} height={H + 30} title={meta.alt} mode={mode}>
      <Panel
        width={W}
        height={H}
        title={`${CIRCLES} circles, turning at whole numbers of turns per loop`}
        still={scene(W, H, nodes)}
      />
    </Figure>
  )
}
