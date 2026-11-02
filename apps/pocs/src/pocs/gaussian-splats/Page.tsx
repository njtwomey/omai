import { type SplatEllipses, type SplatRender, fitSplatsMethod, splatsProblem } from 'mdml-engine/fields'
import { foldIn, key } from 'mdml-engine/random'
import { bool, defaults, dict, int } from 'mdml-engine/space'
import { asTensor, toFlat } from 'mdml-engine/tensor'
import { type Config, Log, metricSeries } from 'mdml-engine/trace'
import { cartesian, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, SpaceControls, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { BakedSession, type BakedView } from '../../shared/BakedSession.tsx'
import { PixelCanvas } from '../../shared/PixelCanvas.tsx'

const FRAMES = 40
const CURVE = figure({ width: 420, height: 240 })

const CONTROLS = dict({
  views: int(3, 16, { default: 6, label: 'training views' }),
  resolution: int(8, 24, { default: 16, label: 'image side (pixels)' }),
  gaussians: int(5, 200, { default: 30, label: 'Gaussians at the start' }),
  densify: bool({ default: false, label: 'adaptive density control (split and clone)' }),
  steps: int(50, 1000, { default: 300, log: true, label: 'steps' }),
})
type Controls = { views: number; resolution: number; gaussians: number; densify: boolean; steps: number }
const START = defaults(CONTROLS) as unknown as Controls

/** The method's configuration for the page's controls: one object, two held-out views, the library's other defaults. */
const configOf = (c: Controls): Config =>
  ({
    ...(defaults(fitSplatsMethod.config) as Config),
    views: c.views,
    heldOut: 2,
    resolution: c.resolution,
    objects: 1,
    gaussians: c.gaussians,
    densify: c.densify,
    densifyUntil: Math.max(21, c.steps - 20),
  }) as Config

interface Frame {
  readonly step: number
  readonly render: SplatRender
  readonly ellipses: SplatEllipses
  readonly heldPsnr: number
}

export default function SplatsPage(): ReactNode {
  const [controls, setControls] = useState<Controls>(START)
  const config = useMemo(() => configOf(controls), [controls])
  // The first training view, from the generator and key the session's run uses (its root key folded with 7).
  const truth = useMemo(() => {
    const data = splatsProblem(config as never, foldIn(key(0), 7)).data
    const size = controls.resolution * controls.resolution * 3
    return Array.from(toFlat(asTensor(data.images)).slice(0, size))
  }, [config, controls.resolution])
  return (
    <div className="flex flex-col gap-4">
      <p className="max-w-3xl text-sm text-muted">
        3-D Gaussian splatting represents a scene as a cloud of coloured, semi-transparent 3-D Gaussians. To draw a
        view, each Gaussian is projected onto the image as an ellipse and the ellipses are blended from far to near;
        training moves, stretches, recolours and fades the Gaussians until the drawing matches the photographs. Below is
        one training view (left), the drawing at the step you are on (middle), and the Gaussians themselves as ellipses
        over it (right). Adaptive density control splits and clones Gaussians where the picture is still poor, so the
        count grows. These are toy sizes: a single object, {controls.resolution} by {controls.resolution} pixel views, a
        few hundred steps.
      </p>
      <ControlPanel title="the scene and the training" columns={2}>
        <SpaceControls space={CONTROLS} value={controls} onChange={(v) => setControls(v as unknown as Controls)} />
      </ControlPanel>
      <BakedSession<Frame>
        method={fitSplatsMethod}
        config={config}
        steps={controls.steps}
        frames={FRAMES}
        logs={{ 'splats/loss': Log(), 'splats/psnr': Log(), 'splats/count': Log() }}
        bake={(session, t) => ({
          step: t,
          render: session.probe(t, 'splats/render' as never, { view: 0 }) as unknown as SplatRender,
          ellipses: session.probe(t, 'splats/ellipses' as never, { view: 0 }) as unknown as SplatEllipses,
          heldPsnr: session.probe(t, 'splats/heldOutPsnr' as never) as unknown as number,
        })}
        onReset={() => setControls(START)}
      >
        {(view) => <View view={view} truth={truth} side={controls.resolution} />}
      </BakedSession>
    </div>
  )
}

/** A dashed vertical line at the step shown. */
function marker(t: number, lo: number, hi: number, mode: ReturnType<typeof useColourMode>): ReturnType<typeof lines> {
  return lines([{ x: [t, t], y: [lo, hi], colour: 'muted', dash: 'dashed', width: 1 }], { mode })
}

function View(props: {
  readonly view: BakedView<Frame>
  readonly truth: readonly number[]
  readonly side: number
}): ReactNode {
  const { view, truth, side } = props
  const { live, t, frame, baked, run, seek } = view
  const mode = useColourMode()
  const psnr = useMemo(() => metricSeries(run, 'splats/psnr'), [run])
  const count = useMemo(() => metricSeries(run, 'splats/count'), [run])
  const held = (() => {
    const seen = new Map<number, number>()
    for (let s = 0; s <= live.head; s++) {
      const f = baked.at(s)
      if (f !== undefined) seen.set(f.step, f.heldPsnr)
    }
    return [...seen.entries()].sort((a, b) => a[0] - b[0])
  })()
  const psnrOption = useMemo(() => {
    const train = psnr.steps.map((st, i) => [st, psnr.values[i] ?? Number.NaN] as const).filter(([st]) => st <= t)
    const shown = held.filter(([st]) => st <= t)
    const all = [...psnr.values, ...held.map((h) => h[1])].filter(Number.isFinite)
    return cartesian(
      [
        lines(
          [{ x: train.map((p) => p[0]), y: train.map((p) => p[1]), name: 'training views', width: 1.5, opacity: 0.6 }],
          {
            mode,
          },
        ),
        lines([{ x: shown.map((p) => p[0]), y: shown.map((p) => p[1]), name: 'held-out views', width: 2.5 }], {
          mode,
          colourOffset: 3,
        }),
        marker(t, Math.min(...all, 0), Math.max(...all, 1), mode),
      ],
      { mode, figure: CURVE, x: { label: 'step', limits: [0, Math.max(1, live.head)] }, y: { label: 'PSNR (dB)' } },
    )
  }, [psnr, held, t, mode, live.head])
  const countOption = useMemo(() => {
    const upto = count.steps.map((st, i) => [st, count.values[i] ?? Number.NaN] as const).filter(([st]) => st <= t)
    return cartesian(
      [
        lines([{ x: upto.map((p) => p[0]), y: upto.map((p) => p[1]), name: 'Gaussians', width: 2 }], { mode }),
        marker(t, 0, Math.max(1, ...count.values), mode),
      ],
      {
        mode,
        figure: CURVE,
        x: { label: 'step', limits: [0, Math.max(1, live.head)] },
        y: { label: 'number of Gaussians' },
      },
    )
  }, [count, t, mode, live.head])
  const at = count.steps.findLastIndex((st) => st <= t)
  const e = frame?.ellipses
  return (
    <div className="flex flex-col gap-4">
      <div className="grid items-start gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))' }}>
        <PixelCanvas width={side} height={side} rgb={truth} caption="training view 0: truth" />
        <PixelCanvas
          width={side}
          height={side}
          rgb={frame?.render.rgb}
          caption={`render at step ${t}${frame?.render.psnr === undefined ? '' : `: ${frame.render.psnr.toFixed(1)} dB`}`}
        />
        <PixelCanvas
          width={side}
          height={side}
          rgb={frame?.render.rgb}
          caption="the Gaussians as ellipses (one standard deviation)"
        >
          {e === undefined ? null : (
            <svg
              viewBox={`0 0 ${e.width} ${e.height}`}
              className="absolute inset-0 h-full w-full"
              role="img"
              aria-label="the Gaussians"
            >
              {e.ellipses.map((g) => (
                <ellipse
                  key={g.index}
                  cx={g.x}
                  cy={g.y}
                  rx={Math.max(g.major, 0.15)}
                  ry={Math.max(g.minor, 0.15)}
                  transform={`rotate(${(g.angle * 180) / Math.PI} ${g.x} ${g.y})`}
                  fill={`rgb(${g.rgb.map((v) => Math.round(255 * Math.min(1, Math.max(0, v)))).join(',')})`}
                  fillOpacity={0.35 * g.opacity}
                  stroke="white"
                  strokeOpacity={0.35 + 0.5 * g.opacity}
                  strokeWidth={0.08}
                />
              ))}
            </svg>
          )}
        </PixelCanvas>
        <div className="flex flex-col gap-2">
          <Chart option={psnrOption} figure={CURVE} onInit={seek as never} />
          <Chart option={countOption} figure={CURVE} onInit={seek as never} />
          <dl className="grid grid-cols-3 gap-x-6 text-sm">
            <div>
              <dt className="text-xs text-muted">step</dt>
              <dd className="font-medium tabular-nums">
                {t} of {live.head}
              </dd>
            </div>
            <div>
              <dt className="text-xs text-muted">Gaussians</dt>
              <dd className="font-medium tabular-nums">{count.values[at] ?? ''}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted">held-out PSNR</dt>
              <dd className="font-medium tabular-nums">
                {frame === undefined ? '' : `${frame.heldPsnr.toFixed(1)} dB`}
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  )
}
