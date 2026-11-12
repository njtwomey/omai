import { multiViewScene, renderMultiView } from 'omai-engine/datasets'
import {
  fitSplatsMethod,
  gaussianEllipsoids,
  gaussiansToPly,
  renderSplatView,
  type SplatState,
  splatsProblem,
} from 'omai-engine/fields'
import { foldIn, key } from 'omai-engine/random'
import { face3d, type Item3D } from 'omai-engine/scene'
import { bool, defaults, dict, int } from 'omai-engine/space'
import { asTensor, toFlat } from 'omai-engine/tensor'
import { type Config, Log, metricSeries, type Session } from 'omai-engine/trace'
import { cartesian, lines } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, SpaceControls, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { BakedSession, type BakedView } from '../../../shared/BakedSession.tsx'
import { OrbitImage, type Pose, poseCamera, START_POSE, useProgressiveImage } from '../../../shared/orbit.tsx'
import { PixelCanvas } from '../../../shared/PixelCanvas.tsx'
import { cameraDots, frustum, Orbitable, rgba, up } from '../../../shared/view3d.tsx'

const FRAMES = 30
const CURVE = figure({ width: 210, height: 150 })
/** The camera orbit of the scene: close enough that three objects fill most of the view. */
const RADIUS = 2.9
const FOV = 40
/** The drawn width of an image panel, and the sides the orbit view is rendered at: a quick one to follow a drag, then a finer one. */
const PANEL = 128
const SIDES = [16, 32, 64] as const
const MOST_DRAWN = 300

const CONTROLS = dict({
  objects: int(1, 4, { default: 3, label: 'objects in the scene' }),
  views: int(3, 16, { default: 12, label: 'training views' }),
  resolution: int(8, 24, { default: 16, label: 'image side (pixels)' }),
  gaussians: int(5, 200, { default: 60, label: 'Gaussians at the start' }),
  densify: bool({ default: true, label: 'adaptive density control' }),
  steps: int(50, 1000, { default: 300, log: true, label: 'steps' }),
})
type Controls = {
  objects: number
  views: number
  resolution: number
  gaussians: number
  densify: boolean
  steps: number
}
const START = defaults(CONTROLS) as unknown as Controls

/** The method's configuration for the page's controls: two held-out views, cameras close to the scene, the library's other defaults. */
const configOf = (c: Controls): Config =>
  ({
    ...(defaults(fitSplatsMethod.config) as Config),
    views: c.views,
    heldOut: 2,
    resolution: c.resolution,
    objects: c.objects,
    radius: RADIUS,
    fov: FOV,
    gaussians: c.gaussians,
    initialScale: 0.2,
    meansRate: 0.01,
    scalesRate: 0.01,
    densify: c.densify,
    densifyUntil: Math.max(21, c.steps - 20),
  }) as Config

interface Frame {
  readonly step: number
  readonly heldPsnr: number
}

export default function SplatsPage(): ReactNode {
  const [controls, setControls] = useState<Controls>(START)
  const config = useMemo(() => configOf(controls), [controls])
  // The data the session's run uses (its root key folded with 7): the training views and their cameras.
  const problem = useMemo(() => splatsProblem(config as never, foldIn(key(0), 7)), [config])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel title="the scene and the training" columns={6}>
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
          heldPsnr: session.probe(t, 'splats/heldOutPsnr' as never) as unknown as number,
        })}
        onReset={() => setControls(START)}
      >
        {(view) => <View view={view} problem={problem} controls={controls} />}
      </BakedSession>
    </div>
  )
}

/** A dashed vertical line at the step shown. */
function marker(t: number, lo: number, hi: number, mode: ReturnType<typeof useColourMode>): ReturnType<typeof lines> {
  return lines([{ x: [t, t], y: [lo, hi], colour: 'muted', dash: 'dashed', width: 1 }], { mode })
}

/** Offers the Gaussians at the step shown as a `.ply` file, the layout the 3-D Gaussian splatting viewers read. */
function PlyButton(props: { readonly state: SplatState; readonly step: number }): ReactNode {
  return (
    <button
      type="button"
      className="w-fit cursor-pointer rounded-md border border-line bg-bg px-2.5 py-1 text-xs text-fg hover:bg-surface"
      title="The Gaussians at this step as a binary .ply (x y z, f_dc, opacity logit, log scale, rotation), the 3DGS layout"
      onClick={() => {
        const blob = new Blob([gaussiansToPly(props.state.params) as BlobPart], { type: 'application/octet-stream' })
        const link = document.createElement('a')
        link.href = URL.createObjectURL(blob)
        link.download = `gaussians-step-${props.step}.ply`
        link.click()
        URL.revokeObjectURL(link.href)
      }}
    >
      download .ply
    </button>
  )
}

function View(props: {
  readonly view: BakedView<Frame>
  readonly problem: ReturnType<typeof splatsProblem>
  readonly controls: Controls
}): ReactNode {
  const { view, problem, controls } = props
  const { live, t, baked, run, seek } = view
  const side = controls.resolution
  const mode = useColourMode()
  const [pose, setPose] = useState<Pose>(START_POSE)
  // The Gaussians at the step shown: read back from the session's checkpoints.
  const session = live.session as unknown as Session<SplatState, never>
  const state = useMemo(() => session.stateAt(t), [session, t])
  const truth = useMemo(
    () => Array.from(toFlat(asTensor(problem.data.images)).slice(0, side * side * 3)),
    [problem, side],
  )
  const drawn = useMemo(
    () => renderSplatView(state.params, problem.data.cameras[0] as never, [side, side], { background: [1, 1, 1] }),
    [state, problem, side],
  )
  const scene = useMemo(() => multiViewScene(controls.objects), [controls.objects])
  const camera = useMemo(() => poseCamera(pose, RADIUS, FOV), [pose])
  const deps = [state, camera, controls.objects]
  const novel = useProgressiveImage(
    SIDES,
    (s) => renderSplatView(state.params, camera, [s, s], { background: [1, 1, 1] }).rgb,
    deps,
  )
  const exact = useProgressiveImage(SIDES, (s) => Array.from(toFlat(renderMultiView(scene, camera, s, s))), deps)
  const items = useMemo<Item3D[]>(() => {
    const all = gaussianEllipsoids(state.params)
      .map((g) => ({ g, weight: g.opacity * Math.hypot(...g.axes[0]) }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, MOST_DRAWN)
    const rings = all.flatMap(({ g }) =>
      [1, 2].map((k) => {
        const [a, b] = [g.axes[0], g.axes[k] as readonly number[]]
        const ring = Array.from({ length: 16 }, (_, i) => {
          const th = (2 * Math.PI * i) / 16
          return up(g.centre.map((c, d) => c + 2 * (Math.cos(th) * (a[d] as number) + Math.sin(th) * (b[d] as number))))
        })
        return face3d(ring, {
          fill: rgba(g.rgb, 0.12 + 0.4 * g.opacity),
          stroke: rgba(g.rgb, 0.35 + 0.5 * g.opacity),
          strokeWidth: 0.6,
        })
      }),
    )
    return [...rings, ...cameraDots(problem.data.cameras), ...frustum(camera)]
  }, [state, problem, camera])
  // The held-out PSNR of the steps prepared, and the curves.
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
          { mode },
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
        y: { label: 'Gaussians' },
      },
    )
  }, [count, t, mode, live.head])
  const at = count.steps.findLastIndex((st) => st <= t)
  const last = baked.at(t)
  const timing = (n: typeof novel): string =>
    `${n.side} by ${n.side}${n.ms[n.stage] === undefined ? '' : `, ${Math.round(n.ms[n.stage] as number)} ms`}`
  return (
    <div className="flex flex-wrap items-start gap-x-5 gap-y-3">
      <div className="flex flex-col gap-3">
        <div className="flex gap-3">
          <PixelCanvas width={side} height={side} rgb={truth} size={PANEL} caption="training view 0: truth" />
          <PixelCanvas
            width={side}
            height={side}
            rgb={drawn.rgb}
            size={PANEL}
            caption={`its drawing, step ${t}: ${drawn.psnr === undefined ? '' : `${drawn.psnr.toFixed(1)} dB`}`.replace(
              /: $/,
              '',
            )}
          />
          <OrbitImage
            pose={pose}
            onPose={setPose}
            image={novel}
            limit={PANEL}
            caption={`novel view, drag me (${timing(novel)})`}
          />
          <OrbitImage pose={pose} onPose={setPose} image={exact} limit={PANEL} caption="the true scene from there" />
        </div>
        <div className="flex items-start gap-3">
          <Chart option={psnrOption} figure={CURVE} onInit={seek as never} />
          <Chart option={countOption} figure={CURVE} onInit={seek as never} />
          <dl className="flex flex-col gap-1.5 text-sm">
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
              <dd className="font-medium tabular-nums">{last === undefined ? '' : `${last.heldPsnr.toFixed(1)} dB`}</dd>
            </div>
            <PlyButton state={state} step={t} />
          </dl>
        </div>
      </div>
      <figure className="flex flex-col gap-1">
        <Orbitable items={items} width={520} height={340} />
        <figcaption className="max-w-[520px] text-xs text-muted">
          the Gaussians at step {t} as ellipsoids (two cross-sections each, two standard deviations, the{' '}
          {Math.min(MOST_DRAWN, state.params.means.shape[0] ?? 0)} most visible), the training cameras (grey), the novel
          camera (blue, drawn nearer than it is)
        </figcaption>
      </figure>
    </div>
  )
}
