import { multiViewScene, renderMultiView } from 'mdml-engine/datasets'
import { fitNerfMethod, type NerfState, nerfDensityGrid, nerfProblem, renderView } from 'mdml-engine/fields'
import { foldIn, key } from 'mdml-engine/random'
import { type Item3D, point3d } from 'mdml-engine/scene'
import { bool, defaults, dict, int, real } from 'mdml-engine/space'
import { asTensor, fromStorage, toFlat } from 'mdml-engine/tensor'
import { type Config, Log, metricSeries, type Session } from 'mdml-engine/trace'
import { cartesian, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, SpaceControls, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { BakedSession, type BakedView } from '../../shared/BakedSession.tsx'
import { OrbitImage, type Pose, poseCamera, START_POSE, useProgressiveImage } from '../../shared/orbit.tsx'
import { PixelCanvas } from '../../shared/PixelCanvas.tsx'
import { cameraDots, frustum, Orbitable, rgba, up } from '../../shared/view3d.tsx'

const FRAMES = 30
const CURVE = figure({ width: 210, height: 150 })
/** The camera orbit of the scene: close enough that three objects fill most of the view. */
const RADIUS = 2.9
const FOV = 40
/** The drawn width of an image panel, and the sides the orbit view is rendered at: a quick one to follow a drag, then a finer one. */
const PANEL = 128
const SIDES = [20, 40] as const
/** The density is asked for on a grid of this many points a side, in the cube around the scene. */
const GRID = 16
const MOST_DRAWN = 1500

const CONTROLS = dict({
  objects: int(1, 4, { default: 3, label: 'objects in the scene' }),
  views: int(3, 16, { default: 12, label: 'training views' }),
  resolution: int(8, 24, { default: 16, label: 'image side (pixels)' }),
  samples: int(4, 32, { default: 8, label: 'samples per ray' }),
  steps: int(50, 1000, { default: 400, log: true, label: 'steps' }),
  learningRate: real(5e-4, 2e-2, { default: 1e-2, log: true, label: 'learning rate' }),
  viewDependent: bool({ default: false, label: 'colour depends on the viewing direction' }),
})
type Controls = {
  objects: number
  views: number
  resolution: number
  samples: number
  steps: number
  learningRate: number
  viewDependent: boolean
}
const START = defaults(CONTROLS) as unknown as Controls

/** The method's configuration for the page's controls: two held-out views, cameras close to the scene, the library's other defaults. */
const configOf = (c: Controls): Config => {
  const base = defaults(fitNerfMethod.config) as Config
  return {
    ...base,
    views: c.views,
    heldOut: 2,
    resolution: c.resolution,
    objects: c.objects,
    radius: RADIUS,
    fov: FOV,
    batchRays: 128,
    samples: c.samples,
    learningRate: c.learningRate,
    // The scene is lit by one fixed light, so a colour that ignores the direction fits it with far fewer views.
    ...(c.viewDependent ? {} : { directionEncoding: { kind: 'none' } }),
  } as Config
}

interface Frame {
  readonly step: number
  readonly heldPsnr: number
}

export default function NerfPage(): ReactNode {
  const [controls, setControls] = useState<Controls>(START)
  const config = useMemo(() => configOf(controls), [controls])
  // The scene's images and the model's architecture, from the key the session's run uses (its root key folded with 7).
  const problem = useMemo(() => nerfProblem(config as never, foldIn(key(0), 7)), [config])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel title="the scene and the training" columns={4}>
        <SpaceControls space={CONTROLS} value={controls} onChange={(v) => setControls(v as unknown as Controls)} />
      </ControlPanel>
      <BakedSession<Frame>
        method={fitNerfMethod}
        config={config}
        steps={controls.steps}
        frames={FRAMES}
        logs={{ 'nerf/loss': Log(), 'nerf/psnr': Log() }}
        bake={(session, t) => ({
          step: t,
          heldPsnr: session.probe(t, 'nerf/heldOutPsnr' as never) as unknown as number,
        })}
        onReset={() => setControls(START)}
      >
        {(view) => <View view={view} problem={problem} controls={controls} />}
      </BakedSession>
    </div>
  )
}

function View(props: {
  readonly view: BakedView<Frame>
  readonly problem: ReturnType<typeof nerfProblem>
  readonly controls: Controls
}): ReactNode {
  const { view, problem, controls } = props
  const { live, t, baked, run, seek } = view
  const side = controls.resolution
  const mode = useColourMode()
  const [pose, setPose] = useState<Pose>(START_POSE)
  const [threshold, setThreshold] = useState(0.2)
  // The network's weights at the step shown: read back from the session's checkpoints.
  const session = live.session as unknown as Session<NerfState, never>
  const state = useMemo(() => session.stateAt(t), [session, t])
  const { data, model } = problem
  const sampling = useMemo(() => ({ near: data.near, far: data.far, whiteBackground: data.whiteBackground }), [data])
  const truth = useMemo(() => Array.from(toFlat(asTensor(data.images)).slice(0, side * side * 3)), [data, side])
  const drawn = useMemo(
    () =>
      renderView(model, state.params, data.cameras[0] as never, [side, side], {
        ...sampling,
        samples: controls.samples,
        image: fromStorage(Float64Array.from(truth), [side, side, 3], 'float64'),
      }),
    [state, model, data, sampling, side, controls.samples, truth],
  )
  const scene = useMemo(() => multiViewScene(controls.objects), [controls.objects])
  const camera = useMemo(() => poseCamera(pose, RADIUS, FOV), [pose])
  const deps = [state, camera, controls.objects]
  // A quick look while dragging with the training's samples, a finer one with twice as many once it rests.
  const novel = useProgressiveImage(
    SIDES,
    (s) =>
      renderView(model, state.params, camera, [s, s], {
        ...sampling,
        samples: s === SIDES[0] ? controls.samples : 2 * controls.samples,
      }).rgb,
    deps,
  )
  const exact = useProgressiveImage(SIDES, (s) => Array.from(toFlat(renderMultiView(scene, camera, s, s))), deps)
  // The learned density, asked of the network on a grid: the volume as points.
  const grid = useMemo(
    () =>
      nerfDensityGrid(model, state.params, {
        bounds: [
          [-1, -1, -1],
          [1, 1, 1],
        ],
        resolution: GRID,
      }),
    [model, state],
  )
  const cloud = useMemo<Item3D[]>(() => {
    const n = GRID
    const keep: number[] = []
    for (let i = 0; i < grid.density.length; i++)
      if ((grid.density[i] as number) >= threshold * grid.maxDensity) keep.push(i)
    keep.sort((a, b) => (grid.density[b] as number) - (grid.density[a] as number))
    return keep.slice(0, MOST_DRAWN).map((i) => {
      const [x, y, z] = [i % n, Math.floor(i / n) % n, Math.floor(i / (n * n))]
      const at = [x, y, z].map((v) => -1 + (2 * (v + 0.5)) / n)
      const share = (grid.density[i] as number) / Math.max(grid.maxDensity, 1e-9)
      return point3d(up(at), {
        fill: rgba(
          [grid.colour[3 * i] as number, grid.colour[3 * i + 1] as number, grid.colour[3 * i + 2] as number],
          0.3 + 0.7 * share,
        ),
        r: 3,
      })
    })
  }, [grid, threshold])
  const items = useMemo<Item3D[]>(
    () => [...cloud, ...cameraDots(data.cameras), ...frustum(camera)],
    [cloud, data, camera],
  )
  const psnr = useMemo(() => metricSeries(run, 'nerf/psnr'), [run])
  const loss = useMemo(() => metricSeries(run, 'nerf/loss'), [run])
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
          [
            {
              x: train.map((p) => p[0]),
              y: train.map((p) => p[1]),
              name: 'training batches',
              width: 1.5,
              opacity: 0.6,
            },
          ],
          { mode },
        ),
        lines([{ x: shown.map((p) => p[0]), y: shown.map((p) => p[1]), name: 'held-out views', width: 2.5 }], {
          mode,
          colourOffset: 3,
        }),
        lines(
          [{ x: [t, t], y: [Math.min(...all, 0), Math.max(...all, 1)], colour: 'muted', dash: 'dashed', width: 1 }],
          { mode },
        ),
      ],
      { mode, figure: CURVE, x: { label: 'step', limits: [0, Math.max(1, live.head)] }, y: { label: 'PSNR (dB)' } },
    )
  }, [psnr, held, t, mode, live.head])
  const at = loss.steps.findLastIndex((st) => st <= t)
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
            caption={`its rendering, step ${t}${drawn.psnr === undefined ? '' : `: ${drawn.psnr.toFixed(1)} dB`}`}
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
          <dl className="flex flex-col gap-1.5 text-sm">
            <div>
              <dt className="text-xs text-muted">step</dt>
              <dd className="font-medium tabular-nums">
                {t} of {live.head}
              </dd>
            </div>
            <div>
              <dt className="text-xs text-muted">loss</dt>
              <dd className="font-medium tabular-nums">{(loss.values[at] ?? Number.NaN).toPrecision(3)}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted">held-out PSNR</dt>
              <dd className="font-medium tabular-nums">{last === undefined ? '' : `${last.heldPsnr.toFixed(1)} dB`}</dd>
            </div>
          </dl>
        </div>
      </div>
      <figure className="flex flex-col gap-1">
        <Orbitable items={items} width={520} height={340} />
        <figcaption className="flex max-w-[520px] flex-col gap-1 text-xs text-muted">
          <span>
            the density the network has at step {t}, asked on a {GRID} by {GRID} by {GRID} grid and drawn where it is at
            least{' '}
            <input
              type="range"
              min={0.05}
              max={0.9}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              aria-label="density threshold as a share of the largest"
              className="h-3 w-24 align-middle"
            />{' '}
            {Math.round(100 * threshold)}% of its largest, in the colour the network gives there; the training cameras
            (grey) and the novel camera (drawn nearer than it is)
          </span>
        </figcaption>
      </figure>
    </div>
  )
}
