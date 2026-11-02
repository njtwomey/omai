import { fitNerfMethod, type NerfImage, nerfProblem } from 'mdml-engine/fields'
import { foldIn, key } from 'mdml-engine/random'
import { defaults, dict, int, real } from 'mdml-engine/space'
import { asTensor, toFlat } from 'mdml-engine/tensor'
import { type Config, Log, metricSeries } from 'mdml-engine/trace'
import { cartesian, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, SpaceControls, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { BakedSession, type BakedView } from '../../shared/BakedSession.tsx'
import { PixelCanvas } from '../../shared/PixelCanvas.tsx'

const FRAMES = 40
const CURVE = figure({ width: 420, height: 280 })

const CONTROLS = dict({
  views: int(3, 16, { default: 6, label: 'training views' }),
  resolution: int(8, 24, { default: 16, label: 'image side (pixels)' }),
  samples: int(4, 32, { default: 8, label: 'samples per ray' }),
  steps: int(50, 1000, { default: 400, log: true, label: 'steps' }),
  learningRate: real(5e-4, 2e-2, { default: 5e-3, log: true, label: 'learning rate' }),
})
type Controls = { views: number; resolution: number; samples: number; steps: number; learningRate: number }
const START = defaults(CONTROLS) as unknown as Controls

/** The method's configuration for the page's controls: one object, two held-out views, the library's other defaults. */
const configOf = (c: Controls): Config =>
  ({
    ...(defaults(fitNerfMethod.config) as Config),
    views: c.views,
    heldOut: 2,
    resolution: c.resolution,
    batchRays: 128,
    objects: 1,
    samples: c.samples,
    learningRate: c.learningRate,
  }) as Config

interface Frame {
  readonly step: number
  readonly train: NerfImage
  readonly held: NerfImage
  readonly heldPsnr: number
}

export default function NerfPage(): ReactNode {
  const [controls, setControls] = useState<Controls>(START)
  const config = useMemo(() => configOf(controls), [controls])
  // The scene's images, from the generator and key the session's run uses (its root key folded with 7).
  const scene = useMemo(() => {
    const data = nerfProblem(config as never, foldIn(key(0), 7)).data
    const side = controls.resolution
    const slice = (images: typeof data.images, v: number): number[] =>
      Array.from(toFlat(asTensor(images)).slice(v * side * side * 3, (v + 1) * side * side * 3))
    return {
      train: Array.from({ length: controls.views }, (_, v) => slice(data.images, v)),
      held: slice((data.heldOut as NonNullable<typeof data.heldOut>).images, 0),
    }
  }, [config, controls.views, controls.resolution])
  return (
    <div className="flex flex-col gap-4">
      <p className="max-w-3xl text-sm text-muted">
        A neural radiance field is a network that maps a 3-D position and a viewing direction to a colour and a density.
        To draw a pixel, the camera&apos;s ray through it is sampled at a few points, the network is asked at each, and
        the answers are composited (volume rendering); training moves the weights until the rendered pixels match the
        photographs. Below are the training views (top row), then one training view and one held-out view, each as the
        truth (left) and as the network renders it at the step you are on (right). The held-out view was never trained
        on, so it is the test of whether the network learned the scene. These are toy sizes: a single object,{' '}
        {controls.resolution} by {controls.resolution} pixel views, a few hundred steps.
      </p>
      <ControlPanel title="the scene and the training" columns={2}>
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
          train: session.probe(t, 'nerf/render' as never, { view: 0 }) as unknown as NerfImage,
          held: session.probe(t, 'nerf/render' as never, { heldOut: 0 }) as unknown as NerfImage,
          heldPsnr: session.probe(t, 'nerf/heldOutPsnr' as never) as unknown as number,
        })}
        onReset={() => setControls(START)}
      >
        {(view) => <View view={view} scene={scene} side={controls.resolution} />}
      </BakedSession>
    </div>
  )
}

function View(props: {
  readonly view: BakedView<Frame>
  readonly scene: { readonly train: readonly (readonly number[])[]; readonly held: readonly number[] }
  readonly side: number
}): ReactNode {
  const { view, scene, side } = props
  const { live, t, frame, baked, run, seek } = view
  const mode = useColourMode()
  const psnr = useMemo(() => metricSeries(run, 'nerf/psnr'), [run])
  const loss = useMemo(() => metricSeries(run, 'nerf/loss'), [run])
  // The held-out peak signal-to-noise ratio at the steps whose views were prepared.
  const held = (() => {
    const seen = new Map<number, number>()
    for (let s = 0; s <= live.head; s++) {
      const f = baked.at(s)
      if (f !== undefined) seen.set(f.step, f.heldPsnr)
    }
    return [...seen.entries()].sort((a, b) => a[0] - b[0])
  })()
  const option = useMemo(() => {
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
          {
            mode,
          },
        ),
        lines([{ x: shown.map((p) => p[0]), y: shown.map((p) => p[1]), name: 'held-out views', width: 2.5 }], {
          mode,
          colourOffset: 3,
        }),
        lines(
          [{ x: [t, t], y: [Math.min(...all, 0), Math.max(...all, 1)], colour: 'muted', dash: 'dashed', width: 1 }],
          {
            mode,
          },
        ),
      ],
      { mode, figure: CURVE, x: { label: 'step', limits: [0, Math.max(1, live.head)] }, y: { label: 'PSNR (dB)' } },
    )
  }, [psnr, held, t, mode, live.head])
  const at = loss.steps.findLastIndex((st) => st <= t)
  return (
    <div className="flex flex-col gap-4">
      <div>
        <p className="mb-1 text-xs text-muted">the {scene.train.length} training views (truth)</p>
        <div
          className="grid gap-2"
          style={{ gridTemplateColumns: `repeat(${Math.min(scene.train.length, 8)}, minmax(0, 72px))` }}
        >
          {scene.train.map((rgb, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: views are positional
            <PixelCanvas key={i} width={side} height={side} rgb={rgb} caption={`view ${i}`} />
          ))}
        </div>
      </div>
      <div className="grid grid-cols-1 items-start gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.3fr)]">
        <div className="grid grid-cols-2 gap-3 md:col-span-2" style={{ maxWidth: 460 }}>
          <PixelCanvas width={side} height={side} rgb={scene.train[0]} caption="training view 0: truth" />
          <PixelCanvas
            width={side}
            height={side}
            rgb={frame?.train.rgb}
            caption={`NeRF at step ${t}${frame?.train.psnr === undefined ? '' : `: ${frame.train.psnr.toFixed(1)} dB`}`}
          />
          <PixelCanvas width={side} height={side} rgb={scene.held} caption="held-out view: truth" />
          <PixelCanvas
            width={side}
            height={side}
            rgb={frame?.held.rgb}
            caption={`NeRF at step ${t}${frame?.held.psnr === undefined ? '' : `: ${frame.held.psnr.toFixed(1)} dB`}`}
          />
        </div>
        <div className="flex flex-col gap-2">
          <Chart option={option} figure={CURVE} onInit={seek as never} />
          <dl className="grid grid-cols-3 gap-x-6 text-sm">
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
