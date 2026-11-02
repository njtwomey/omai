import { generate, type RegressionData } from 'mdml-engine/data'
import { fieldImage } from 'mdml-engine/datasets'
import {
  type FieldGrid,
  type FieldPath,
  type FieldState,
  fieldBackbones,
  fieldContract,
  fieldEncodings,
  fitFieldAlgorithm,
  neuralField,
} from 'mdml-engine/fields'
import { adam } from 'mdml-engine/optim'
import { foldIn, key } from 'mdml-engine/random'
import { choice, defaults, dict, int, real } from 'mdml-engine/space'
import { asTensor, toFlat } from 'mdml-engine/tensor'
import { type Config, defineMethod, Log, metricSeries } from 'mdml-engine/trace'
import { cartesian, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, SpaceControls, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { BakedSession } from '../../shared/BakedSession.tsx'
import { PixelCanvas } from '../../shared/PixelCanvas.tsx'

// A toy size: the picture is SIDE by SIDE pixels and the network has two hidden layers of WIDTH units.
const SIDE = 32
const WIDTH = 32
const DEPTH = 2
const FRAMES = 60
const CURVE = figure({ width: 560, height: 150 })
/** The drawn width of a picture panel (a whole multiple of the picture's side). */
const PANEL = 128

/** What the viewer chooses: the picture, the two slots of the field, and how long and how fast to train. */
const CONTROLS = dict({
  pattern: choice(['mixed', 'checker', 'stripes', 'discs'], { default: 'mixed', label: 'picture' }),
  encoding: choice(['none', 'frequency', 'fourier', 'hash'], {
    default: 'none',
    label: 'coordinate encoding',
    descriptions: {
      none: 'the raw (x, y) goes straight into the network',
      frequency: 'sines and cosines of the coordinates at doubling frequencies (as in NeRF)',
      fourier: 'sines and cosines at random frequencies',
      hash: 'a trainable multiresolution hash grid (Instant-NGP)',
    },
  }),
  backbone: choice(['mlp', 'siren'], {
    default: 'mlp',
    label: 'network',
    descriptions: { mlp: 'ReLU layers', siren: 'sine activations (SIREN)' },
  }),
  steps: int(50, 1000, { default: 300, log: true, label: 'steps' }),
  learningRate: real(1e-4, 3e-2, { default: 3e-3, log: true, label: 'learning rate' }),
})
type Controls = { pattern: string; encoding: string; backbone: string; steps: number; learningRate: number }
const START = defaults(CONTROLS) as unknown as Controls

/** What the method is configured by (the number of steps belongs to the session). */
const FIT = dict({
  pattern: choice(['mixed', 'checker', 'stripes', 'discs'], { default: 'mixed' }),
  encoding: choice(['none', 'frequency', 'fourier', 'hash'], { default: 'none' }),
  backbone: choice(['mlp', 'siren'], { default: 'mlp' }),
  learningRate: real(1e-4, 3e-2, { default: 3e-3, log: true }),
})
type FitConfig = {
  [knob: string]: string | number
  pattern: string
  encoding: string
  backbone: string
  learningRate: number
}

/** The picture, the field of the two slots and the fit's options a configuration describes. */
function problem(config: FitConfig, key: Parameters<typeof foldIn>[0]) {
  const picture = generate(fieldImage, { key: foldIn(key, 0), knobs: { side: SIDE, pattern: config.pattern } })
  const model = neuralField({
    inputDim: 2,
    outputDim: 3,
    encoding: {
      ...(defaults(fieldEncodings.space.options[config.encoding] as never) as object),
      kind: config.encoding,
    } as never,
    backbone: {
      ...(defaults(fieldBackbones.space.options[config.backbone] as never) as object),
      kind: config.backbone,
    } as never,
    width: WIDTH,
    depth: DEPTH,
    key: foldIn(key, 1),
  })
  const data = { x: picture.data.x, y: picture.data.y }
  return {
    picture,
    model,
    data,
    options: { optimizer: adam({ learningRate: config.learningRate }), decode: { resolution: SIDE } },
  }
}

/** The library's field fit (`fitFieldAlgorithm`) on the library's picture generator, with the picture chosen here. */
const fitPicture = defineMethod<FieldState, FitConfig, FieldPath>({
  name: 'fit-picture',
  config: FIT as never,
  make: (config, key) => {
    const p = problem(config, key)
    return fitFieldAlgorithm(p.model, p.data, p.options)
  },
  contract: (config, key) => {
    const p = problem(config, key)
    return fieldContract(p.model, p.data, p.options)
  },
})

export default function InrPage(): ReactNode {
  const [controls, setControls] = useState<Controls>(START)
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel title="the picture and the network" columns={5}>
        <SpaceControls space={CONTROLS} value={controls} onChange={(v) => setControls(v as unknown as Controls)} />
      </ControlPanel>
      <Training controls={controls} reset={() => setControls(START)} />
    </div>
  )
}

interface Frame {
  readonly render: FieldGrid
}

function Training(props: { readonly controls: Controls; readonly reset: () => void }): ReactNode {
  const { controls, reset } = props
  const fit = useMemo<Config>(
    () => ({
      pattern: controls.pattern,
      encoding: controls.encoding,
      backbone: controls.backbone,
      learningRate: controls.learningRate,
    }),
    [controls.pattern, controls.encoding, controls.backbone, controls.learningRate],
  )
  // The target, from the same generator and key the session's run uses (its root key folded with 7).
  const target = useMemo(() => {
    const g = problem(fit as FitConfig, foldIn(key(0), 7)).picture
    return Array.from(toFlat(asTensor((g.data as RegressionData).y)))
  }, [fit])
  return (
    <BakedSession<Frame>
      method={fitPicture}
      config={fit}
      steps={controls.steps}
      frames={FRAMES}
      logs={{ 'field/loss': Log(), 'field/psnr': Log() }}
      bake={(session, t) => ({ render: session.probe(t, 'field/render' as never) as unknown as FieldGrid })}
      onReset={reset}
    >
      {({ live, t, frame, run, seek }) => (
        <View head={live.head} run={run} t={t} frame={frame} target={target} seek={seek} />
      )}
    </BakedSession>
  )
}

function View(props: {
  readonly head: number
  readonly run: Parameters<typeof metricSeries>[0]
  readonly t: number
  readonly frame: Frame | undefined
  readonly target: readonly number[]
  readonly seek: (chart: never) => void
}): ReactNode {
  const { head, run, t, frame, target, seek } = props
  const mode = useColourMode()
  const psnr = useMemo(() => metricSeries(run, 'field/psnr'), [run])
  const loss = useMemo(() => metricSeries(run, 'field/loss'), [run])
  const option = useMemo(() => {
    const upto = psnr.steps.map((st, i) => [st, psnr.values[i] ?? Number.NaN] as const).filter(([st]) => st <= t)
    const finite = psnr.values.filter(Number.isFinite)
    return cartesian(
      [
        lines([{ x: upto.map((p) => p[0]), y: upto.map((p) => p[1]), name: 'PSNR', width: 2 }], { mode }),
        lines(
          [
            {
              x: [t, t],
              y: [Math.min(...finite, 0), Math.max(...finite, 1)],
              colour: 'muted',
              dash: 'dashed',
              width: 1,
            },
          ],
          { mode },
        ),
      ],
      { mode, figure: CURVE, x: { label: 'step', limits: [0, Math.max(1, head)] }, y: { label: 'PSNR (dB)' } },
    )
  }, [psnr, t, mode, head])
  const at = loss.steps.findLastIndex((st) => st <= t)
  return (
    // aabb / aabb / cccc: the two pictures side by side, small and centred, the curve and readouts across the width.
    <div
      className="grid w-full max-w-xl gap-x-3 gap-y-2"
      style={{ gridTemplateColumns: '1fr 1fr', gridTemplateAreas: '"a b" "c c"' }}
    >
      <div className="flex justify-center" style={{ gridArea: 'a' }}>
        <PixelCanvas width={SIDE} height={SIDE} rgb={target} size={PANEL} caption="target picture" />
      </div>
      <div className="flex justify-center" style={{ gridArea: 'b' }}>
        <PixelCanvas
          width={SIDE}
          height={SIDE}
          rgb={frame?.render.values}
          size={PANEL}
          caption={`the network's picture, step ${t}`}
        />
      </div>
      <div className="flex flex-col gap-2" style={{ gridArea: 'c' }}>
        <Chart option={option} figure={CURVE} onInit={seek as never} />
        <dl className="grid grid-cols-3 gap-x-6 text-sm">
          <div>
            <dt className="text-xs text-muted">step</dt>
            <dd className="font-medium tabular-nums">
              {t} of {head}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-muted">loss</dt>
            <dd className="font-medium tabular-nums">{(loss.values[at] ?? Number.NaN).toPrecision(3)}</dd>
          </div>
          <div>
            <dt className="text-xs text-muted">PSNR</dt>
            <dd className="font-medium tabular-nums">{(psnr.values[at] ?? Number.NaN).toFixed(1)} dB</dd>
          </div>
        </dl>
      </div>
    </div>
  )
}
