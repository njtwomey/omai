import { additiveRegressor, gam, s as splineOf } from 'omai-engine/gam'
import { netTerm, splineTerm } from 'omai-engine/nn'
import { choice, defaults, dict, int, real } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import { metricSeries } from 'omai-engine/trace'
import { cartesian, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import {
  Chart,
  ControlPanel,
  DatasetPicker,
  type DatasetSelection,
  type LiveSession,
  SessionPlayer,
  SpaceControls,
  selectEntry,
  useColourMode,
  useDataset,
  useProbe,
  useRun,
  useScrubClick,
  useSession,
  useSessionPreferences,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { type CurveData, curveData, gridOf } from '../../shared/curve-data.ts'
import { StaleVeil } from '../../shared/StaleVeil.tsx'

const DATA = 'regression1d'
// Points at which every curve is evaluated: enough that steps and kinks are drawn as such.
const GRID = 301
const FIRST = (): DatasetSelection => selectEntry(DATA, 'sineCurve')
const WIDE = figure({ width: 640, height: 380 })
const NARROW = figure({ width: 420, height: 380 })
const FG = '#e6edf3'
const ORANGE = '#e8743b'

const SPACE = dict({
  term: choice(['spline', 'net'], {
    default: 'spline',
    label: 'the term is',
    descriptions: {
      spline: 'a learnable B-spline: the coefficients (and, with learnable knots, the knot positions) are trained',
      net: 'a small neural net of the feature (a neural additive model term): two hidden layers of 32 units',
    },
  }),
  degree: int(0, 6, { default: 3, label: 'spline degree (0 constant, 1 linear, 3 cubic) (splines)' }),
  lambda: real(0.001, 1000, { default: 5, log: true, label: 'smoothness penalty λ (splines)' }),
  knots: choice(['uniform', 'quantile', 'learnable'], {
    default: 'uniform',
    label: 'knot placement (splines)',
    descriptions: {
      uniform: 'knots spread evenly over the range of the feature',
      quantile: 'knots at quantiles of the feature, so each span holds about the same number of rows',
      learnable: 'knot positions are parameters, trained along with the coefficients (they stay ordered)',
    },
  }),
  shape: choice(['none', 'increasing', 'decreasing', 'convex', 'concave', 'bounded'], {
    default: 'none',
    label: 'shape constraint (splines)',
    descriptions: {
      none: 'no constraint',
      increasing: 'the coefficients are built to rise, so the curve cannot fall (exact at every step)',
      decreasing: 'the coefficients are built to fall, so the curve cannot rise',
      convex: 'the second differences of the coefficients are kept non-negative: the curve cannot bend downwards',
      concave: 'the second differences of the coefficients are kept non-positive: the curve cannot bend upwards',
      bounded: 'the coefficients are squashed into the range of the data, so the curve cannot leave it',
    },
  }),
  interiorKnots: int(2, 30, { default: 8, label: 'interior knots (splines)' }),
  epochs: int(20, 600, { default: 200, log: true, label: 'steps' }),
  learningRate: real(0.005, 0.3, { default: 0.05, log: true, label: 'learning rate' }),
})
type Controls = {
  term: string
  shape: string
  degree: number
  lambda: number
  knots: string
  interiorKnots: number
  epochs: number
  learningRate: number
}
const START: Controls = {
  term: 'spline',
  shape: 'none',
  degree: 3,
  lambda: 5,
  knots: 'uniform',
  interiorKnots: 8,
  epochs: 200,
  learningRate: 0.05,
}

export default function GamNeuralPage(): ReactNode {
  const [selection, setSelection] = useState<DatasetSelection>(FIRST)
  const [controls, setControls] = useState<Controls>(START)
  const generated = useDataset(DATA, selection)
  const curve = useMemo(() => curveData(generated.data, generated.truth as never), [generated.data, generated.truth])
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 items-start gap-3 md:grid-cols-2">
        <ControlPanel title="data" columns={1}>
          <DatasetPicker collection={DATA} value={selection} onChange={setSelection} />
        </ControlPanel>
        <ControlPanel title="the network and its penalty" columns={1}>
          <SpaceControls space={SPACE} value={controls} onChange={(v) => setControls(v as unknown as Controls)} />
        </ControlPanel>
      </div>
      {curve === undefined ? (
        <p className="text-sm text-muted">This page needs a one-dimensional regression data set.</p>
      ) : (
        <Training curve={curve} controls={controls} reset={() => setControls(START)} />
      )}
    </div>
  )
}

function Training(props: {
  readonly curve: CurveData
  readonly controls: Controls
  readonly reset: () => void
}): ReactNode {
  const { curve, controls, reset } = props
  const prefs = useSessionPreferences()
  const [at, setAt] = useState<number | null>(null)
  const method = useMemo(() => {
    const model = additiveRegressor.with({
      lambda: controls.lambda,
      knots: { kind: controls.knots },
      interiorKnots: controls.interiorKnots,
      gridSize: GRID,
      degree: controls.degree,
      epochs: controls.epochs,
      learningRate: controls.learningRate,
      ...(controls.term === 'net'
        ? { terms: [netTerm(0, { width: 32, depth: 2 })] }
        : controls.shape === 'none'
          ? {}
          : {
              terms: [
                splineTerm(0, {
                  interiorKnots: controls.interiorKnots,
                  degree: controls.degree,
                  knots: { kind: controls.knots },
                  shape: { kind: controls.shape },
                  lambda: controls.lambda,
                }),
              ],
            }),
    })
    return model.method({ x: curve.x.map((v) => [v]), y: [...curve.y] })
  }, [curve, controls])
  const live = useSession(
    method,
    { config: defaults(method.config) as Config, mode: prefs.mode, steps: controls.epochs },
    [method],
    { hold: prefs.hold },
  )
  // The classic fit of the same smooth, for comparison: REML chooses its own smoothing.
  const classic = useMemo(() => {
    try {
      const fitted = gam
        .with({ terms: [splineOf(0, { k: controls.interiorKnots + 4 })] })
        .fit({ x: curve.x.map((v) => [v]), y: [...curve.y] })
      const grid = gridOf(curve.lo, curve.hi, GRID)
      return { grid, values: Array.from(fitted.predictTerms(grid.map((v) => [v])).data as Float64Array) }
    } catch {
      return undefined
    }
  }, [curve, controls.interiorKnots])
  return (
    <div className="flex flex-col gap-3">
      <StaleVeil stale={live.stale}>
        <View live={live} at={at} onScrub={setAt} curve={curve} classic={classic} />
      </StaleVeil>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={reset} />
    </div>
  )
}

function View(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the session is generic over the method
  readonly live: LiveSession<any, any>
  readonly at: number | null
  readonly onScrub: (t: number | null) => void
  readonly curve: CurveData
  readonly classic: { readonly grid: number[]; readonly values: number[] } | undefined
}): ReactNode {
  const { live, curve, classic } = props
  const mode = useColourMode()
  const run = useRun(live)
  const t = Math.max(0, Math.min(props.at ?? live.head, live.head))
  const shape = useProbe(live, t, 'predictor/0/term/0/curve')
  const knotsProbe = useProbe(live, t, 'additive/knots')
  const basisProbe = useProbe(live, t, 'predictor/0/term/0/basis')
  const loss = useMemo(() => metricSeries(run, 'train/loss'), [run])
  const steps = useMemo(() => Array.from({ length: live.head + 1 }, (_, i) => i), [live.head])
  const seek = useScrubClick(steps, live.head, props.onScrub)
  const meanY = useMemo(() => curve.y.reduce((a, b) => a + b, 0) / curve.y.length, [curve])
  const pieces = (basisProbe.value as { grid?: number[]; functions?: number[][] } | null | undefined)?.functions
  const piecesGrid = (basisProbe.value as { grid?: number[] } | null | undefined)?.grid
  const shown = shape.value as { grid?: number[]; values?: number[] } | null | undefined
  const knots = useMemo(() => {
    const k = knotsProbe.value as { terms?: { knots?: number[] }[] }[] | undefined
    return Array.from(k?.[0]?.terms?.[0]?.knots ?? [])
  }, [knotsProbe.value])
  const fitted = useMemo(() => {
    const grid = shown?.grid
    const values = shown?.values
    const yLo = Math.min(...curve.y) - meanY
    const yHi = Math.max(...curve.y) - meanY
    const truthGrid = curve.truthAt?.(gridOf(curve.lo, curve.hi, GRID))
    const truthAtData = curve.truthAt?.([...curve.x])
    const truthMean = truthAtData === undefined ? 0 : truthAtData.reduce((a, b) => a + b, 0) / truthAtData.length
    return cartesian(
      [
        points([{ x: [...curve.x], y: curve.y.map((v) => v - meanY), name: 'data', opacity: 0.5 }], {
          mode,
          colourOffset: 7,
        }),
        ...(truthGrid === undefined
          ? []
          : [
              lines(
                [
                  {
                    x: gridOf(curve.lo, curve.hi, GRID),
                    y: truthGrid.map((v) => v - truthMean),
                    name: 'truth',
                    dash: 'dashed',
                    width: 2,
                    colour: 'muted',
                  },
                ],
                { mode },
              ),
            ]),
        // The basis functions times their coefficients, shifted so that they add up to the centred curve: a light dotted trace.
        ...(pieces === undefined || piecesGrid === undefined
          ? []
          : [
              lines(
                pieces.map((f, k) => ({
                  x: piecesGrid,
                  y: f,
                  name: k === 0 ? 'weighted basis' : '',
                  dash: 'dotted' as const,
                  width: 1.2,
                  opacity: 0.45,
                  colour: ORANGE,
                })),
                { mode },
              ),
            ]),
        ...(classic === undefined
          ? []
          : [
              lines(
                [
                  {
                    x: classic.grid,
                    y: classic.values,
                    name: 'classic GAM (REML)',
                    dash: 'dotted',
                    width: 2,
                    colour: FG,
                  },
                ],
                { mode },
              ),
            ]),
        ...(grid === undefined || values === undefined
          ? []
          : [lines([{ x: grid, y: values, name: 'network', width: 3, colour: ORANGE }], { mode })]),
        ...(knots.length === 0
          ? []
          : [points([{ x: knots, y: knots.map(() => yLo), name: 'knots', colour: ORANGE }], { mode })]),
      ],
      {
        mode,
        figure: WIDE,
        x: { label: 'x', limits: [curve.lo, curve.hi] },
        y: { label: 'effect', limits: [yLo - 0.1 * (yHi - yLo), yHi + 0.1 * (yHi - yLo)] },
      },
    )
  }, [shown, knots, pieces, piecesGrid, curve, classic, meanY, mode])
  const lossOption = useMemo(() => {
    const upto = loss.steps.map((st, i) => [st, loss.values[i] ?? Number.NaN] as const).filter(([st]) => st <= t)
    return cartesian(
      [
        lines([{ x: upto.map((p) => p[0]), y: upto.map((p) => p[1]), name: 'training loss', width: 2 }], { mode }),
        lines(
          [
            {
              x: [t, t],
              y: [Math.min(...loss.values), Math.max(...loss.values)],
              colour: FG,
              dash: 'dashed',
              width: 1,
            },
          ],
          { mode },
        ),
      ],
      {
        mode,
        figure: NARROW,
        x: { label: 'step', limits: [0, Math.max(1, live.head)] },
        y: { label: 'loss', scale: 'log' },
      },
    )
  }, [loss, t, mode, live.head])
  return (
    <div className="grid grid-cols-1 gap-x-4 gap-y-2 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <Chart option={fitted} figure={WIDE} />
      <Chart option={lossOption} figure={NARROW} onInit={seek} />
    </div>
  )
}
