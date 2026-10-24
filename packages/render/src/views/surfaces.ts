import type { EChartsOption } from 'echarts'
import {
  type DecisionSurface,
  type HyperparameterSurface,
  type PosteriorGrid,
  surfaceAt,
} from 'whitebox-engine/exposition'
import { type Panel, subplots } from '../charts/compose.ts'
import { contours, decisionRegions } from '../charts/fields.ts'
import { band, lines } from '../charts/lines.ts'
import { heatmap } from '../charts/matrices.ts'
import { type AxisSpec, cartesian, type Layer } from '../charts/plot.ts'
import { type PointSet, points } from '../charts/points.ts'
import type { DataAspect } from '../figure/aspect.ts'
import { gridSpec } from '../figure/grid.ts'
import { formatNumber } from '../theme/format.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { outlined, short, spanOf, type ViewFrames, type ViewOptions, viewFigure, viewMode } from './_view.ts'

/** Rows of input data: `rows[i]` is row i's features. */
export type Rows = readonly (readonly number[])[]

/** Options of {@link decisionSurfaceView}. */
export interface SurfaceViewOptions extends ViewOptions {
  /** Training rows to draw over the surface instead of the record's (their first two columns, or `columns`). */
  readonly points?: Rows
  /** A label per row: a class index or a class name from the record's `classes`. */
  readonly labels?: readonly (number | string)[]
  /** Which columns of `points` are the surface's two features (default `[0, 1]`). */
  readonly columns?: readonly [number, number]
  /**
   * Contours: levels (of the probability of class 1, or of the decision function when the record has one), or `false`
   * for none (default the record's decision levels, or 0.25, 0.5, 0.75 for two classes' probabilities).
   */
  readonly contours?: readonly number[] | false
  /** The data aspect (default `auto`; `equal` when the features share units). */
  readonly aspect?: DataAspect
  /** Rows to ring (indices into the rows drawn): support vectors, the rows just seen. */
  readonly highlight?: readonly number[]
  /** A weight per row drawn as its marker's area (sample weights; residuals by their size). */
  readonly rowWeights?: readonly number[]
}

/** The class index of each label: numbers as given, names looked up in the classes (unknown names give -1). */
export function classIndices(labels: readonly (number | string)[], classes: readonly string[] | undefined): number[] {
  return labels.map((l) => (typeof l === 'number' ? l : (classes?.indexOf(l) ?? -1)))
}

/**
 * Rows coloured by class as point sets, one per class in class order (so class k takes tab10 slot k, as its region
 * does), with unlabelled rows muted.
 *
 * @param rows - The rows.
 * @param columns - The two columns drawn.
 * @param labels - A class index per row (or none).
 * @param names - Class names.
 * @param style - Style shared by every set.
 * @param sizes - A size per row (marker areas), split with the rows.
 * @returns The point sets.
 */
export function classPointSets(
  rows: Rows,
  columns: readonly [number, number],
  labels: readonly number[] | undefined,
  names: readonly string[] | undefined,
  style: Partial<PointSet> = {},
  sizes?: readonly number[],
): PointSet[] {
  const [a, b] = columns
  const x = rows.map((r) => r[a] ?? 0)
  const y = rows.map((r) => r[b] ?? 0)
  const sized = (idx: readonly number[]): Partial<PointSet> =>
    sizes === undefined ? {} : { sizes: idx.map((i) => Math.abs(sizes[i] ?? 0)), sizeRange: [2, 22] }
  if (labels === undefined)
    return [{ x, y, name: 'data', colour: 'fg', size: 6, ...style, ...sized(rows.map((_, i) => i)) }]
  const k = Math.max(-1, ...labels) + 1
  const sets: PointSet[] = []
  for (let c = -1; c < k; c++) {
    const idx = labels.flatMap((l, i) => (l === c ? [i] : []))
    if (idx.length === 0) continue
    sets.push({
      x: idx.map((i) => x[i] ?? 0),
      y: idx.map((i) => y[i] ?? 0),
      name: c < 0 ? 'unlabelled' : (names?.[c] ?? `class ${c}`),
      colour: c < 0 ? 'muted' : `tab${c % 10}`,
      size: 7,
      opacity: 1,
      symbol: ['circle', 'rect', 'triangle', 'diamond'][Math.max(c, 0) % 4] ?? 'circle',
      ...style,
      ...sized(idx),
    })
  }
  return sets
}

function oneHot(labels: readonly (readonly number[])[], k: number): number[][][] {
  return Array.from({ length: k }, (_, c) => labels.map((row) => row.map((v) => (v === c ? 1 : 0))))
}

/**
 * The rows a surface view draws: the ones given (with their columns), else the record's own rows.
 *
 * @param surface - The record.
 * @param options - Rows, labels and columns given to the view.
 * @returns The rows, the two columns drawn and the class index of each row (when known).
 */
export function surfaceRows(
  surface: DecisionSurface,
  options: Pick<SurfaceViewOptions, 'points' | 'labels' | 'columns'> = {},
): { readonly rows?: Rows; readonly columns: readonly [number, number]; readonly labels?: readonly number[] } {
  if (options.points !== undefined)
    return {
      rows: options.points,
      columns: options.columns ?? [0, 1],
      ...(options.labels === undefined ? {} : { labels: classIndices(options.labels, surface.classes) }),
    }
  const own = surface.rows
  if (own === undefined) return { columns: [0, 1] }
  const labels = options.labels === undefined ? own.labels : classIndices(options.labels, surface.classes)
  return { rows: own.points, columns: [0, 1], ...(labels === undefined ? {} : { labels }) }
}

/**
 * The layers of a decision surface: smooth class regions (faded where the model is unsure) with their boundaries and
 * probability contours, a binary decision function's regions with its margins dashed, or a smooth field with contours
 * for regressors and densities, or, for one feature, the class probabilities (or values) as curves; then the training
 * rows coloured by class (the record's own unless others are given), with the highlighted ones ringed.
 *
 * @param surface - The record.
 * @param options - Mode, training rows and labels, contours, highlighted rows and row weights.
 * @returns The layers and the axes.
 */
export function decisionSurfaceLayers(
  surface: DecisionSurface,
  options: SurfaceViewOptions = {},
): { readonly layers: Layer[]; readonly x: AxisSpec; readonly y: AxisSpec } {
  const mode: ColourMode = viewMode(options)
  const { x, y } = surface
  // Class names: the model's, else its rows' (a detector's inlier and outlier).
  const classes = surface.classes ?? surface.rows?.classes
  const k = surface.probabilities?.length ?? (surface.labels === undefined ? 0 : Math.max(...surface.labels.flat()) + 1)
  const layers: Layer[] = []
  const { rows, columns, labels } = surfaceRows(surface, options)
  const xAxis: AxisSpec = { label: surface.features[0] }
  if (y.length < 2) {
    // One feature: the probabilities (or values) as curves over it.
    const curves =
      surface.probabilities !== undefined
        ? surface.probabilities.map((p, c) => ({
            x,
            y: p[0] ?? [],
            name: classes?.[c] ?? `class ${c}`,
            colour: `tab${c % 10}`,
          }))
        : [
            {
              x,
              y: surface.values?.[0] ?? surface.labels?.[0] ?? [],
              name: surface.field?.name ?? 'prediction',
              colour: 'tab0',
              // A piecewise-constant field (a histogram's bins) is drawn flat between its points.
              ...(surface.field?.smooth === false ? { step: 'middle' as const } : {}),
            },
          ]
    layers.push(lines(curves, { mode }))
    if (rows !== undefined) {
      // Rows at their targets when the record has them (regressors), along zero otherwise.
      const targets = surface.rows?.targets
      const at = targets !== undefined && targets.length === rows.length ? targets : undefined
      const flat = rows.map((r, i) => [r[columns[0]] ?? 0, at?.[i] ?? 0])
      layers.push(points(classPointSets(flat, [0, 1], labels, classes, { size: 6, opacity: 0.5 }), { mode }))
    }
    return {
      layers,
      x: xAxis,
      y: { label: surface.probabilities !== undefined ? 'probability' : (surface.field?.name ?? 'prediction') },
    }
  }
  // A binary decision function f (an SVM's margin, a linear model's log-odds): older records kept it in `values`.
  const binary = surface.classes?.length === 2 && surface.field === undefined
  const f =
    surface.decision ?? (binary && surface.probabilities === undefined ? surface.values : undefined) ?? undefined
  const levels =
    options.contours ?? surface.levels ?? (surface.decision === undefined && f !== undefined ? [-1, 1] : undefined)
  if (f !== undefined && surface.probabilities === undefined) {
    // The regions are f's sign, drawn smoothly from the scores, with the margins (f = ±1 for an SVM) dashed.
    layers.push(decisionRegions([f.map((r) => r.map((v) => -v)), f], { x, y, mode, confidence: false, opacity: 0.5 }))
    if (levels !== undefined && levels !== false && levels.length > 0)
      layers.push(
        contours(f, { x, y, levels, colour: 'fg', dash: 'dashed', width: 1, opacity: 0.7, labels: false, mode }),
      )
  } else if (surface.probabilities !== undefined || surface.labels !== undefined) {
    const fields = surface.probabilities ?? oneHot(surface.labels ?? [], k)
    layers.push(decisionRegions(fields, { x, y, mode, confidence: surface.probabilities !== undefined, opacity: 0.5 }))
    if (f !== undefined && surface.levels !== undefined && options.contours === undefined)
      // A probabilistic SVM: its margins still come from f.
      layers.push(
        contours(f, {
          x,
          y,
          levels: surface.levels,
          colour: 'fg',
          dash: 'dashed',
          width: 1,
          opacity: 0.7,
          labels: false,
          mode,
        }),
      )
    else {
      const p = options.contours ?? (surface.probabilities !== undefined && k === 2 ? [0.25, 0.5, 0.75] : false)
      if (p !== false && surface.probabilities?.[1] !== undefined)
        layers.push(
          contours(surface.probabilities[1], {
            x,
            y,
            levels: p.filter((l) => l !== 0.5),
            colour: 'fg',
            dash: 'dashed',
            width: 0.8,
            opacity: 0.6,
            mode,
          }),
        )
    }
  } else if (surface.values !== undefined) {
    const smooth = surface.field?.smooth !== false
    // A density is coloured from zero, not from its tiniest tail value.
    const [lo, hi] = surface.field?.name === 'density' ? spanOf(surface.values.flat()) : [Number.NaN, Number.NaN]
    layers.push(
      heatmap(surface.values, {
        x,
        y,
        mode,
        colourBar: true,
        ...(lo >= 0 && hi > 0 ? { range: [0, hi] as [number, number] } : {}),
        label: surface.field?.name ?? 'prediction',
        ...(smooth ? {} : { smooth: 'cells' as const }),
      }),
    )
    if (smooth)
      layers.push(contours(surface.values, { x, y, levels: 6, colour: 'fg', width: 0.7, opacity: 0.55, mode }))
  }
  if (rows !== undefined) {
    layers.push(
      outlined(points(classPointSets(rows, columns, labels, classes, {}, options.rowWeights), { mode }), mode),
    )
    const ringed = options.highlight
    if (ringed !== undefined && ringed.length > 0)
      layers.push(
        outlined(
          points(
            [
              {
                x: ringed.map((i) => rows[i]?.[columns[0]] ?? 0),
                y: ringed.map((i) => rows[i]?.[columns[1]] ?? 0),
                name: 'highlighted',
                colour: 'none',
                size: 13,
                outline: false,
              },
            ],
            { mode },
          ),
          mode,
          1.6,
        ),
      )
  }
  return { layers, x: xAxis, y: { label: surface.features[1] } }
}

/** The axes of a surface panel: its features, limited to the grid when it spans two features. */
export function surfaceAxes(
  surface: DecisionSurface,
  axes: { readonly x: AxisSpec; readonly y: AxisSpec },
): { readonly x: AxisSpec; readonly y: AxisSpec } {
  if (surface.y.length < 2) return axes
  return {
    x: { ...axes.x, limits: [surface.x[0] ?? 0, surface.x[surface.x.length - 1] ?? 1] },
    y: { ...axes.y, limits: [surface.y[0] ?? 0, surface.y[surface.y.length - 1] ?? 1] },
  }
}

/**
 * A decision surface as one figure: the model's regions or field over two features, with the training rows (the
 * fitted model's, for a record with steps: {@link decisionSurfaceFrames} plays them).
 *
 * @param surface - The record.
 * @param options - Mode, figure, training rows and labels, contours and aspect.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { decisionSurface } from 'whitebox-engine/exposition'
 * import { decisionSurfaceView } from 'whitebox-render/views'
 * const s = decisionSurface({ features: ['a', 'b'], x: [0, 1], y: [0, 1], probabilities: [[[0.9, 0.4], [0.6, 0.1]], [[0.1, 0.6], [0.4, 0.9]]] })
 * decisionSurfaceView(s).series.length > 0 // => true
 * ```
 */
export function decisionSurfaceView(surface: DecisionSurface, options: SurfaceViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.25 })
  const { layers, x, y } = decisionSurfaceLayers(surface, { ...options, mode: fig.mode })
  const axes = surfaceAxes(surface, { x, y })
  return cartesian(layers, {
    figure: fig,
    mode: fig.mode,
    ...axes,
    aspect: options.aspect ?? 'auto',
    tooltip: 'item',
  })
}

/**
 * The training objective of a surface's steps as a curve: the record's full objective trace when it has one, else the
 * objective at each step; with the iteration of each step.
 *
 * @param surface - The record.
 * @returns The curve and its name, or `undefined` without an objective.
 */
export function surfaceObjective(
  surface: DecisionSurface,
): { readonly name: string; readonly x: readonly number[]; readonly y: readonly number[] } | undefined {
  if (surface.objective !== undefined && surface.objective.values.length > 1)
    return { name: surface.objective.name, x: surface.objective.iterations, y: surface.objective.values }
  const steps = (surface.steps ?? []).filter((s) => s.objective !== undefined)
  if (steps.length < 2) return undefined
  return { name: 'objective', x: steps.map((s) => s.iteration), y: steps.map((s) => s.objective ?? 0) }
}

/**
 * A decision surface through training as frames for a player: at each recorded step, the regions (or field) of the
 * model then, the training rows sized by their weight at that step and the ones that mattered ringed (support vectors,
 * the rows just seen), beside the training objective with the step marked. A record without steps is one frame.
 *
 * @param surface - The record.
 * @param options - As {@link decisionSurfaceView}.
 * @returns The frames (the last is the fitted model).
 */
export function decisionSurfaceFrames(surface: DecisionSurface, options: SurfaceViewOptions = {}): ViewFrames {
  const steps = surface.steps ?? []
  const objective = surfaceObjective(surface)
  const fig = viewFigure(options, { aspect: objective === undefined ? 1.25 : 1.9 })
  const mode = fig.mode
  const count = Math.max(1, steps.length)
  const stepName = surface.stepName ?? 'step'
  const option = (frame: number): EChartsOption => {
    const t = Math.min(Math.max(frame, 0), count - 1)
    const step = steps[t]
    const at = surfaceAt(surface, t)
    const { layers, x, y } = decisionSurfaceLayers(at, {
      ...options,
      mode,
      ...(options.highlight === undefined && step?.highlight !== undefined ? { highlight: step.highlight } : {}),
      ...(options.rowWeights === undefined && step?.rowWeights !== undefined ? { rowWeights: step.rowWeights } : {}),
    })
    const axes = surfaceAxes(surface, { x, y })
    const ringed = step?.highlight?.length
    const main: Panel = {
      layers,
      options: {
        ...axes,
        aspect: options.aspect ?? 'auto',
        ...(step === undefined
          ? {}
          : {
              title: `${stepName} ${step.iteration}${
                ringed === undefined ? '' : ` · ${ringed} ${surface.highlightName ?? 'highlighted'}`
              }`,
            }),
      },
    }
    if (objective === undefined || step === undefined)
      return cartesian(main.layers, { ...main.options, figure: fig, mode, tooltip: 'item' })
    const k = nearestIndexOf(objective.x, step.iteration)
    const progress: Panel = {
      layers: [
        lines([{ x: objective.x, y: objective.y, name: objective.name, colour: 'fgSecondary', width: 1.5 }], { mode }),
        points([{ x: [objective.x[k] ?? 0], y: [objective.y[k] ?? 0], name: '__now', colour: 'accent', size: 9 }], {
          mode,
        }),
      ],
      options: { x: { label: stepName }, y: { label: objective.name } },
    }
    const grid = gridSpec(1, 2, {
      widthRatios: [1.5, 1],
      wspace: 72,
      cells: [
        { id: 'main', row: 0, col: 0 },
        { id: 'objective', row: 0, col: 1 },
      ],
    })
    return subplots(grid, { main, objective: progress }, { figure: fig, legend: false, tooltip: 'item' })
  }
  return {
    count,
    label: (f) => {
      const s = steps[Math.min(Math.max(f, 0), count - 1)]
      if (s === undefined) return 'fitted'
      const last = steps[steps.length - 1]?.iteration ?? 0
      return `${stepName} ${s.iteration}/${last}${s.objective === undefined ? '' : ` · ${short(s.objective)}`}`
    },
    option,
  }
}

// The index of the entry nearest a value (the iterations of an objective trace are increasing).
function nearestIndexOf(values: readonly number[], target: number): number {
  let best = 0
  for (let i = 1; i < values.length; i++)
    if (Math.abs((values[i] ?? 0) - target) < Math.abs((values[best] ?? 0) - target)) best = i
  return best
}

/** Options of {@link posteriorView}. */
export interface PosteriorViewOptions extends ViewOptions {
  /** The input's name (default `x`). */
  readonly inputLabel?: string
  readonly outputLabel?: string
  /** Show the record's posterior samples (default true). */
  readonly samples?: boolean
}

/**
 * A predictive distribution on a grid (a GP, Bayesian regression, isotonic fits): for one input, the mean with bands of
 * one and two standard deviations, posterior samples and the observations; for two inputs, the mean as coloured
 * points with the observations.
 *
 * @param grid - The record.
 * @param options - Mode, figure, axis labels and whether to draw samples.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { posteriorGrid } from 'whitebox-engine/exposition'
 * import { posteriorView } from 'whitebox-render/views'
 * const g = posteriorGrid({ inputs: [[0], [1], [2]], mean: [0, 1, 0], variance: [1, 0.1, 1], observed: { inputs: [[1]], targets: [1] } })
 * posteriorView(g).series.some((s) => s.name === 'observed') // => true
 * ```
 */
export function posteriorView(grid: PosteriorGrid, options: PosteriorViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.6 })
  const mode = fig.mode
  const d = grid.inputs[0]?.length ?? 1
  const layers: Layer[] = []
  const xs = grid.inputs.map((r) => r[0] ?? 0)
  if (d === 1 && grid.densities !== undefined && grid.densities.length > 0) {
    // Densities over a parameter: the prior dashed, a likelihood dotted, the posterior solid over its shaded area.
    for (const dens of grid.densities) {
      if (dens.role === 'posterior')
        layers.push(
          band(
            [{ x: xs, lower: xs.map(() => 0), upper: [...dens.values], colour: 'tab0', opacity: 0.2, name: '__area' }],
            {
              mode,
            },
          ),
        )
      layers.push(
        lines(
          [
            {
              x: xs,
              y: dens.values,
              name: dens.name,
              colour: dens.role === 'posterior' ? 'tab0' : dens.role === 'prior' ? 'fgSecondary' : 'tab1',
              width: dens.role === 'posterior' ? 2.2 : 1.6,
              ...(dens.role === 'prior'
                ? { dash: 'dashed' as const }
                : dens.role === 'likelihood'
                  ? { dash: 'dotted' as const }
                  : {}),
            },
          ],
          { mode },
        ),
      )
    }
    return cartesian(layers, {
      figure: fig,
      mode,
      x: { label: options.inputLabel ?? grid.labels?.input ?? 'parameter' },
      y: { label: options.outputLabel ?? grid.labels?.output ?? 'density' },
      tooltip: 'item',
    })
  }
  if (d === 1 && grid.series !== undefined && grid.series.length > 0) {
    // Several trajectories, each with its band of two standard deviations.
    grid.series.forEach((sr, i) => {
      const sd = sr.variance.map((v) => Math.sqrt(Math.max(v, 0)))
      if (sd.some((v) => v > 0))
        layers.push(
          band(
            [
              {
                x: xs,
                lower: sr.mean.map((m, j) => m - 2 * (sd[j] ?? 0)),
                upper: sr.mean.map((m, j) => m + 2 * (sd[j] ?? 0)),
                colour: `tab${i % 10}`,
                opacity: 0.14,
                name: `__band${i}`,
              },
            ],
            { mode },
          ),
        )
      layers.push(lines([{ x: xs, y: sr.mean, name: sr.name, colour: `tab${i % 10}`, width: 1.8 }], { mode }))
    })
    return cartesian(layers, {
      figure: fig,
      mode,
      x: { label: options.inputLabel ?? grid.labels?.input ?? 'x' },
      y: { label: options.outputLabel ?? grid.labels?.output ?? 'y' },
      tooltip: 'item',
    })
  }
  if (d === 1) {
    const x = grid.inputs.map((r) => r[0] ?? 0)
    const sd = grid.variance.map((v) => Math.sqrt(Math.max(v, 0)))
    for (const k of [2, 1])
      layers.push(
        band(
          [
            {
              x,
              lower: grid.mean.map((m, i) => m - k * (sd[i] ?? 0)),
              upper: grid.mean.map((m, i) => m + k * (sd[i] ?? 0)),
              colour: 'tab0',
              opacity: k === 2 ? 0.14 : 0.24,
              name: `__band${k}`,
            },
          ],
          { mode },
        ),
      )
    if ((options.samples ?? true) && grid.samples !== undefined)
      layers.push(
        lines(
          grid.samples.map((s, i) => ({
            x,
            y: s,
            name: `__sample${i}`,
            colour: `tab${(i + 1) % 10}`,
            width: 1,
            opacity: 0.7,
          })),
          { mode },
        ),
      )
    layers.push(lines([{ x, y: grid.mean, name: 'mean', colour: 'tab0', width: 2.2 }], { mode }))
    if (grid.observed !== undefined)
      layers.push(
        outlined(
          points(
            [
              {
                x: grid.observed.inputs.map((r) => r[0] ?? 0),
                y: grid.observed.targets,
                name: 'observed',
                colour: 'fg',
                size: 7,
              },
            ],
            { mode },
          ),
          mode,
        ),
      )
    return cartesian(layers, {
      figure: fig,
      mode,
      x: { label: options.inputLabel ?? grid.labels?.input ?? 'x' },
      y: { label: options.outputLabel ?? grid.labels?.output ?? 'y' },
      legend: false,
      tooltip: 'item',
    })
  }
  layers.push(
    points(
      [
        {
          x: grid.inputs.map((r) => r[0] ?? 0),
          y: grid.inputs.map((r) => r[1] ?? 0),
          values: [...grid.mean],
          name: '__mean',
          size: 9,
          symbol: 'rect',
          opacity: 0.9,
          outline: false,
        },
      ],
      { mode },
    ),
  )
  if (grid.observed !== undefined)
    layers.push(
      outlined(
        points(
          [
            {
              x: grid.observed.inputs.map((r) => r[0] ?? 0),
              y: grid.observed.inputs.map((r) => r[1] ?? 0),
              name: 'observed',
              colour: 'fg',
              size: 6,
            },
          ],
          { mode },
        ),
        mode,
      ),
    )
  return cartesian(layers, { figure: fig, mode, x: { label: 'x0' }, y: { label: 'x1' }, legend: false })
}

/** Options of {@link hyperparameterSurfaceView}. */
export interface HyperSurfaceViewOptions extends ViewOptions {
  /** The objective's name for the colour bar (default 'objective'). */
  readonly label?: string
  /** A point to mark (a control's value), in the parameters' own units. */
  readonly selected?: readonly [number, number]
}

/**
 * An objective over two hyperparameters (a log marginal likelihood, a validation score): a smooth field with contours
 * (the colour range is the upper part of the values, so a steep fall-off does not wash the optimum out), log axes where
 * the record asks, and the optimiser's path to its end.
 *
 * @param surface - The record.
 * @param options - Mode, figure, the objective's name and a selected point.
 * @returns The ECharts option.
 */
export function hyperparameterSurfaceView(
  surface: HyperparameterSurface,
  options: HyperSurfaceViewOptions = {},
): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.35 })
  const mode = fig.mode
  const [lx, ly] = surface.logScale ?? [false, false]
  // Log axes are drawn as log10 coordinates with powers of ten as ticks, so the field stays one smooth image.
  const tx = (v: number): number => (lx ? Math.log10(Math.max(v, 1e-300)) : v)
  const ty = (v: number): number => (ly ? Math.log10(Math.max(v, 1e-300)) : v)
  const xs = surface.x.map(tx)
  const ys = surface.y.map(ty)
  const finite = surface.values
    .flat()
    .filter(Number.isFinite)
    .sort((a, b) => a - b)
  // The colours span the better three quarters of the values: the upper ones, or the lower for a loss.
  const higher = surface.higherIsBetter !== false
  const top = higher ? (finite[finite.length - 1] ?? 1) : (finite[Math.ceil(0.75 * (finite.length - 1))] ?? 1)
  const low = higher ? (finite[Math.floor(0.25 * (finite.length - 1))] ?? 0) : (finite[0] ?? 0)
  const clipped = surface.values.map((row) =>
    row.map((v) => (Number.isFinite(v) ? Math.min(Math.max(v, low), top) : Number.NaN)),
  )
  const layers: Layer[] = [
    heatmap(clipped, { x: xs, y: ys, mode, range: [low, top] }),
    contours(clipped, { x: xs, y: ys, levels: 8, colour: 'fg', width: 0.7, opacity: 0.5, labels: false, mode }),
  ]
  const path = surface.path
  if (path !== undefined && path.length > 0) {
    layers.push(
      lines(
        [
          {
            x: path.map((p) => tx(p[0])),
            y: path.map((p) => ty(p[1])),
            name: 'optimiser',
            colour: 'fg',
            width: 1.6,
            markers: true,
          },
        ],
        {
          mode,
        },
      ),
    )
    const end = path[path.length - 1] ?? [0, 0]
    layers.push(
      outlined(
        points([{ x: [tx(end[0])], y: [ty(end[1])], name: 'optimum', colour: 'accent', size: 12, symbol: 'diamond' }], {
          mode,
        }),
        mode,
        1.5,
      ),
    )
  }
  if (options.selected !== undefined)
    layers.push(
      outlined(
        points(
          [{ x: [tx(options.selected[0])], y: [ty(options.selected[1])], name: 'selected', colour: 'tab3', size: 11 }],
          { mode },
        ),
        mode,
        1.5,
      ),
    )
  const power = (v: number): string => formatNumber(10 ** v, { digits: 2 })
  return cartesian(layers, {
    figure: fig,
    mode,
    title: options.label ?? surface.objective ?? 'objective',
    x: { label: surface.parameters[0], limits: [xs[0] ?? 0, xs[xs.length - 1] ?? 1], ...(lx ? { format: power } : {}) },
    y: { label: surface.parameters[1], limits: [ys[0] ?? 0, ys[ys.length - 1] ?? 1], ...(ly ? { format: power } : {}) },
    legend: false,
    tooltip: 'item',
  })
}
