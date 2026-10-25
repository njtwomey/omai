import type { EChartsOption, SeriesOption } from 'echarts'
import { multivariateNormal } from 'whitebox-engine/dists'
import type { ClusteringSteps, EmbeddingSteps, InputRows, MixtureSteps } from 'whitebox-engine/exposition'
import { covarianceEllipse } from 'whitebox-engine/numerics/geometry'
import { toArray, toFlat } from 'whitebox-engine/tensor'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { decisionRegions } from '../charts/fields.ts'
import { referenceLines } from '../charts/functions.ts'
import { type Curve, lines } from '../charts/lines.ts'
import { cartesian, type Layer, layer } from '../charts/plot.ts'
import { type PointSet, points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { sampleRamp, sequentialRamp } from '../theme/palettes.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { outlined, short, spanOf, type ViewFrames, type ViewOptions, viewFigure, viewMode } from './_view.ts'
import type { Rows } from './surfaces.ts'

/** Options of the step views: the input rows the steps assign (their first two columns are drawn). */
export interface StepsViewOptions extends ViewOptions {
  readonly points?: Rows
  /** Which columns of `points` to draw (default `[0, 1]`). */
  readonly columns?: readonly [number, number]
}

const clusterColour = (c: number): string => (c < 0 ? 'muted' : `tab${c % 10}`)

/** The rows a step view draws: the ones given (with their columns), else the record's own (two coordinates). */
function rowsOf(
  record: { readonly rows?: InputRows },
  options: StepsViewOptions,
): { readonly rows?: Rows; readonly columns: readonly [number, number]; readonly names: readonly [string, string] } {
  if (options.points !== undefined) {
    const columns = options.columns ?? [0, 1]
    return { rows: options.points, columns, names: [`x${columns[0]}`, `x${columns[1]}`] }
  }
  const own = record.rows
  return own === undefined
    ? { columns: [0, 1], names: ['x0', 'x1'] }
    : { rows: own.points, columns: [0, 1], names: own.features }
}

/**
 * A square grid (equal spans, for an equal-aspect panel) around rows' two drawn columns, padded by 5%, `n` points a
 * side.
 */
function gridOver(rows: Rows, columns: readonly [number, number], n: number): { x: number[]; y: number[] } {
  const [lx, hx] = spanOf(rows.map((r) => r[columns[0]] ?? 0))
  const [ly, hy] = spanOf(rows.map((r) => r[columns[1]] ?? 0))
  const span = Math.max(hx - lx, hy - ly) || 1
  const px = 0.05 * span + (span - (hx - lx)) / 2
  const py = 0.05 * span + (span - (hy - ly)) / 2
  const axis = (lo: number, hi: number): number[] => Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
  return { x: axis(lx - px, hx + px), y: axis(ly - py, hy + py) }
}

/**
 * The regions of a clustering over the drawn plane: each grid point takes the colour of the highest-scoring cluster
 * (the nearest centre, the most responsible component), drawn faintly under the rows.
 */
function regionLayer(
  grid: { readonly x: readonly number[]; readonly y: readonly number[] },
  scores: readonly (readonly (readonly number[])[])[],
  mode: ColourMode,
): Layer {
  return decisionRegions(scores, {
    x: grid.x,
    y: grid.y,
    mode,
    confidence: false,
    opacity: 0.22,
    colours: scores.map((_, c) => clusterColour(c)),
  })
}

/** Rows grouped by cluster as point sets with stable colours (cluster k is tab10 slot k; noise is muted). */
function clusterSets(
  rows: Rows,
  columns: readonly [number, number],
  assignment: readonly number[],
  style: (c: number) => Partial<PointSet> = () => ({}),
): PointSet[] {
  const [a, b] = columns
  const groups = new Map<number, number[]>()
  for (const [i, c] of assignment.entries()) groups.set(c, [...(groups.get(c) ?? []), i])
  return [...groups.entries()]
    .sort((p, q) => p[0] - q[0])
    .map(([c, idx]) => ({
      x: idx.map((i) => rows[i]?.[a] ?? 0),
      y: idx.map((i) => rows[i]?.[b] ?? 0),
      name: c < 0 ? 'noise' : `cluster ${c}`,
      colour: clusterColour(c),
      size: 7,
      ...style(c),
    }))
}

// A curve over steps with the current step marked.
function progressPanel(
  values: readonly number[],
  at: number,
  label: string,
  mode: ColourMode,
  xs?: readonly number[],
): Panel {
  const x = xs ?? values.map((_, i) => i)
  return {
    layers: [
      lines([{ x, y: values, name: label, colour: 'fgSecondary', width: 1.5 }], { mode }),
      points([{ x: [x[at] ?? at], y: [values[at] ?? 0], name: `__now`, colour: 'accent', size: 9 }], { mode }),
    ],
    options: { x: { label: xs === undefined ? 'step' : 'iteration' }, y: { label } },
  }
}

/**
 * The steps of a clustering as frames for a player: the rows coloured by their cluster after each iteration (noise
 * muted; core, border and noise rows sized by role for density methods), the centres with the trails they moved
 * along (centroid methods), and the objective with the current step marked. Without the input rows, the cluster sizes
 * per step stand in for the scatter.
 *
 * @param steps - The record.
 * @param options - Mode, figure, input rows and columns.
 * @returns The frames.
 */
export function clusteringFrames(steps: ClusteringSteps, options: StepsViewOptions = {}): ViewFrames {
  const mode = viewMode(options)
  const count = steps.assignments.length
  const { rows: drawn, columns, names } = rowsOf(steps, options)
  const objective = steps.objective
  // The side curve only when there is something to follow: an objective over several steps, or rows being assigned.
  const assigned = steps.assignments.map((a) => a.filter((c) => c >= 0).length)
  const progress =
    objective !== undefined && objective.length > 1
      ? objective
      : count > 1 && new Set(assigned).size > 1
        ? assigned
        : undefined
  const progressLabel = progress === objective ? 'objective' : 'rows assigned'
  const fig = viewFigure(options, { aspect: progress === undefined ? 1.25 : 1.75 })
  const grid0 = drawn === undefined ? undefined : gridOver(drawn, columns, 60)
  const option = (frame: number): EChartsOption => {
    const t = Math.min(Math.max(frame, 0), count - 1)
    const assignment = steps.assignments[t] ?? []
    const main: Layer[] = []
    if (drawn !== undefined) {
      const centresNow = steps.centres?.[t]
      // Centroid methods in two features: the nearest-centre regions under the rows.
      if (grid0 !== undefined && centresNow !== undefined && centresNow.length > 1 && centresNow[0]?.length === 2)
        main.push(
          regionLayer(
            grid0,
            centresNow.map((m) =>
              grid0.y.map((b) =>
                grid0.x.map((a) => -(((m[columns[0]] ?? 0) - a) ** 2) - ((m[columns[1]] ?? 0) - b) ** 2),
              ),
            ),
            mode,
          ),
        )
      const roles = steps.roles
      if (roles === undefined) main.push(points(clusterSets(drawn, columns, assignment), { mode }))
      else {
        // Sets per cluster and role, so core rows stand out from border rows and noise is a cross.
        const sets: PointSet[] = []
        for (const role of ['core', 'border', 'noise'] as const) {
          const keep = assignment.map((c, i) => (roles[i] === role ? c : Number.NaN))
          const idx = keep.flatMap((c, i) => (Number.isNaN(c) ? [] : [i]))
          const rows = idx.map((i) => drawn[i] ?? [])
          const sub = idx.map((i) => assignment[i] ?? -1)
          sets.push(
            ...clusterSets(rows, columns, sub, () =>
              role === 'core'
                ? { size: 8 }
                : role === 'border'
                  ? { size: 5, opacity: 0.8 }
                  : { size: 6, symbol: 'path://M0,0L10,10M10,0L0,10', opacity: 0.8 },
            ).map((s) => ({ ...s, name: role === 'noise' ? 'noise' : `${s.name} (${role})` })),
          )
        }
        main.push(points(sets, { mode }))
      }
      // Constraints (COP k-means): must-link pairs joined by solid lines, cannot-link pairs by dashed ones.
      const pairLines = (
        pairs: readonly (readonly [number, number])[] | undefined,
        dash: boolean,
      ): Curve | undefined =>
        pairs === undefined || pairs.length === 0
          ? undefined
          : {
              x: pairs.flatMap(([i, j]) => [drawn[i]?.[columns[0]] ?? 0, drawn[j]?.[columns[0]] ?? 0, Number.NaN]),
              y: pairs.flatMap(([i, j]) => [drawn[i]?.[columns[1]] ?? 0, drawn[j]?.[columns[1]] ?? 0, Number.NaN]),
              name: dash ? 'cannot link' : 'must link',
              colour: dash ? 'danger' : 'fg',
              width: 1.2,
              ...(dash ? { dash: 'dashed' as const } : {}),
            }
      // The pairs the final assignment breaks, on the last step, thick in the accent colour.
      const c = steps.constraints
      const last = t === count - 1
      const broken = (pairs: readonly (readonly [number, number])[] | undefined, at: readonly number[] | undefined) =>
        last && pairs !== undefined && at !== undefined
          ? at.flatMap((k) => (pairs[k] === undefined ? [] : [pairs[k]]))
          : []
      const brokenPairs = [...broken(c?.mustLink, c?.brokenMustLink), ...broken(c?.cannotLink, c?.brokenCannotLink)]
      const links = [
        pairLines(c?.mustLink, false),
        pairLines(c?.cannotLink, true),
        ...(brokenPairs.length === 0
          ? []
          : [{ ...(pairLines(brokenPairs, false) as Curve), name: 'broken', colour: 'accent', width: 2.6 }]),
      ].filter((l): l is Curve => l !== undefined)
      if (links.length > 0) main.push(lines(links, { mode }))
      // Rows newly labelled at this step (queried, pseudo-labelled): ringed in the accent colour.
      const added = steps.added?.[t]
      if (added !== undefined && added.length > 0)
        main.push(
          outlined(
            points(
              [
                {
                  x: added.map((i) => drawn[i]?.[columns[0]] ?? 0),
                  y: added.map((i) => drawn[i]?.[columns[1]] ?? 0),
                  name: steps.addedName ?? 'added',
                  colour: 'accent',
                  size: 11,
                  symbol: 'diamond',
                },
              ],
              { mode },
            ),
            mode,
            1.4,
          ),
        )
      // Seeds (semi-supervised): the rows whose labels were given, ringed.
      const seeds = steps.seeds
      if (seeds !== undefined && seeds.length > 0)
        main.push(
          outlined(
            points(
              [
                {
                  x: seeds.map((i) => drawn[i]?.[columns[0]] ?? 0),
                  y: seeds.map((i) => drawn[i]?.[columns[1]] ?? 0),
                  name: 'labelled',
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
      const centres = steps.centres
      if (centres !== undefined) {
        const k = centres[t]?.length ?? 0
        const trails: Curve[] = Array.from({ length: k }, (_, c) => ({
          x: centres.slice(0, t + 1).map((cs) => cs[c]?.[columns[0]] ?? Number.NaN),
          y: centres.slice(0, t + 1).map((cs) => cs[c]?.[columns[1]] ?? Number.NaN),
          name: `__trail${c}`,
          colour: clusterColour(c),
          width: 1.5,
          dash: 'dotted',
        }))
        main.push(lines(trails, { mode }))
        main.push(
          outlined(
            points(
              (centres[t] ?? []).map((m, c) => ({
                x: [m[columns[0]] ?? 0],
                y: [m[columns[1]] ?? 0],
                name: `centre ${c}`,
                colour: clusterColour(c),
                size: 16,
                symbol: 'diamond',
              })),
              { mode },
            ),
            mode,
            1.5,
          ),
        )
      }
    } else {
      const k = Math.max(0, ...assignment) + 1
      const sizes = Array.from({ length: k }, (_, c) => assignment.filter((a) => a === c).length)
      main.push(
        bars(
          sizes.map((_, c) => `${c}`),
          [{ name: 'rows', values: sizes }],
          { mode },
        ),
      )
    }
    const scatter: Panel = {
      layers: main,
      options:
        drawn === undefined
          ? { x: { label: 'cluster' }, y: { label: 'rows' } }
          : {
              x: { label: names[0], ...(grid0 === undefined ? {} : { limits: limitsOf2(grid0.x) }) },
              y: { label: names[1], ...(grid0 === undefined ? {} : { limits: limitsOf2(grid0.y) }) },
              // Regions fill their grid: equal scaling would widen an axis past it and leave unshaded bands.
              aspect: grid0 === undefined ? 'equal' : 'auto',
            },
    }
    if (progress === undefined || count < 2)
      return cartesian(scatter.layers, { ...scatter.options, figure: fig, mode, legend: false })
    const grid = gridSpec(1, 2, {
      widthRatios: [1.7, 1],
      wspace: 64,
      cells: [
        { id: 'main', row: 0, col: 0 },
        { id: 'progress', row: 0, col: 1 },
      ],
    })
    return subplots(
      grid,
      { main: scatter, progress: progressPanel(progress, t, progressLabel, mode) },
      { figure: fig, legend: false },
    )
  }
  return {
    count,
    label: (f) => `step ${f}/${count - 1}${objective === undefined ? '' : ` · ${short(objective[f] ?? 0)}`}`,
    option,
  }
}

// The first and last of an increasing axis.
const limitsOf2 = (v: readonly number[]): [number, number] => [v[0] ?? 0, v[v.length - 1] ?? 1]

/** A dendrogram's drawing: each merge as a U (x along the leaf order, y the merge height) and the leaf order. */
export interface DendrogramLayout {
  /** order[k]: the row at position k along the axis. */
  readonly order: readonly number[]
  /** links[m]: merge m's U as four points (left foot, left top, right top, right foot). */
  readonly links: readonly (readonly (readonly [number, number])[])[]
}

/**
 * Lay out a dendrogram from scipy-style linkage rows `[a, b, height, size]` (cluster ids n, n + 1, ... are the merges
 * in order): leaves in the order a depth-first walk from the root meets them, merges at the mean of their children.
 *
 * @param merges - The linkage rows.
 * @param n - The number of rows clustered.
 * @returns The leaf order and the links.
 * @example
 * ```ts
 * import { dendrogramLayout } from 'whitebox-render/views'
 * dendrogramLayout([[0, 1, 1, 2], [2, 3, 2, 3]], 3).order // => [2, 0, 1]
 * ```
 */
export function dendrogramLayout(
  merges: readonly (readonly [number, number, number, number])[],
  n: number,
): DendrogramLayout {
  const order: number[] = []
  const root = n + merges.length - 1
  const stack = [root]
  while (stack.length > 0) {
    const id = stack.pop() as number
    if (id < n) order.push(id)
    else {
      const m = merges[id - n]
      if (m !== undefined) stack.push(m[1], m[0])
    }
  }
  const x = new Map<number, number>(order.map((leaf, k) => [leaf, k]))
  const y = new Map<number, number>()
  const links = merges.map((m, i) => {
    const [a, b, h] = m
    const xa = x.get(a) ?? 0
    const xb = x.get(b) ?? 0
    x.set(n + i, (xa + xb) / 2)
    y.set(n + i, h)
    return [
      [xa, y.get(a) ?? 0],
      [xa, h],
      [xb, h],
      [xb, y.get(b) ?? 0],
    ] as const
  })
  return { order, links }
}

/**
 * The flat clusters of a linkage cut into k clusters: a cluster index per row, numbered in leaf order.
 *
 * @param merges - The linkage rows.
 * @param n - The number of rows.
 * @param k - The number of clusters (1 to n).
 * @returns The cluster of each row.
 */
export function cutClusters(
  merges: readonly (readonly [number, number, number, number])[],
  n: number,
  k: number,
): number[] {
  const parent = Array.from({ length: n + merges.length }, (_, i) => i)
  const find = (i: number): number => {
    let r = i
    while (parent[r] !== r) r = parent[r] ?? r
    return r
  }
  merges.slice(0, Math.max(0, n - Math.max(1, Math.min(n, k)))).forEach((m, i) => {
    parent[find(m[0])] = n + i
    parent[find(m[1])] = n + i
  })
  const { order } = dendrogramLayout(merges, n)
  const index = new Map<number, number>()
  const out = new Array<number>(n).fill(0)
  for (const leaf of order) {
    const r = find(leaf)
    if (!index.has(r)) index.set(r, index.size)
    out[leaf] = index.get(r) ?? 0
  }
  return out
}

/** Options of {@link dendrogramView}. */
export interface DendrogramViewOptions extends StepsViewOptions {
  /** Cut into this many clusters (default the record's final number of clusters). */
  readonly clusters?: number
}

/**
 * A hierarchical clustering's merges as a dendrogram, cut into clusters (links below the cut coloured by cluster, the
 * cut dashed), beside the rows coloured by the same clusters when the input rows are given.
 *
 * @param steps - A record with `merges`.
 * @param options - Mode, figure, the number of clusters, input rows and columns.
 * @returns The ECharts option.
 */
export function dendrogramView(steps: ClusteringSteps, options: DendrogramViewOptions = {}): EChartsOption {
  const { rows: drawn, columns, names } = rowsOf(steps, options)
  const fig = viewFigure(options, { aspect: drawn === undefined ? 1.6 : 2 })
  const mode = fig.mode
  const merges = steps.merges ?? []
  const n = merges.length + 1
  const final = steps.assignments[steps.assignments.length - 1] ?? []
  const k = Math.max(1, Math.min(n, options.clusters ?? Math.max(0, ...final) + 1))
  const { links } = dendrogramLayout(merges, n)
  const labels = cutClusters(merges, n, k)
  const cut = k > 1 && k <= n - 1 ? ((merges[n - k - 1]?.[2] ?? 0) + (merges[n - k]?.[2] ?? 0)) / 2 : undefined
  // Each link takes its cluster's colour when it lies below the cut, the muted colour above it.
  const groups = new Map<number, (readonly (readonly [number, number])[])[]>()
  const leafOf = (id: number): number => {
    let r = id
    while (r >= n) r = merges[r - n]?.[0] ?? 0
    return r
  }
  links.forEach((u, i) => {
    const below = n - 1 - i >= k
    const c = below ? (labels[leafOf(n + i)] ?? 0) : -1
    groups.set(c, [...(groups.get(c) ?? []), u])
  })
  const curves: Curve[] = [...groups.entries()].map(([c, us]) => ({
    x: us.flatMap((u) => [...u.map((p) => p[0]), Number.NaN]),
    y: us.flatMap((u) => [...u.map((p) => p[1]), Number.NaN]),
    name: c < 0 ? '__above' : `cluster ${c}`,
    colour: clusterColour(c),
    width: 1.3,
  }))
  const tree: Layer[] = [lines(curves, { mode })]
  if (cut !== undefined) tree.push(referenceLines({ y: [cut] }, { mode }))
  const dendro: Panel = {
    layers: tree,
    options: {
      title: `cut into ${k} cluster${k === 1 ? '' : 's'}`,
      x: { label: 'rows (leaf order)', show: false, limits: [-0.5, n - 0.5] },
      y: { label: 'merge height' },
    },
  }
  if (drawn === undefined) return cartesian(dendro.layers, { ...dendro.options, figure: fig, mode, legend: false })
  const scatter: Panel = {
    layers: [points(clusterSets(drawn, columns, labels), { mode })],
    options: { aspect: 'equal', x: { label: names[0] }, y: { label: names[1] } },
  }
  const grid = gridSpec(1, 2, {
    widthRatios: [1.3, 1],
    wspace: 48,
    cells: [
      { id: 'dendrogram', row: 0, col: 0 },
      { id: 'rows', row: 0, col: 1 },
    ],
  })
  return subplots(grid, { dendrogram: dendro, rows: scatter }, { figure: fig, legend: false })
}

/** Options of {@link reachabilityView}. */
export interface ReachabilityViewOptions extends ViewOptions {
  /** A reachability threshold to mark (DBSCAN-like extraction at that radius). */
  readonly threshold?: number
}

/**
 * The OPTICS reachability plot: each row's reachability distance in the cluster ordering, coloured by its final
 * cluster (noise muted), so clusters are valleys.
 *
 * @param steps - A record with `reachability` and `ordering`.
 * @param options - Mode, figure and a threshold.
 * @returns The ECharts option.
 */
export function reachabilityView(steps: ClusteringSteps, options: ReachabilityViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 2.2 })
  const mode = fig.mode
  const reach = steps.reachability ?? []
  const order = steps.ordering ?? reach.map((_, i) => i)
  const final = steps.assignments[steps.assignments.length - 1] ?? []
  // Undefined reachability (the first row of each component) is infinite, or -1 as some records store it: a full bar.
  const defined = (r: number): boolean => Number.isFinite(r) && r >= 0
  const finite = reach.filter(defined)
  const top = finite.length > 0 ? Math.max(...finite) * 1.08 : 1
  const values = reach.map((r) => (defined(r) ? r : top))
  const clusters = [...new Set(order.map((i) => final[i] ?? -1))].sort((a, b) => a - b)
  const groups = clusters.map((c) => ({
    name: c < 0 ? 'noise' : `cluster ${c}`,
    colour: clusterColour(c),
    values: values.map((v, k) => ((final[order[k] ?? 0] ?? -1) === c ? v : 0)),
  }))
  const layers: Layer[] = [
    bars(
      order.map((_, k) => `${k}`),
      groups,
      { mode, stacked: true },
    ),
  ]
  if (options.threshold !== undefined)
    layers.push(
      referenceLines({ y: [{ value: options.threshold, label: `ε = ${short(options.threshold)}` }] }, { mode }),
    )
  return cartesian(layers, {
    figure: fig,
    mode,
    x: { label: 'cluster order', show: false },
    y: { label: 'reachability distance' },
    legend: false,
    tooltip: 'item',
  })
}

function covarianceBlock(flat: readonly number[], d: number, columns: readonly [number, number]): number[][] {
  const [a, b] = columns
  if (flat.length === d * d)
    return [
      [flat[a * d + a] ?? 1, flat[a * d + b] ?? 0],
      [flat[b * d + a] ?? 0, flat[b * d + b] ?? 1],
    ]
  if (flat.length === d)
    return [
      [flat[a] ?? 1, 0],
      [0, flat[b] ?? 1],
    ]
  const v = flat[0] ?? 1
  return [
    [v, 0],
    [0, v],
  ]
}

function fullCovariance(flat: readonly number[], d: number): number[][] {
  if (flat.length === d * d) return Array.from({ length: d }, (_, i) => flat.slice(i * d, (i + 1) * d))
  return Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => (i === j ? (flat.length === d ? (flat[i] ?? 1) : (flat[0] ?? 1)) : 0)),
  )
}

/**
 * The weighted log density of each component at each row, under the mixture after step t: over all d features when
 * the rows have them, else over the first two (the rows a record carries), from the components' 2-D marginals.
 *
 * @param steps - The record.
 * @param t - The step.
 * @param rows - The rows.
 * @returns scores[k][i], or `undefined` without covariances.
 */
export function mixtureScores(steps: MixtureSteps, t: number, rows: Rows): number[][] | undefined {
  const means = steps.means[t] ?? []
  const covs = steps.covariances?.[t]
  const weights = steps.weights[t] ?? []
  const d = means[0]?.length ?? 0
  if (covs === undefined || d === 0 || rows.length === 0) return undefined
  const full = (rows[0]?.length ?? 0) >= d
  const x = rows.map((r) => (full ? r.slice(0, d) : [r[0] ?? 0, r[1] ?? 0]))
  return means.map((m, k) => {
    const dist = full
      ? multivariateNormal(m, { covariance: fullCovariance(covs[k] ?? [], d) })
      : multivariateNormal([m[0] ?? 0, m[1] ?? 0], { covariance: covarianceBlock(covs[k] ?? [], d, [0, 1]) })
    const lp = toFlat(dist.logProb(x))
    return lp.map((v) => v + Math.log(Math.max(weights[k] ?? 0, 1e-300)))
  })
}

/**
 * The most responsible component of each row under the mixture after step t (from the record's responsibilities when
 * it holds that step, else from its Gaussian parameters).
 *
 * @param steps - The record.
 * @param t - The step.
 * @param rows - The input rows.
 * @returns A component index per row.
 */
export function mixtureAssignment(steps: MixtureSteps, t: number, rows: Rows): number[] {
  const T = steps.objective.length
  const resp = steps.responsibilities
  const stored = resp === undefined ? undefined : resp[t - (T - resp.length)]
  if (stored !== undefined && stored.length === rows.length) return stored.map((r) => r.indexOf(Math.max(...r)))
  const scores = mixtureScores(steps, t, rows)
  if (scores === undefined) return rows.map(() => 0)
  return rows.map((_, i) => {
    let best = 0
    for (let k = 1; k < scores.length; k++)
      if ((scores[k]?.[i] ?? -Infinity) > (scores[best]?.[i] ?? -Infinity)) best = k
    return best
  })
}

/**
 * The EM steps of a mixture as frames for a player: the rows coloured by their most responsible component, each
 * Gaussian component's 1 and 2 standard-deviation ellipses (over the two drawn columns) and its mean, the objective
 * (log likelihood or ELBO) with the current step marked, and the mixing weights at that step.
 *
 * @param steps - The record.
 * @param options - Mode, figure, input rows and columns.
 * @returns The frames.
 */
export function mixtureFrames(steps: MixtureSteps, options: StepsViewOptions = {}): ViewFrames {
  const fig = viewFigure(options, { aspect: 1.75 })
  const mode = fig.mode
  const count = steps.objective.length
  const { rows: drawn, columns, names } = rowsOf(steps, options)
  const d = steps.means[0]?.[0]?.length ?? 0
  const grid0 = drawn === undefined || d !== 2 ? undefined : gridOver(drawn, columns, 50)
  const gridRows = grid0 === undefined ? undefined : grid0.y.flatMap((b) => grid0.x.map((a) => [a, b]))
  const option = (frame: number): EChartsOption => {
    const t = Math.min(Math.max(frame, 0), count - 1)
    const means = steps.means[t] ?? []
    const main: Layer[] = []
    // Two features: the regions of the most responsible component under the rows.
    const scores = gridRows === undefined || grid0 === undefined ? undefined : mixtureScores(steps, t, gridRows)
    if (scores !== undefined && grid0 !== undefined && scores.length > 1)
      main.push(
        regionLayer(
          grid0,
          scores.map((sc) => grid0.y.map((_, i) => sc.slice(i * grid0.x.length, (i + 1) * grid0.x.length))),
          mode,
        ),
      )
    if (drawn !== undefined)
      main.push(
        points(
          clusterSets(drawn, columns, mixtureAssignment(steps, t, drawn), () => ({
            size: 6,
            opacity: 0.75,
          })),
          { mode },
        ),
      )
    const covs = steps.covariances?.[t]
    if (covs !== undefined && d >= 2) {
      const rings: Curve[] = []
      means.forEach((m, k) => {
        const sigma = covarianceBlock(covs[k] ?? [], d, columns)
        for (const nStd of [1, 2]) {
          const e = toArray(
            covarianceEllipse([m[columns[0]] ?? 0, m[columns[1]] ?? 0], sigma, { nStd, points: 72 }).points,
          ) as number[][]
          const closed = [...e, e[0] ?? [0, 0]]
          rings.push({
            x: closed.map((p) => p[0] ?? 0),
            y: closed.map((p) => p[1] ?? 0),
            name: `__ring${k}-${nStd}`,
            colour: clusterColour(k),
            width: nStd === 1 ? 2 : 1,
            dash: nStd === 1 ? 'solid' : 'dashed',
          })
        }
      })
      main.push(lines(rings, { mode }))
    }
    main.push(
      outlined(
        points(
          means.map((m, k) => ({
            x: [m[columns[0]] ?? 0],
            y: [m[columns[1] < d ? columns[1] : 0] ?? 0],
            name: `component ${k}`,
            colour: clusterColour(k),
            size: 14,
            symbol: 'diamond',
          })),
          { mode },
        ),
        mode,
        1.5,
      ),
    )
    const weights = steps.weights[t] ?? []
    const panels: Record<string, Panel> = {
      main: {
        layers: main,
        options: {
          x: { label: names[0], ...(grid0 === undefined ? {} : { limits: limitsOf2(grid0.x) }) },
          y: { label: names[1], ...(grid0 === undefined ? {} : { limits: limitsOf2(grid0.y) }) },
          aspect: grid0 === undefined ? 'equal' : 'auto',
        },
      },
      objective: progressPanel(
        steps.objective,
        t,
        steps.family.startsWith('bayesian') ? 'ELBO' : 'log likelihood',
        mode,
      ),
      weights: {
        layers: [
          bars(
            weights.map((_, k) => `${k}`),
            weights.map((_, k) => ({
              name: `component ${k}`,
              colour: clusterColour(k),
              values: weights.map((w, j) => (j === k ? w : 0)),
            })),
            { mode, stacked: true },
          ),
        ],
        options: { x: { label: 'component' }, y: { label: 'weight', limits: [0, 1] } },
      },
    }
    const grid = gridSpec(2, 2, {
      widthRatios: [1.7, 1],
      wspace: 64,
      hspace: 44,
      cells: [
        { id: 'main', row: 0, col: 0, rowSpan: 2 },
        { id: 'objective', row: 0, col: 1 },
        { id: 'weights', row: 1, col: 1 },
      ],
    })
    return subplots(grid, panels, { figure: fig, legend: false })
  }
  return {
    count,
    label: (f) => `step ${f}/${count - 1} · ${short(steps.objective[f] ?? 0)}`,
    option,
  }
}

/** Options of {@link embeddingFrames}. */
export interface EmbeddingViewOptions extends ViewOptions {
  /** A label per row (class or group) to colour the embedding by, instead of the record's colouring. */
  readonly labels?: readonly (number | string)[]
  /** A number per row (a regression target) to colour the embedding by on a ramp with a colour bar. */
  readonly values?: readonly number[]
  /** Draw the method's neighbour graph under the points (default false; at most 4000 edges). */
  readonly graph?: boolean
}

/**
 * An embedding's optimisation as frames for a player: the rows' positions at each recorded step (coloured by their
 * labels), optionally the neighbour graph, and the objective with the current step marked.
 *
 * @param steps - The record.
 * @param options - Mode, figure, labels and the graph.
 * @returns The frames.
 */
export function embeddingFrames(steps: EmbeddingSteps, options: EmbeddingViewOptions = {}): ViewFrames {
  const count = steps.positions.length
  // An objective worth a curve: more than one value (LLE and Isomap record one).
  const objective = steps.objective !== undefined && steps.objective.length > 1 ? steps.objective : undefined
  // Colour by the labels or values given, else by the record's own colouring.
  const own = options.labels === undefined && options.values === undefined ? steps.colourBy : undefined
  const values = options.values ?? (own?.kind === 'values' ? own.values : undefined)
  const given = options.labels ?? (own?.kind === 'classes' ? own.labels : undefined)
  const classNames = own?.kind === 'classes' ? own.classes : undefined
  const names = [...new Set(options.labels ?? [])]
  const labels = values !== undefined ? undefined : given?.map((l) => (typeof l === 'number' ? l : names.indexOf(l)))
  const nameOf = (c: number): string | undefined =>
    typeof names[c] === 'string' ? String(names[c]) : (classNames?.[c] ?? undefined)
  // A legend of classes sits above the panel, out of the points' way.
  const fig = viewFigure(options, {
    aspect: objective === undefined ? 1.25 : 1.75,
    ...(labels === undefined ? {} : { padding: { top: 40 } }),
  })
  const mode = fig.mode
  const valueSpan = values === undefined ? undefined : spanOf(values)
  const option = (frame: number): EChartsOption => {
    const t = Math.min(Math.max(frame, 0), count - 1)
    const pos = (steps.positions[t] ?? []).map((p) => [p[0] ?? 0, p[1] ?? 0])
    const main: Layer[] = []
    if (options.graph === true && steps.graph !== undefined && steps.graph.length <= 4000) {
      const xs: number[] = []
      const ys: number[] = []
      for (const [i, j] of steps.graph) xs.push(pos[i]?.[0] ?? 0, pos[j]?.[0] ?? 0, Number.NaN)
      for (const [i, j] of steps.graph) ys.push(pos[i]?.[1] ?? 0, pos[j]?.[1] ?? 0, Number.NaN)
      main.push(lines([{ x: xs, y: ys, name: '__graph', colour: 'grid', width: 0.6 }], { mode }))
    }
    const sets =
      values !== undefined
        ? [
            {
              x: pos.map((p) => p[0] ?? 0),
              y: pos.map((p) => p[1] ?? 0),
              values: [...values],
              name: '__rows',
              size: 6,
            },
          ]
        : labels === undefined
          ? [{ x: pos.map((p) => p[0] ?? 0), y: pos.map((p) => p[1] ?? 0), name: 'rows', colour: 'tab0', size: 6 }]
          : clusterSets(pos, [0, 1], labels, (c) => {
              return { size: 6, name: nameOf(c) ?? (c < 0 ? 'unlabelled' : `class ${c}`) }
            })
    main.push(points(sets, { mode }))
    if (valueSpan !== undefined)
      main.push(
        layer([{ type: 'scatter', name: '__colourbar', data: [], silent: true } as SeriesOption], undefined, {
          colourBar: {
            min: valueSpan[0],
            max: valueSpan[1],
            colours: sampleRamp(sequentialRamp('viridis'), 9),
            label: own?.kind === 'values' ? (own.name ?? 'target') : 'target',
          },
        }),
      )
    const embed: Panel = {
      layers: main,
      options: { x: { label: 'dimension 1' }, y: { label: 'dimension 2' }, aspect: 'equal' },
    }
    if (objective === undefined) return cartesian(embed.layers, { ...embed.options, figure: fig, mode, legend: false })
    const grid = gridSpec(1, 2, {
      widthRatios: [1.5, 1],
      // Room for the colour bar between the panels.
      wspace: valueSpan === undefined ? 64 : 120,
      cells: [
        { id: 'main', row: 0, col: 0 },
        { id: 'objective', row: 0, col: 1 },
      ],
    })
    return subplots(
      grid,
      { main: embed, objective: progressPanel(objective, t, 'objective', mode, steps.iterations) },
      { figure: fig },
    )
  }
  return {
    count,
    label: (f) =>
      `iteration ${steps.iterations[f] ?? f}${steps.objective === undefined ? '' : ` · ${short(steps.objective[f] ?? 0)}`}`,
    option,
  }
}
