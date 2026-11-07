/**
 * Drawings of parameterised models beyond a plain diagram: a U-Net (or encoder-decoder) as the familiar U of
 * convolution blocks sized by channels and resolution, with its forward-pass order for animation; a Bayes network as
 * cards holding each variable's marginal and conditional probability table; and plate models with combined plates
 * nested inside each other.
 */

import { MdmlError, maxOf, minOf } from 'mdml-engine/base'
import type { BayesNet } from 'mdml-engine/pgm'
import {
  type Paint,
  path,
  rect,
  type Scene,
  type SceneNode,
  type Structure,
  type StructureGroup,
  scene,
  text,
} from 'mdml-engine/scene'
import { layered } from '../layout/layered.ts'

/** One resolution level of a U-Net or encoder–decoder: its channels and its spatial size (side in pixels). */
export interface UNetLevel {
  readonly channels: number
  readonly size: number
}

/**
 * The levels of a U-Net whose spatial size halves at each level down: `channels[i]` channels at `size / 2^i`.
 *
 * @param size - The input's side in pixels.
 * @param channels - Channels per level, top first (the last is the bottleneck).
 * @returns The levels.
 * @throws MdmlError with code `value` for no levels or a non-positive size.
 * @example
 * ```ts
 * import { unetLevels } from 'mdml-render/structures'
 * unetLevels(64, [32, 64, 128]) // => [{ channels: 32, size: 64 }, { channels: 64, size: 32 }, { channels: 128, size: 16 }]
 * ```
 */
export function unetLevels(size: number, channels: readonly number[]): UNetLevel[] {
  if (channels.length === 0 || !(size > 0)) throw new MdmlError('value', 'unetLevels: need a size and channels')
  return channels.map((c, i) => ({ channels: c, size: Math.max(1, Math.round(size / 2 ** i)) }))
}

/** Options of {@link uNet}. */
export interface UNetOptions {
  /** Draw the skip connections across the U (default true); without them it is a plain encoder–decoder. */
  readonly skips?: boolean
  /** Feed a time (noise-level) embedding into every block, as a diffusion model's U-Net does (default false). */
  readonly timeEmbedding?: boolean
  /** Input and output channels (default: the first level's for the input, none drawn for the output). */
  readonly inputChannels?: number
  readonly outputChannels?: number
  /** Convolutions per block, drawn as slices (default 2). */
  readonly convs?: number
  /** Blocks to highlight (ids `enc:<i>`, `mid`, `dec:<i>`, `out`, `skip:<i>`, `time`): one frame of a forward pass. */
  readonly active?: readonly string[]
  /** Draw the legend of arrow kinds (default true). */
  readonly legend?: boolean
}

/**
 * The ids of a U-Net's blocks (as {@link uNet} names them) in the order a forward pass visits them: down the encoder,
 * the bottleneck, then up the decoder (each decoder block after the skip that feeds it), and the output.
 *
 * @param levels - The levels.
 * @param options - Whether there are skips (default true).
 * @returns The ids, one frame each of an animated pass.
 * @example
 * ```ts
 * import { uNetOrder, unetLevels } from 'mdml-render/structures'
 * uNetOrder(unetLevels(32, [8, 16])) // => ['enc:0', 'mid', 'skip:0', 'dec:0', 'out']
 * ```
 */
export function uNetOrder(levels: readonly UNetLevel[], options: Pick<UNetOptions, 'skips'> = {}): string[] {
  const n = levels.length - 1
  const enc = Array.from({ length: n }, (_, i) => `enc:${i}`)
  const dec = Array.from({ length: n }, (_, i) => `dec:${n - 1 - i}`)
  const withSkips = (options.skips ?? true) ? dec.flatMap((d) => [`skip:${d.slice(4)}`, d]) : dec
  return [...enc, 'mid', ...withSkips, 'out']
}

/**
 * A U-Net (or, without skips, an encoder–decoder) as the familiar U: encoder blocks down the left, each a stack of
 * convolution slices as tall as (the square root of) its spatial size and as wide as (the log of) its channels, the
 * bottleneck at the bottom, decoder blocks up the right, skip connections across, and, for diffusion, a time
 * embedding feeding every block. Arrows are coloured by kind (convolution, downsampling, upsampling, copy) with a
 * legend; `active` highlights blocks for an animated forward pass (see {@link uNetOrder}).
 *
 * @param levels - The levels, top first; the last is the bottleneck (see {@link unetLevels}).
 * @param options - Skips, time embedding, channels in and out, convolutions per block, active blocks and legend.
 * @returns The scene (block ids as in `active`).
 * @throws MdmlError with code `value` for fewer than two levels.
 * @index function Draw a U-Net as a scene: `uNet(unetLevels(64, [16, 32, 64]))`.
 * @example
 * ```ts
 * import { uNet, unetLevels } from 'mdml-render/structures'
 * const s = uNet(unetLevels(64, [16, 32, 64]))
 * s.children.some((n) => n.id === 'skip:0') // => true
 * ```
 */
export function uNet(levels: readonly UNetLevel[], options: UNetOptions = {}): Scene {
  if (levels.length < 2) throw new MdmlError('value', 'uNet: need at least two levels (one and the bottleneck)')
  const skips = options.skips ?? true
  const convs = Math.max(1, options.convs ?? 2)
  const active = new Set(options.active ?? [])
  const n = levels.length - 1
  const top = levels[0] as UNetLevel
  const minC = minOf(levels.map((l) => l.channels))
  const sliceW = (c: number): number => 5 + 4 * Math.log2(c / minC + 1)
  const blockH = (s: number): number => Math.max(22, 150 * Math.sqrt(s / top.size))
  const blockW = (c: number): number => convs * sliceW(c) + (convs - 1) * 3
  const time = options.timeEmbedding === true
  const marginTop = time ? 64 : 34
  const rowGap = 26
  const stepX = 46
  // Rows: level i's blocks are centred on rowY[i].
  const rowY: number[] = []
  let y = marginTop
  for (const l of levels) {
    const h = blockH(l.size)
    rowY.push(y + h / 2)
    y += h + rowGap
  }
  const height = y - rowGap + (options.legend === false ? 24 : 64)
  const left = 70
  const midX = left + n * stepX + blockW((levels[n] as UNetLevel).channels) / 2 + 30
  const width = 2 * midX - left + 70
  const encX = (i: number): number => left + i * stepX
  const decX = (i: number): number => width - left - i * stepX - blockW((levels[i] as UNetLevel).channels)
  const children: SceneNode[] = []
  const stroke = (id: string, base: Paint): { stroke: Paint; strokeWidth: number } =>
    active.has(id) ? { stroke: 'accent', strokeWidth: 2.5 } : { stroke: base, strokeWidth: 1 }

  const block = (id: string, x: number, l: UNetLevel, fill: Paint, label: boolean): void => {
    const h = blockH(l.size)
    const cy = rowY[levels.indexOf(l)] ?? 0
    const w = sliceW(l.channels)
    const slices: SceneNode[] = Array.from({ length: convs }, (_, k) =>
      rect(x + k * (w + 3), cy - h / 2, w, h, { fill, opacity: active.has(id) ? 1 : 0.8, ...stroke(id, 'fg') }),
    )
    children.push({ kind: 'group', id, title: `${l.channels} channels, ${l.size}×${l.size}`, children: slices })
    children.push(text(x + blockW(l.channels) / 2, cy - h / 2 - 6, String(l.channels), { size: 10, anchor: 'middle' }))
    if (label)
      children.push(text(x - 6, cy, `${l.size}²`, { size: 10, anchor: 'end', baseline: 'middle', fill: 'muted' }))
  }
  const arrowPath = (
    id: string,
    points: readonly (readonly [number, number])[],
    colour: Paint,
    dash?: readonly number[],
  ): void => {
    children.push(
      path(points, {
        id,
        arrow: true,
        ...(active.has(id) ? { stroke: 'accent', strokeWidth: 2.5 } : { stroke: colour, strokeWidth: 1.6 }),
        ...(dash === undefined ? {} : { dash }),
      }),
    )
  }

  // Encoder down the left, the bottleneck, the decoder up the right.
  levels.slice(0, n).forEach((l, i) => {
    block(`enc:${i}`, encX(i), l, 'tab0', true)
  })
  const bottom = levels[n] as UNetLevel
  block('mid', midX - blockW(bottom.channels) / 2, bottom, 'tab4', true)
  levels.slice(0, n).forEach((l, i) => {
    block(`dec:${i}`, decX(i), l, 'tab1', false)
  })

  for (let i = 0; i < n; i++) {
    const l = levels[i] as UNetLevel
    const next = levels[i + 1] as UNetLevel
    // Down: from the bottom of encoder block i to the top of the next (or the bottleneck).
    const ex = encX(i) + blockW(l.channels) / 2
    // Into and out of the bottleneck on either side of its centre, so the two arrows never overlap.
    const quarter = blockW(bottom.channels) / 4
    const nextX = i + 1 === n ? midX - quarter : encX(i + 1) + blockW(next.channels) / 2
    const y0 = (rowY[i] ?? 0) + blockH(l.size) / 2
    const y1 = (rowY[i + 1] ?? 0) - blockH(next.size) / 2 - 18
    arrowPath(
      `down:${i}`,
      [
        [ex, y0 + 2],
        [ex, (y0 + y1) / 2],
        [nextX, (y0 + y1) / 2],
        [nextX, y1],
      ],
      'tab3',
    )
    // Up: from the top of the block below to the bottom of decoder block i.
    const dx = decX(i) + blockW(l.channels) / 2
    const belowX = i + 1 === n ? midX + quarter : decX(i + 1) + blockW(next.channels) / 2
    const by = (rowY[i + 1] ?? 0) - blockH(next.size) / 2 - 18
    arrowPath(
      `up:${i}`,
      [
        [belowX, by],
        [belowX, (y0 + by) / 2],
        [dx, (y0 + by) / 2],
        [dx, y0 + 4],
      ],
      'tab2',
    )
    // Skip: across the U at the level's height.
    if (skips) {
      const sy = rowY[i] ?? 0
      arrowPath(
        `skip:${i}`,
        [
          [encX(i) + blockW(l.channels) + 4, sy],
          [decX(i) - 6, sy],
        ],
        'muted',
        [6, 4],
      )
    }
  }
  // In and out near the top of the first level's blocks, clear of the size label at their middle.
  const ty = (rowY[0] ?? 0) - blockH(top.size) / 2 + 16
  arrowPath(
    'in',
    [
      [16, ty],
      [left - 3, ty],
    ],
    'fg',
  )
  children.push(text(16, ty - 8, `${options.inputChannels ?? top.channels} in`, { size: 10, fill: 'muted' }))
  const outX = width - left + 10
  arrowPath(
    'out',
    [
      [decX(0) + blockW(top.channels) + 2, ty],
      [outX + 24, ty],
    ],
    'fg',
  )
  if (options.outputChannels !== undefined)
    children.push(text(outX + 24, ty - 8, `${options.outputChannels} out`, { size: 10, fill: 'muted', anchor: 'end' }))
  // A time embedding along the top, feeding every block.
  if (time) {
    const ey = 22
    children.push(
      rect(midX - 50, ey - 11, 100, 22, {
        id: 'time',
        radius: 11,
        fill: 'surface',
        ...stroke('time', 'tab6'),
      }),
      text(midX, ey, 'time embedding', { size: 10, anchor: 'middle', baseline: 'middle' }),
    )
    const feed = (id: string, x: number, l: UNetLevel): void => {
      const by = (rowY[levels.indexOf(l)] ?? 0) - blockH(l.size) / 2 - 16
      children.push(
        path(
          [
            [midX, ey + 11],
            [x + blockW(l.channels) / 2 + 9, by],
          ],
          { stroke: 'tab6', opacity: 0.35, dash: [2, 3], id },
        ),
      )
    }
    levels.slice(0, n).forEach((l, i) => {
      feed(`time:enc:${i}`, encX(i), l)
      feed(`time:dec:${i}`, decX(i), l)
    })
  }
  // The legend of arrow kinds.
  if (options.legend !== false) {
    const ly = height - 24
    const kinds: [string, Paint, readonly number[] | undefined][] = [
      ['downsample', 'tab3', undefined],
      ['upsample', 'tab2', undefined],
      ...(skips ? [['copy and concatenate', 'muted', [6, 4]] as [string, Paint, readonly number[]]] : []),
    ]
    let lx = 16
    for (const [name, colour, dash] of kinds) {
      children.push(
        path(
          [
            [lx, ly],
            [lx + 24, ly],
          ],
          { stroke: colour, strokeWidth: 1.6, arrow: true, ...(dash === undefined ? {} : { dash }) },
        ),
        text(lx + 30, ly, name, { size: 10, baseline: 'middle', fill: 'muted' }),
      )
      lx += 40 + name.length * 6
    }
    children.push(
      rect(lx, ly - 7, 8, 14, { fill: 'tab0', opacity: 0.8, stroke: 'fg', strokeWidth: 1 }),
      text(lx + 14, ly, 'convolutions (width: channels, height: size)', {
        size: 10,
        baseline: 'middle',
        fill: 'muted',
      }),
    )
  }
  return scene(width, height, children, { title: skips ? 'U-Net' : 'encoder–decoder' })
}

/** Options of {@link bayesNetCards}. */
export interface BayesNetCardOptions {
  /** Observed states by variable: the matching CPT row is highlighted and the variable shaded as observed. */
  readonly evidence?: Readonly<Record<string, string>>
  /** Marginal (or posterior) probabilities by variable, drawn as a bar per state under its name. */
  readonly marginals?: Readonly<Record<string, readonly number[]>>
  /** Ranks down the page (default) or across. */
  readonly direction?: 'down' | 'right'
  /** Rows shown per table before the rest are elided (default 8). */
  readonly maxRows?: number
}

const CELL_W = 42
const ROW_H = 17
const FONT = 11

/**
 * A Bayes network drawn with its parameters: each variable a card holding its name, its marginal (when given) as a
 * bar per state, and its conditional probability table (one row per assignment of its parents, one column per state,
 * each probability as a number over a bar). With `evidence`, observed variables are shaded and the CPT row that applies
 * is highlighted, so the cards read as the arithmetic of the network. Cards are laid out in ranks, arrows between them.
 *
 * @param bn - The network (`bayesNet` in mdml-engine's `pgm`).
 * @param options - Evidence, marginals, direction and the most rows per table.
 * @returns The scene (cards have the variables' names as ids).
 * @index function A Bayes network as probability-table cards: `bayesNetCards(bn)`.
 * @example
 * ```ts
 * import { bayesNet } from 'mdml-engine/pgm'
 * import { bayesNetCards } from 'mdml-render/structures'
 * const bn = bayesNet([
 *   { name: 'rain', states: ['no', 'yes'], parents: [], cpt: [0.8, 0.2] },
 *   { name: 'wet', states: ['no', 'yes'], parents: ['rain'], cpt: [0.9, 0.1, 0.2, 0.8] },
 * ])
 * bayesNetCards(bn).children.filter((n) => n.kind === 'group').map((n) => n.id) // => ['rain', 'wet']
 * ```
 */
export function bayesNetCards(bn: BayesNet, options: BayesNetCardOptions = {}): Scene {
  const evidence = options.evidence ?? {}
  const maxRows = options.maxRows ?? 8
  const byName = new Map(bn.nodes.map((n) => [n.name, n]))
  const textW = (s: string, size = FONT): number => s.length * size * 0.6
  const assignments = (parents: readonly string[]): string[][] => {
    let rows: string[][] = [[]]
    for (const p of parents) {
      const states = byName.get(p)?.states ?? []
      rows = rows.flatMap((r) => states.map((s) => [...r, s]))
    }
    return rows
  }
  const cards = bn.nodes.map((node) => {
    const rows = assignments(node.parents)
    const labels = rows.map((r) => r.map((s, i) => `${node.parents[i]}=${s}`).join(', '))
    const marg = options.marginals?.[node.name]
    // Room on the left for the parents' assignments, or at least for the marginal's `P`.
    const labelW = Math.max(
      node.parents.length === 0 ? 0 : maxOf(labels.map((l) => textW(l, FONT - 1))) + 12,
      marg === undefined ? 0 : 18,
    )
    const shown = Math.min(rows.length, maxRows)
    const width = Math.max(labelW + node.states.length * CELL_W + 16, textW(node.name, FONT + 2) + 24)
    const height =
      26 + (marg === undefined ? 0 : ROW_H + 4) + ROW_H + shown * ROW_H + (rows.length > shown ? ROW_H : 0) + 8
    return { node, rows, labels, labelW, shown, marg, width, height }
  })
  const graph = {
    nodes: cards.map((c) => ({ id: c.node.name, width: c.width, height: c.height })),
    edges: bn.nodes.flatMap((n) => n.parents.map((p) => ({ from: p, to: n.name }))),
  }
  const layout = layered(graph, { direction: options.direction ?? 'down', rankSep: 40, nodeSep: 28 })
  const children: SceneNode[] = layout.edges.map((e, i) =>
    path(e.points, { id: `edge:${i}`, stroke: 'muted', strokeWidth: 1.3, arrow: true, smooth: e.points.length > 2 }),
  )
  for (const c of cards) {
    const b = layout.nodes[c.node.name]
    if (b === undefined) continue
    const x0 = b.x - b.width / 2
    const y0 = b.y - b.height / 2
    const observed = evidence[c.node.name]
    const parts: SceneNode[] = [
      rect(x0, y0, b.width, b.height, {
        radius: 6,
        fill: observed === undefined ? 'surface' : 'grid',
        stroke: observed === undefined ? 'muted' : 'fg',
        strokeWidth: observed === undefined ? 1 : 1.5,
      }),
      text(x0 + 10, y0 + 15, c.node.name, { size: FONT + 2, weight: 'bold', baseline: 'middle' }),
    ]
    if (observed !== undefined)
      parts.push(
        text(x0 + b.width - 8, y0 + 15, `= ${observed}`, {
          size: FONT,
          anchor: 'end',
          baseline: 'middle',
          fill: 'accent',
        }),
      )
    let y = y0 + 28
    const cellX = (k: number): number => x0 + 8 + c.labelW + k * CELL_W
    const bar = (k: number, value: number, rowY: number, fill: Paint, opacity: number): SceneNode =>
      rect(cellX(k) + 2, rowY + 2, Math.max(0, Math.min(1, value)) * (CELL_W - 4), ROW_H - 4, {
        fill,
        opacity,
        radius: 2,
      })
    if (c.marg !== undefined) {
      c.node.states.forEach((_, k) => {
        const v = c.marg?.[k] ?? 0
        parts.push(
          bar(k, v, y, 'accent', 0.35),
          text(cellX(k) + CELL_W / 2, y + ROW_H / 2, v.toFixed(2), {
            size: FONT - 1,
            anchor: 'middle',
            baseline: 'middle',
          }),
        )
      })
      parts.push(text(x0 + 8, y + ROW_H / 2, 'P', { size: FONT - 1, baseline: 'middle', fill: 'accent' }))
      y += ROW_H + 4
    }
    // The table's header: the states.
    c.node.states.forEach((s, k) => {
      const mark = observed === s
      parts.push(
        text(cellX(k) + CELL_W / 2, y + ROW_H / 2, s, {
          size: FONT - 1,
          anchor: 'middle',
          baseline: 'middle',
          fill: mark ? 'accent' : 'muted',
          ...(mark ? { weight: 'bold' as const } : {}),
        }),
      )
    })
    parts.push(
      path(
        [
          [x0 + 6, y + ROW_H],
          [x0 + b.width - 6, y + ROW_H],
        ],
        { stroke: 'grid' },
      ),
    )
    y += ROW_H
    for (let r = 0; r < c.shown; r++) {
      const assignment = c.rows[r] ?? []
      const applies =
        c.node.parents.length > 0 &&
        c.node.parents.every((p, i) => evidence[p] !== undefined && evidence[p] === assignment[i])
      if (applies) parts.push(rect(x0 + 4, y, b.width - 8, ROW_H, { fill: 'accent', opacity: 0.14, radius: 3 }))
      if (c.node.parents.length > 0)
        parts.push(
          text(x0 + 8, y + ROW_H / 2, c.labels[r] ?? '', { size: FONT - 1, baseline: 'middle', fill: 'muted' }),
        )
      c.node.states.forEach((_, k) => {
        const v = c.node.cpt[r * c.node.states.length + k] ?? 0
        parts.push(
          bar(k, v, y, 'tab0', 0.3),
          text(cellX(k) + CELL_W / 2, y + ROW_H / 2, v.toFixed(2), {
            size: FONT - 1,
            anchor: 'middle',
            baseline: 'middle',
          }),
        )
      })
      y += ROW_H
    }
    if (c.rows.length > c.shown)
      parts.push(
        text(x0 + b.width / 2, y + ROW_H / 2, `… ${c.rows.length - c.shown} more rows`, {
          size: FONT - 1,
          anchor: 'middle',
          baseline: 'middle',
          fill: 'muted',
        }),
      )
    children.push({
      kind: 'group',
      id: c.node.name,
      title: `${c.node.name}: ${c.node.states.join(', ')}`,
      children: parts,
    })
  }
  return scene(layout.width, layout.height, children, { title: 'Bayes network' })
}

/**
 * A structure whose plates are named by the plates they combine (`plate:document×word`, as `toStructure` names the
 * plates of a plate model) with those plates nested: `document×word` drawn inside `document`, each plate labelled by
 * its last index set. Plates that some combination needs but no variable sits in alone are added, so every level of the
 * nesting has its frame.
 *
 * @param s - The structure.
 * @param counts - The size of each plate by name, shown in the plate's corner (default the plate's name in capitals).
 * @returns The structure with nested plates.
 * @example
 * ```ts
 * import { structure } from 'mdml-engine/scene'
 * import { nestPlates } from 'mdml-render/structures'
 * const s = structure('plate-model', [{ id: 'x', label: 'x', role: 'observed', group: 'plate:doc×word' }], [], { groups: [{ id: 'plate:doc×word', label: 'doc × word', kind: 'plate' }] })
 * nestPlates(s).groups.map((g) => [g.id, g.parent]) // => [['plate:doc', undefined], ['plate:doc×word', 'plate:doc']]
 * ```
 */
export function nestPlates(s: Structure, counts: Readonly<Record<string, string | number>> = {}): Structure {
  const PREFIX = 'plate:'
  const plates = s.groups.filter((g) => g.kind === 'plate' && g.id.startsWith(PREFIX))
  const others = s.groups.filter((g) => !plates.includes(g))
  const ids = new Set<string>()
  for (const g of plates) {
    const names = g.id.slice(PREFIX.length).split('×')
    for (let k = 1; k <= names.length; k++) ids.add(PREFIX + names.slice(0, k).join('×'))
  }
  const nested: StructureGroup[] = [...ids]
    .sort((a, b) => a.split('×').length - b.split('×').length || a.localeCompare(b))
    .map((id) => {
      const names = id.slice(PREFIX.length).split('×')
      const last = names[names.length - 1] ?? ''
      const parent = names.length > 1 ? PREFIX + names.slice(0, -1).join('×') : undefined
      return {
        id,
        label: last,
        kind: 'plate' as const,
        count: counts[last] ?? last.toUpperCase(),
        ...(parent === undefined ? {} : { parent }),
      }
    })
  return { ...s, groups: [...others, ...nested] }
}
