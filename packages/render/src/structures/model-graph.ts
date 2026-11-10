/**
 * Plate notation for a model spec's structure (`toStructure` in `omai-engine/probprog/model`): `modelGraph` makes the
 * picture the diagram draws read as a plate diagram. Plates over dimensions that overlap without nesting (nodes over
 * `[student]`, `[exam]` and `[student, exam]`) are drawn crossing, each plate's size is written as `N = 40` when it is
 * known, data nodes are the small dots plate notation uses for observed constants, and the vector sites a page wants
 * to annotate element by element (`w[0]`, `w[1]`) can be unrolled. `diagram` applies it to any structure that came
 * from a spec, so a `StructureView` of `toStructure(model)` needs no preparation; call it yourself to pass options.
 */

import type { Structure, StructureEdge, StructureGroup, StructureNode } from 'omai-engine/scene'

/** Options of {@link modelGraph}. */
export interface ModelGraphOptions {
  /**
   * Sites to draw element by element: a site's name to how many elements it has (`{ w: 3 }` draws `w[0]`, `w[1]`, `w[2]`
   * in place of `w`, with the edges of `w` to and from each), or `auto` for every site that is over one dimension of
   * known size up to `maxElements`. Plates whose members are all unrolled are dropped. Default none.
   */
  readonly expand?: 'auto' | Readonly<Record<string, number>>
  /** The most elements `auto` unrolls a site into (default 8). */
  readonly maxElements?: number
  /** The symbol of each plate's size by dimension name (default the dimension's initial in capitals, or the whole name in capitals when two dimensions share an initial). */
  readonly symbols?: Readonly<Record<string, string>>
  /** Draw data nodes as the small dots of plate notation (default) or as the ordinary constant nodes. */
  readonly data?: 'dot' | 'node'
}

/** The id a plate over dimensions takes (as `ids.ts` of the model layer names them). */
const PLATE = 'plate:'
const TIMES = '×'

/**
 * Whether a structure was made from a model spec: its nodes carry the `{ spec, site, dims }` record `toStructure`
 * attaches.
 *
 * @param s - The structure.
 * @returns `true` for a structure of a model spec.
 * @example
 * ```ts
 * import { structure } from 'omai-engine/scene'
 * import { isModelStructure } from 'omai-render/structures'
 * isModelStructure(structure('g', [{ id: 'x', label: 'x', role: 'latent', data: { spec: 'latent', site: 'x', dims: [] } }], [])) // => true
 * ```
 */
export function isModelStructure(s: Structure): boolean {
  return s.nodes.some((n) => typeof n.data?.['spec'] === 'string' && n.data['dims'] !== undefined)
}

const namesOf = (id: string): string[] => id.slice(PLATE.length).split(TIMES)
const plateOf = (names: readonly string[]): string => `${PLATE}${names.join(TIMES)}`

/** The symbol of each dimension's size: its initial in capitals, or its whole name when two share an initial. */
function symbolsOf(dims: readonly string[], given: Readonly<Record<string, string>> | undefined): Map<string, string> {
  const initials = new Map<string, number>()
  for (const d of dims) initials.set(d.charAt(0).toUpperCase(), (initials.get(d.charAt(0).toUpperCase()) ?? 0) + 1)
  return new Map(
    dims.map((d) => [
      d,
      given?.[d] ?? ((initials.get(d.charAt(0).toUpperCase()) ?? 0) > 1 ? d.toUpperCase() : d.charAt(0).toUpperCase()),
    ]),
  )
}

/**
 * Plates that draw the same dimension twice: the dimensions that are the last (innermost) name of more than one plate.
 * Plates over `[student]`, `[exam]` and `[student, exam]` draw `exam` twice, which is what crossing plates avoid.
 */
function repeated(plates: readonly StructureGroup[]): Set<string> {
  const last = new Map<string, number>()
  for (const g of plates) {
    const d = namesOf(g.id).at(-1) as string
    last.set(d, (last.get(d) ?? 0) + 1)
  }
  return new Set([...last].filter(([, n]) => n > 1).map(([d]) => d))
}

/**
 * Draw the structure of a model spec as a plate diagram. Four things change from what `toStructure` gives, and a
 * structure that did not come from a spec is returned as it is:
 *
 * - **Crossing plates.** When two plates draw the same dimension (nodes over `[student]`, `[exam]` and `[student,
 *   exam]`), the nested chain would draw `exam` twice. The nodes over several dimensions instead list every plate they
 *   are in (`StructureNode.groups`) and the plate that stood for their overlap is marked `intersection`, so the diagram
 *   draws the frames crossing where it can and the nested chain where it cannot.
 * - **Sizes.** A plate whose size is known is counted `N = 40` (the symbol of the dimension, `symbols` to choose it), one
 *   whose size is not just `N`.
 * - **Data.** Data nodes are the small dots plate notation draws for observed constants and hyperparameters, with their
 *   name beside them (`data: 'node'` keeps the constant nodes). Generated quantities are dashed circles.
 * - **Elements.** `expand` draws chosen vector sites element by element (`w[0]`, `w[1]`, ...), so inference state can be
 *   painted on each.
 *
 * The result has the type `model-graph` (`model-factor-graph` for the factor view), whose visual language `styleFor`
 * gives: solid directed edges, shaded observed circles, double circles for derived nodes, squares for factors. Ids are
 * kept (a site's name is its node's id) so the overlays of an inference run address the same nodes.
 *
 * @param s - A structure from `toStructure` of a model spec.
 * @param options - Elements to unroll, size symbols, how data is drawn.
 * @returns The structure (the same object when it is not a model's or already prepared).
 * @index function Prepare a model spec's Structure as a plate diagram: crossing plates, sizes, data dots.
 * @example
 * ```ts
 * import { structure } from 'omai-engine/scene'
 * import { modelGraph } from 'omai-render/structures'
 * const node = (id: string, dims: string[], group: string) => ({ id, label: id, role: 'latent' as const, group, data: { spec: 'latent', site: id, dims } })
 * const s = structure('graphical-model', [node('skill', ['student'], 'plate:student'), node('pass', ['student', 'exam'], 'plate:student×exam')], [{ from: 'skill', to: 'pass' }], {
 *   groups: [
 *     { id: 'plate:student', label: 'student', kind: 'plate', count: 40 },
 *     { id: 'plate:student×exam', label: 'exam', kind: 'plate', parent: 'plate:student', count: 5 },
 *   ],
 * })
 * const g = modelGraph(s)
 * g.nodes.find((n) => n.id === 'pass')?.groups // => ['plate:student', 'plate:exam']
 * g.groups.find((x) => x.id === 'plate:student')?.count // => 'S = 40'
 * ```
 */
export function modelGraph(s: Structure, options: ModelGraphOptions = {}): Structure {
  if (!isModelStructure(s) || s.type.startsWith('model-')) return s
  const plates = s.groups.filter((g) => g.kind === 'plate' && g.id.startsWith(PLATE))
  const dims = [...new Set(plates.flatMap((g) => namesOf(g.id)))]
  const symbol = symbolsOf(dims, options.symbols)
  const dup = repeated(plates)
  // The size of each dimension, from the plate that is its last name; its label from the same.
  const sizeOf = new Map<string, number | string>()
  const labelOf = new Map<string, string>()
  for (const g of plates) {
    const d = namesOf(g.id).at(-1) as string
    if (g.count !== undefined && !sizeOf.has(d)) sizeOf.set(d, g.count)
    if (!labelOf.has(d)) labelOf.set(d, g.label)
  }
  const sized = (d: string): string => {
    const n = sizeOf.get(d)
    return typeof n === 'number' ? `${symbol.get(d)} = ${n}` : (symbol.get(d) as string)
  }
  // A plate over several dimensions, one of which another plate also draws, is replaced by membership of each plate.
  const converted = new Map<string, string[]>()
  for (const g of plates) {
    const names = namesOf(g.id)
    if (names.length > 1 && names.some((d) => dup.has(d)))
      converted.set(
        g.id,
        names.map((d) => plateOf([d])),
      )
  }
  const groups: StructureGroup[] = plates.map((g): StructureGroup => {
    const d = namesOf(g.id).at(-1) as string
    const marked = converted.get(g.id)
    return { ...g, count: sized(d), ...(marked === undefined ? {} : { intersection: marked }) }
  })
  const have = new Set(groups.map((g) => g.id))
  for (const ids of converted.values())
    for (const id of ids)
      if (!have.has(id)) {
        have.add(id)
        const d = namesOf(id)[0] as string
        groups.push({ id, label: labelOf.get(d) ?? d, kind: 'plate', count: sized(d) })
      }
  const others = s.groups.filter((g) => !plates.includes(g))
  const isDot = options.data !== 'node'
  let nodes: StructureNode[] = s.nodes.map((n) => {
    const marked = n.group === undefined ? undefined : converted.get(n.group)
    const spec = n.data?.['spec']
    const look =
      spec === 'data' && isDot
        ? { ...n.look, shape: 'dot' as const, fill: 'fg' as const, stroke: 'fg' as const }
        : spec === 'generated'
          ? { ...n.look, dash: n.look?.dash ?? [3, 3], fill: 'surface' as const }
          : n.look
    // The plates say which dimensions a node is over, and the look that it is generated or data: neither is a badge.
    const { badges: all, ...rest } = n
    const { dims: _dims, generated: _generated, role: dataRole, ...badges } = all ?? {}
    const detail =
      spec === 'data' && isDot
        ? [dataRole, n.detail === 'data' ? undefined : n.detail].filter((x) => x !== undefined).join(', ') || undefined
        : spec === 'generated' && n.detail !== undefined
          ? `generated ${n.detail}`
          : n.detail
    return {
      ...rest,
      ...(marked === undefined ? {} : { groups: marked }),
      ...(look === undefined ? {} : { look }),
      ...(detail === undefined ? {} : { detail }),
      ...(Object.keys(badges).length === 0 ? {} : { badges }),
    }
  })
  let edges: readonly StructureEdge[] = s.edges
  if (options.expand !== undefined) ({ nodes, edges } = unroll(nodes, edges, sizeOf, options))
  const used = new Set(nodes.flatMap((n) => [n.group, ...(n.groups ?? [])]).filter((g): g is string => g !== undefined))
  // A plate with nothing in it (every site unrolled) is not drawn; the plates that hold a drawn plate stay.
  const keep = (g: StructureGroup): boolean => used.has(g.id) || groups.some((c) => c.parent === g.id && keep(c))
  const allGroups = options.expand === undefined ? groups : groups.filter(keep)
  const factor = s.type === 'factor-graph'
  return { ...s, type: factor ? 'model-factor-graph' : 'model-graph', nodes, edges, groups: [...others, ...allGroups] }
}

/** Sites drawn element by element: each becomes its elements, with the edges of the site to and from each. */
function unroll(
  nodes: readonly StructureNode[],
  edges: readonly StructureEdge[],
  sizeOf: ReadonlyMap<string, number | string>,
  options: ModelGraphOptions,
): { nodes: StructureNode[]; edges: StructureEdge[] } {
  const max = options.maxElements ?? 8
  const count = (n: StructureNode): number => {
    const given = options.expand === 'auto' ? undefined : (options.expand as Record<string, number>)[n.id]
    if (given !== undefined) return given
    if (options.expand !== 'auto') return 0
    const d = n.data?.['dims']
    if (n.data?.['spec'] === 'data' || !Array.isArray(d) || d.length !== 1) return 0
    const size = sizeOf.get(String(d[0]))
    return typeof size === 'number' && size <= max ? size : 0
  }
  const elements = new Map<string, string[]>()
  const out: StructureNode[] = []
  for (const n of nodes) {
    const k = count(n)
    if (k < 2) {
      out.push(n)
      continue
    }
    const ids = Array.from({ length: k }, (_, i) => `${n.id}[${i}]`)
    elements.set(n.id, ids)
    for (const [i, id] of ids.entries())
      out.push({
        ...n,
        id,
        label: id,
        data: { ...n.data, element: i, of: n.id },
      })
  }
  const ends = (id: string): string[] => elements.get(id) ?? [id]
  const next: StructureEdge[] = []
  for (const e of edges) {
    const from = ends(e.from)
    const to = ends(e.to)
    // Elementwise when both sides are unrolled to the same length, every pair otherwise.
    if (from.length > 1 && from.length === to.length)
      for (const [i, f] of from.entries()) next.push({ ...e, from: f, to: to[i] as string })
    else for (const f of from) for (const t of to) next.push({ ...e, from: f, to: t })
  }
  return { nodes: out, edges: next }
}
