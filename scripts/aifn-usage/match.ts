/**
 * Set the notes' usage of the AIFN engine against omai, and write `docs/aifn/data/match.json` and the markdown tables.
 *
 *   node scripts/aifn-usage/match.ts
 *
 * Runs the three scans in memory (usage of ml-concepts, the AIFN surface, the omai surface) and gives every AIFN
 * declaration a status:
 *
 * - `exact`: omai exports the same name in the module that `aliases.json` expects for the AIFN entry point (or, with no
 *   expectation recorded, in exactly one module).
 * - `alias`: a hand-kept entry of `aliases.json` names the omai symbol (renamed, moved, or a different shape of the
 *   same capability); every target is checked to exist.
 * - `design`: covered by a differently shaped omai API (the entry says which); not a gap.
 * - `internal`: not a capability to port (a type of a symbol that is matched, a helper private to a note, a re-export).
 * - `type`: a type or interface with no same-named omai type; a type follows the value it describes and is not a capability.
 * - `elsewhere`: the name exists in omai only in modules the map does not expect; needs a look, never counted as covered.
 * - `unmatched`: no counterpart found.
 *
 * `aliases.json` is the only hand-kept input; every entry has a reason.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { type AifnSymbol, DOCS, nearestNames, table, writeJson, writeText } from './lib.ts'
import { scanAifn } from './scan-aifn.ts'
import { type OmaiSymbol, scanOmai } from './scan-omai.ts'
import { scanUsage } from './scan-usage.ts'

interface AliasFile {
  /** AIFN entry point (or prefix ending in `/`) -> the omai modules to look in for same-named symbols. */
  readonly modules: readonly { readonly aifn: string; readonly omai: readonly string[]; readonly reason: string }[]
  /** AIFN `entry#name` or canonical id -> omai symbols, with the reason (rename, move). */
  readonly symbols: readonly { readonly aifn: string; readonly omai: readonly string[]; readonly reason: string }[]
  /** Covered by a differently shaped omai API. */
  readonly design: readonly { readonly aifn: string; readonly omai: readonly string[]; readonly reason: string }[]
  /** Not a capability to port. */
  readonly internal: readonly { readonly aifn: string; readonly reason: string }[]
  /** A same-named omai symbol that is a different thing: the AIFN symbol has no counterpart. */
  readonly none: readonly { readonly aifn: string; readonly reason: string }[]
}

type Status = 'exact' | 'alias' | 'design' | 'internal' | 'type' | 'elsewhere' | 'unmatched'

interface GapGroup {
  readonly id: string
  readonly title: string
  readonly area: string
  readonly patterns: readonly string[]
  readonly size: string
  readonly inProgress?: boolean
}

type Matched = {
  readonly status: Status
  readonly omai?: readonly string[]
  readonly reason?: string
  readonly via?: string
}

interface Rule {
  readonly entry: string
  readonly prefix: boolean
  readonly name: string
}

/** `aifn-x/y#name`, `aifn-x/y/**#name`, `aifn-x/y#*`: an entry point (or a prefix of them) and a name (or all). */
function parseRule(key: string): Rule {
  const [entry = '', name = '*'] = key.split('#')
  const prefix = entry.endsWith('/**')
  return { entry: prefix ? entry.slice(0, -3) : entry, prefix, name }
}

function ruleMatches(rule: Rule, entry: string, name: string): boolean {
  const inEntry = rule.prefix ? entry === rule.entry || entry.startsWith(`${rule.entry}/`) : entry === rule.entry
  return inEntry && (rule.name === '*' || rule.name === name)
}

const homeOf = (id: string): string => id.slice(0, id.indexOf('#'))

function loadAliases(omaiIds: ReadonlySet<string>): AliasFile {
  const path = join(import.meta.dirname, 'aliases.json')
  const a = JSON.parse(readFileSync(path, 'utf8')) as AliasFile
  const missing: string[] = []
  for (const e of [...a.symbols, ...a.design]) {
    for (const t of e.omai) if (!omaiIds.has(t)) missing.push(`${e.aifn} -> ${t}`)
  }
  if (missing.length > 0) throw new Error(`aliases.json names omai symbols that do not exist:\n${missing.join('\n')}`)
  return a
}

const area = (source: string): string =>
  source.startsWith('..') ? 'render/misc' : source.split('/').slice(0, 2).join('/')

function pct(n: number, d: number): string {
  return d === 0 ? '-' : `${Math.round((100 * n) / d)}%`
}

async function main(): Promise<void> {
  const usage = scanUsage()
  const aifn = scanAifn()
  const omai = await scanOmai()
  const omaiIds = new Set<string>()
  const omaiByName = new Map<string, OmaiSymbol[]>()
  for (const s of omai) {
    omaiIds.add(`${s.module}#${s.name}`)
    for (const a of s.aliases) omaiIds.add(a)
    omaiByName.set(s.name, [...(omaiByName.get(s.name) ?? []), s])
  }
  const aliases = loadAliases(omaiIds)
  const moduleRules = aliases.modules.map((m) => ({
    rule: parseRule(m.aifn.includes('#') ? m.aifn : `${m.aifn}#*`),
    omai: m.omai,
  }))
  const symbolRules = aliases.symbols.map((m) => ({ rule: parseRule(m.aifn), ...m }))
  const designRules = aliases.design.map((m) => ({ rule: parseRule(m.aifn), ...m }))
  const internalRules = aliases.internal.map((m) => ({ rule: parseRule(m.aifn), ...m }))
  const noneRules = aliases.none.map((m) => ({ rule: parseRule(m.aifn), ...m }))

  // Index the AIFN surface: (entry#name) -> declaration.
  const byKey = new Map<string, AifnSymbol>()
  for (const s of aifn.symbols) {
    byKey.set(s.id, s)
    for (const e of s.alsoIn ?? []) byKey.set(`${e}#${s.name}`, s)
  }
  const entriesOf = (s: AifnSymbol): string[] => [homeOf(s.id), ...(s.alsoIn ?? [])]

  // Expected omai modules for a symbol, from the module rules (the longest matching prefix of each entry point).
  const expectedModules = (s: AifnSymbol): string[] => {
    const out: string[] = []
    for (const entry of entriesOf(s)) {
      let best: { len: number; omai: readonly string[] } | undefined
      for (const m of moduleRules) {
        if (!ruleMatches(m.rule, entry, s.name)) continue
        const len = m.rule.entry.length + (m.rule.name === '*' ? 0 : 1000)
        if (!best || len > best.len) best = { len, omai: m.omai }
      }
      if (best) out.push(...best.omai)
    }
    return out
  }
  const inModule = (o: OmaiSymbol, prefixes: readonly string[]): boolean =>
    [o.module, ...o.aliases.map((a) => a.slice(0, a.indexOf('#')))].some((m) =>
      prefixes.some((p) => m === p || m.startsWith(`${p}/`)),
    )

  const statusOf = new Map<string, Matched>()
  const classify = (s: AifnSymbol): Matched => {
    const entries = entriesOf(s)
    const hit = <T extends { rule: Rule }>(rules: readonly T[]): T | undefined =>
      rules.find((r) => entries.some((e) => ruleMatches(r.rule, e, s.name)))
    const internal = hit(internalRules)
    if (internal) return { status: 'internal', reason: internal.reason }
    const none = hit(noneRules)
    if (none) return { status: 'unmatched', reason: none.reason }
    const design = hit(designRules)
    if (design) return { status: 'design', omai: design.omai, reason: design.reason }
    const alias = hit(symbolRules)
    if (alias) return { status: 'alias', omai: alias.omai, reason: alias.reason }
    if (s.kind === 'namespace') return { status: 'internal', reason: 'a namespace re-export' }
    const cands = omaiByName.get(s.name) ?? []
    const typeOnly = s.kind === 'interface' || s.kind === 'type'
    if (cands.length === 0) return { status: typeOnly ? 'type' : 'unmatched' }
    const expected = expectedModules(s)
    if (expected.length > 0) {
      const ok = cands.filter((c) => inModule(c, expected))
      if (ok.length > 0) return { status: 'exact', omai: ok.map((o) => `${o.module}#${o.name}`), via: 'mapped module' }
      return { status: typeOnly ? 'type' : 'elsewhere', omai: cands.map((o) => `${o.module}#${o.name}`) }
    }
    if (cands.length === 1) return { status: 'exact', omai: [`${cands[0]?.module}#${s.name}`], via: 'unique name' }
    return { status: typeOnly ? 'type' : 'elsewhere', omai: cands.map((o) => `${o.module}#${o.name}`) }
  }
  for (const s of aifn.symbols) statusOf.set(s.id, classify(s))

  // Used symbols, folded onto their declarations.
  interface Use {
    sym: AifnSymbol
    notes: Set<number>
    files: number
    lab: number
    site: number
    scripts: number
    imports: Set<string>
  }
  const used = new Map<string, Use>()
  const unresolved: { id: string; notes: number; files: number; labFiles: number }[] = []
  const add = (
    sym: AifnSymbol,
    id: string,
    notes: readonly number[],
    files: number,
    lab: number,
    site: number,
    scripts: number,
  ): void => {
    const u = used.get(sym.id) ?? {
      sym,
      notes: new Set<number>(),
      files: 0,
      lab: 0,
      site: 0,
      scripts: 0,
      imports: new Set<string>(),
    }
    for (const n of notes) u.notes.add(n)
    u.files += files
    u.lab += lab
    u.site += site
    u.scripts += scripts
    u.imports.add(id)
    used.set(sym.id, u)
  }
  for (const s of usage.symbols) {
    const sym = byKey.get(s.id)
    if (!sym) {
      unresolved.push({ id: s.id, notes: s.notes, files: s.files, labFiles: s.labFiles })
      continue
    }
    add(sym, s.id, s.noteIndexes, s.files, s.labFiles, s.siteFiles, s.scriptFiles)
  }
  const workerUnresolved: string[] = []
  for (const w of usage.workerCalls) {
    const i = w.call.lastIndexOf('/')
    const name = w.call.slice(i + 1)
    const sub = w.call.slice(0, i)
    const sym = ['aifn-compute', 'aifn-methods']
      .map((p) => byKey.get(`${p}/${sub}#${name}`))
      .find((x) => x !== undefined)
    if (sym) add(sym, `worker:${w.call}`, w.noteIndexes, 0, 0, 0, 0)
    else workerUnresolved.push(w.call)
  }

  // Per-note statuses.
  const gapStatuses = new Set<Status>(['unmatched', 'elsewhere'])
  const isValue = (k: string): boolean => k !== 'interface' && k !== 'type'
  const noteSyms = new Map<number, string[]>()
  for (const [id, u] of used) for (const n of u.notes) noteSyms.set(n, [...(noteSyms.get(n) ?? []), id])
  let portable = 0
  let portableValues = 0
  let engineCovered = 0
  let renderCovered = 0
  const soleBlocker = new Map<string, number>()
  for (const [, ids] of noteSyms) {
    const blockers = ids.filter((id) => gapStatuses.has((statusOf.get(id) as Matched).status))
    const valueBlockers = blockers.filter((id) => isValue((used.get(id) as Use).sym.kind))
    if (blockers.length === 0) portable++
    if (!valueBlockers.some((id) => !id.startsWith('aifn-render'))) engineCovered++
    if (!valueBlockers.some((id) => id.startsWith('aifn-render'))) renderCovered++
    if (valueBlockers.length === 0) portableValues++
    if (valueBlockers.length === 1)
      soleBlocker.set(valueBlockers[0] as string, (soleBlocker.get(valueBlockers[0] as string) ?? 0) + 1)
  }

  // Records.
  const noteUsed = [...used.values()].filter((u) => u.notes.size > 0 || u.files > 0)
  const labOnly = [...used.values()].filter((u) => u.notes.size === 0 && u.files === 0)
  const rec = (u: Use): Record<string, unknown> => {
    const m = statusOf.get(u.sym.id) as Matched
    return {
      id: u.sym.id,
      kind: u.sym.kind,
      status: m.status,
      ...(m.omai ? { omai: m.omai.slice(0, 3) } : {}),
      ...(m.via ? { via: m.via } : {}),
      ...(m.reason ? { reason: m.reason } : {}),
      notes: u.notes.size,
      files: u.files,
      lab: u.lab,
      soleBlocker: soleBlocker.get(u.sym.id) ?? 0,
      source: u.sym.source,
      imports: [...u.imports].filter((i) => i !== u.sym.id).slice(0, 3),
      examples: [...u.notes]
        .sort((a, b) => a - b)
        .slice(0, 3)
        .map((i) => usage.notes[i] as string),
    }
  }
  const count = (list: readonly Use[]): Record<string, number> => {
    const c: Record<string, number> = {}
    for (const u of list) {
      const k = (statusOf.get(u.sym.id) as Matched).status
      c[k] = (c[k] ?? 0) + 1
    }
    return c
  }
  const valuesOf = (list: readonly Use[]): Use[] => list.filter((u) => isValue(u.sym.kind))
  // The same coverage weighted by use: every (note, value) pair counts once.
  let pairs = 0
  let pairsCovered = 0
  for (const u of valuesOf(noteUsed)) {
    pairs += u.notes.size
    if (!gapStatuses.has((statusOf.get(u.sym.id) as Matched).status)) pairsCovered += u.notes.size
  }
  const totals = {
    notePairs: pairs,
    notePairsCovered: pairsCovered,
    notesScanned: usage.totals['notes'],
    notesUsingAifn: usage.totals['notesUsingAifn'],
    notesWithAllSymbolsCovered: portable,
    notesWithAllValueSymbolsCovered: portableValues,
    notesWithEverySimulationSymbolCovered: engineCovered,
    notesWithEveryRenderSymbolCovered: renderCovered,
    usedByNotes: noteUsed.length,
    usedByNotesValues: valuesOf(noteUsed).length,
    usedByNotesStatus: count(noteUsed),
    usedByNotesValuesStatus: count(valuesOf(noteUsed)),
    usedOnlyByLab: labOnly.length,
    usedOnlyByLabStatus: count(labOnly),
    unresolvedImports: unresolved.length,
    workerCallsUnresolved: workerUnresolved.length,
    aifnDeclarations: aifn.symbols.length,
    aifnStatus: (() => {
      const c: Record<string, number> = {}
      for (const s of aifn.symbols)
        c[(statusOf.get(s.id) as Matched).status] = (c[(statusOf.get(s.id) as Matched).status] ?? 0) + 1
      return c
    })(),
  }

  const usedRecs = noteUsed
    .map(rec)
    .sort(
      (a, b) => (b['notes'] as number) - (a['notes'] as number) || ((a['id'] as string) < (b['id'] as string) ? -1 : 1),
    )
  const unusedUnmatched = aifn.symbols
    .filter(
      (s) =>
        !used.has(s.id) &&
        isValue(s.kind) &&
        ['unmatched', 'elsewhere'].includes((statusOf.get(s.id) as Matched).status),
    )
    .map((s) => ({ id: s.id, kind: s.kind, status: (statusOf.get(s.id) as Matched).status, source: s.source }))
  const labRecs = labOnly.map(rec)
  // Gap groups: the unmatched value symbols folded into capabilities, in the order of gap-groups.json (first match wins).
  const groups = JSON.parse(readFileSync(join(import.meta.dirname, 'gap-groups.json'), 'utf8')) as GapGroup[]
  const groupRules = groups.map((g) => g.patterns.map(parseRule))
  const gapUses = [...used.values()].filter(
    (u) =>
      (u.notes.size > 0 || u.files > 0) &&
      gapStatuses.has((statusOf.get(u.sym.id) as Matched).status) &&
      isValue(u.sym.kind),
  )
  const groupRows = groups.map((g) => ({ ...g, symbols: [] as Use[] }))
  const ungrouped: Use[] = []
  for (const u of gapUses) {
    const entries = [homeOf(u.sym.id), ...(u.sym.alsoIn ?? [])]
    const at = groupRules.findIndex((rules) => rules.some((r) => entries.some((e) => ruleMatches(r, e, u.sym.name))))
    if (at < 0) ungrouped.push(u)
    else groupRows[at]?.symbols.push(u)
  }
  // Notes whose every gap symbol lies in one group: what building that group alone would free.
  const groupOf = new Map<string, number>()
  groupRows.forEach((g, i) => {
    for (const u of g.symbols) groupOf.set(u.sym.id, i)
  })
  const onlyBlockedBy = new Array<number>(groupRows.length).fill(0)
  for (const ids of noteSyms.values()) {
    const hit = new Set(ids.filter((id) => groupOf.has(id)).map((id) => groupOf.get(id) as number))
    // Readout is in nearly every note; set it aside to see what each other group would free (and count it on its own).
    const rest = new Set([...hit].filter((i) => groupRows[i]?.id !== 'readout'))
    const only = hit.size === 1 ? [...hit][0] : rest.size === 1 ? [...rest][0] : undefined
    if (only !== undefined) onlyBlockedBy[only] = (onlyBlockedBy[only] ?? 0) + 1
  }
  const groupSummary = groupRows.map((g, gi) => {
    const notesIn = new Set<number>()
    for (const u of g.symbols) for (const n of u.notes) notesIn.add(n)
    const usedByNotes = g.symbols.filter((u) => u.notes.size > 0 || u.files > 0)
    return {
      id: g.id,
      title: g.title,
      area: g.area,
      size: g.size,
      inProgress: g.inProgress === true,
      symbolsUsedByNotes: usedByNotes.length,
      notes: notesIn.size,
      notesBlockedOnlyByThis: onlyBlockedBy[gi] ?? 0, // not counting Readout, which most notes also need
      topSymbols: [...usedByNotes]
        .sort((a, b) => b.notes.size - a.notes.size)
        .slice(0, 6)
        .map((u) => `${u.sym.name} (${u.notes.size})`),
    }
  })
  const priority = (n: number): string => (n >= 40 ? 'high' : n >= 10 ? 'medium' : 'low')
  inject(
    'gaps.md',
    'groups',
    table(
      ['gap', 'area', 'notes', 'only this', 'symbols', 'size', 'priority', 'most used'],
      groupSummary
        .filter((g) => !g.inProgress)
        .sort((a, b) => b.notes - a.notes)
        .map((g) => [
          g.title,
          g.area,
          g.notes,
          g.notesBlockedOnlyByThis,
          g.symbolsUsedByNotes,
          g.size,
          priority(g.notes),
          g.topSymbols.slice(0, 4).join(', '),
        ]),
    ),
  )
  writeJson('gap-groups.json', {
    sources: { ml: usage.source },
    groups: groupSummary,
    ungrouped: ungrouped.map((u) => u.sym.id),
  })

  writeJson('match.json', {
    sources: { ml: usage.source, aifn: aifn.commit, omaiExports: omai.length },
    totals,
    used: usedRecs,
    labOnly: labRecs.filter((r) => gapStatuses.has(r['status'] as Status)),
    unresolved,
    workerUnresolved,
  })
  writeJson('unused-unmatched.json', { sources: { aifn: aifn.commit }, symbols: unusedUnmatched })

  // Tables.
  const areaRows = new Map<string, { used: Use[]; notes: Set<number> }>()
  for (const u of noteUsed) {
    const a = area(u.sym.source)
    const r = areaRows.get(a) ?? { used: [], notes: new Set<number>() }
    r.used.push(u)
    for (const n of u.notes) r.notes.add(n)
    areaRows.set(a, r)
  }
  const statusOfUse = (u: Use): Status => (statusOf.get(u.sym.id) as Matched).status
  const areaTable = table(
    [
      'area',
      'notes',
      'values used',
      'same or renamed',
      'other shape',
      'not a capability',
      'gap',
      'top symbols (notes)',
    ],
    [...areaRows.entries()]
      .sort((a, b) => b[1].notes.size - a[1].notes.size)
      .map(([a, r]) => {
        const vals = valuesOf(r.used)
        const n = (f: (st: Status) => boolean): number => vals.filter((u) => f(statusOfUse(u))).length
        const top = [...vals]
          .sort((x, y) => y.notes.size - x.notes.size)
          .slice(0, 5)
          .map((u) => `${u.sym.name} (${u.notes.size})`)
        return [
          a,
          r.notes.size,
          vals.length,
          n((x) => x === 'exact' || x === 'alias'),
          n((x) => x === 'design'),
          n((x) => x === 'internal'),
          n((x) => x === 'unmatched' || x === 'elsewhere'),
          top.join(', '),
        ]
      }),
  )
  const topTable = table(
    ['symbol', 'notes', 'status', 'in omai'],
    [...noteUsed]
      .filter((u) => isValue(u.sym.kind))
      .sort((x, y) => y.notes.size - x.notes.size || (x.sym.id < y.sym.id ? -1 : 1))
      .slice(0, 40)
      .map((u) => {
        const m = statusOf.get(u.sym.id) as Matched
        return [
          `\`${u.sym.id.replace('aifn-', '')}\``,
          u.notes.size,
          m.status,
          m.omai ? `\`${m.omai[0]?.replace('omai-', '')}\`` : '-',
        ]
      }),
  )
  const subTable = table(
    ['import path', 'notes', 'symbols'],
    usage.subpaths
      .filter((x) => x.notes > 0)
      .slice(0, 40)
      .map((x) => [`\`${x.specifier}\``, x.notes, x.symbols]),
  )
  const workerTable = table(
    ['worker call', 'notes'],
    usage.workerCalls.map((w) => [`\`${w.call}\``, w.notes]),
  )
  const gapTable = (list: readonly Record<string, unknown>[]): string =>
    table(
      ['symbol', 'kind', 'notes', 'sole', 'status', 'aifn source'],
      list.map((r) => [
        `\`${r['id']}\``,
        r['kind'] as string,
        r['notes'] as number,
        r['soleBlocker'] as number,
        r['status'] as string,
        `\`${r['source']}\``,
      ]),
    )
  const gaps = usedRecs.filter((r) => gapStatuses.has(r['status'] as Status))
  writeText(
    'data/tables.md',
    [
      '# Generated tables',
      '',
      'Written by `scripts/aifn-usage/match.ts`; do not edit.',
      '',
      '## Usage by area',
      '',
      areaTable,
      '',
      '## The 40 most used values',
      '',
      topTable,
      '',
      '## The most used import paths',
      '',
      subTable,
      '',
      "## Worker calls (`call('module/export')`)",
      '',
      workerTable,
      '',
      '## Used symbols with no omai counterpart yet (values)',
      '',
      gapTable(gaps.filter((r) => isValue(r['kind'] as string))),
      '',
      '## Used types with no omai counterpart',
      '',
      gapTable(gaps.filter((r) => !isValue(r['kind'] as string))),
      '',
    ].join('\n'),
  )
  // The alias map as the notes meet it: every hand-decided symbol that a note uses, most used first.
  const decided = usedRecs.filter(
    (r) => ['alias', 'design', 'internal'].includes(r['status'] as string) || r['reason'] !== undefined,
  )
  const aliasRow = (r: Record<string, unknown>): (string | number)[] => [
    `\`${(r['id'] as string).replace('aifn-', '')}\``,
    r['notes'] as number,
    r['status'] as string,
    ((r['omai'] as string[] | undefined) ?? []).map((o) => `\`${o.replace('omai-', '')}\``).join(', ') || '-',
    (r['reason'] as string | undefined) ?? '',
  ]
  const aliasHead = ['aifn symbol', 'notes', 'status', 'omai', 'why']
  writeText(
    'data/aliases.md',
    [
      '# The alias map as the notes meet it',
      '',
      'Written by `scripts/aifn-usage/match.ts` from `scripts/aifn-usage/aliases.json`; do not edit. One row per AIFN symbol that a note uses and that a hand-kept entry decides (`alias`: renamed or moved; `design`: covered by a differently shaped API; `internal`: not a capability to port), most used first.',
      '',
      table(aliasHead, decided.map(aliasRow)),
      '',
    ].join('\n'),
  )
  inject(
    'mapping.md',
    'renamed-most-used',
    table(
      aliasHead,
      decided
        .filter((r) => r['status'] === 'alias')
        .slice(0, 45)
        .map(aliasRow),
    ),
  )
  inject(
    'mapping.md',
    'module-map',
    table(
      ['aifn entry point', 'omai modules searched'],
      aliases.modules.map((m) => [`\`${m.aifn}\``, m.omai.map((o) => `\`${o.replace('omai-', '')}\``).join(', ')]),
    ),
  )
  inject('usage.md', 'usage-by-area', areaTable)
  inject('usage.md', 'top-symbols', topTable)
  inject('usage.md', 'import-paths', subTable)
  inject('usage.md', 'worker-calls', workerTable)
  inject('usage.md', 'totals', totalsText(totals))
  console.log(JSON.stringify(totals, null, 1))
  const sugIdx = process.argv.indexOf('--suggest')
  if (sugIdx >= 0) suggest(usedRecs, omai, process.argv[sugIdx + 1] ?? '')
}

/** Print the nearest omai names of each unmatched used symbol whose id starts with `prefix`: a review aid, not an output. */
function suggest(used: readonly Record<string, unknown>[], omai: readonly OmaiSymbol[], prefix: string): void {
  for (const r of used) {
    const id = r['id'] as string
    if (!id.startsWith(prefix) || !['unmatched', 'elsewhere'].includes(r['status'] as string)) continue
    const near = nearestNames(id.slice(id.indexOf('#') + 1), omai, 0.4)
    console.log(
      `${id} (${r['notes']}) ${r['kind']} -> ${near.map((x) => `${x.item.module.replace('omai-engine/', '')}#${x.item.name}`).join(' | ')}`,
    )
  }
}

function totalsText(t: Record<string, unknown>): string {
  const g = (k: string): number => (t[k] as number) ?? 0
  const s = t['usedByNotesValuesStatus'] as Record<string, number>
  const covered = (s['exact'] ?? 0) + (s['alias'] ?? 0) + (s['design'] ?? 0) + (s['internal'] ?? 0)
  return [
    `- Notes scanned: ${g('notesScanned')}; notes that reach the aifn packages: ${g('notesUsingAifn')}.`,
    `- Distinct aifn declarations used by notes: ${g('usedByNotes')} (${g('usedByNotesValues')} values, the rest types).`,
    `- Values with a counterpart in omai (same name, renamed, or a differently shaped API): ${covered} of ${g('usedByNotesValues')} (${pct(covered, g('usedByNotesValues'))}).`,
  ].join('\n')
}

/** Replace what sits between `<!-- generated:name -->` and `<!-- /generated:name -->` in a doc. */
function inject(file: string, name: string, text: string): void {
  const path = join(DOCS, file)
  if (!existsSync(path)) return
  const src = readFileSync(path, 'utf8')
  const open = `<!-- generated:${name} -->`
  const close = `<!-- /generated:${name} -->`
  const a = src.indexOf(open)
  const b = src.indexOf(close)
  if (a < 0 || b < 0) return
  writeFileSync(path, `${src.slice(0, a + open.length)}\n${text}\n${src.slice(b)}`)
}

await main()
