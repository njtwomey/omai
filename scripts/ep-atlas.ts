/**
 * The prior atlas (`docs/probprog/ep-priors.md`) as data: its tables are read, each row's `ran by` cell is resolved against the code
 * and the `status` cell is the word the code gives. `scripts/ep-report.ts` writes the status cells (`make ep-report`) and fails when
 * one, or a reference, has drifted from the code (`make ep-report-check`).
 *
 * An atlas row is a row of a table whose header has the columns `ran by` and `status`. The `ran by` cell holds backticked references:
 * `atom:<name>` (an EP atom that must exist and run under EP), `test:<path>` or `test:<path>#<text>` (a test file of the repository
 * that must exist and, with the text, contain it). A row that nothing runs says `not yet` in plain words instead, and its status is
 * `planned`. Every other cell (the `note` among them) is prose and is never touched.
 *
 * Status words: `runs` when every reference holds, `planned` for `not yet`, `broken` when a reference does not hold (an atom that does
 * not exist or does not run, a missing file or text) or the row names nothing at all; a broken row is also a problem of the check.
 */

/** What the atlas needs of the code: which atoms exist and run under EP, and whether a test file holds a text. */
export interface AtlasCode {
  /** The EP atoms by name: true when the coverage says EP runs it. */
  readonly atoms: ReadonlyMap<string, boolean>
  /** Whether the repository file exists and, when `needle` is given, contains it. */
  readonly fileHas: (path: string, needle: string | undefined) => boolean
}

/** The word the code gives a row. */
export type AtlasStatus = 'runs' | 'planned' | 'broken'

/** One row of an atlas table, with the verdict of the code. */
export interface AtlasRow {
  /** One-based line of the file. */
  readonly line: number
  /** The row in words: its first cells, for a message. */
  readonly label: string
  /** The atoms the row names in its `ran by` cell. */
  readonly atoms: readonly string[]
  /** The tests the row names in its `ran by` cell, as written (`path` or `path#text`). */
  readonly tests: readonly string[]
  /** The status cell as it is in the file. */
  readonly written: string
  /** The word the code gives it. */
  readonly status: AtlasStatus
  /** What does not hold, in words: empty for a row that is fine. */
  readonly problems: readonly string[]
}

/** The atlas read against the code: the rows, the text with the status cells set to the code's words, and every disagreement. */
export interface AtlasReading {
  readonly rows: readonly AtlasRow[]
  /** The file with each status cell set to the word the code gives it, and the tables it changed padded to their widest cell. */
  readonly text: string
  /** Disagreements: a reference that does not hold, a row that names nothing, a status word that differs from the code's. */
  readonly problems: readonly string[]
  /** The disagreements that rewriting the status cells does not settle: the ones of `problems` that are not a stale status word. */
  readonly unfixable: readonly string[]
}

const NOT_YET = /\bnot yet\b/i
const SPLIT = /(?<!\\)\|/

const cellsOf = (line: string): string[] =>
  line
    .trim()
    .slice(1, -1)
    .split(SPLIT)
    .map((c) => c.trim())

/** A markdown table from its cells: every column padded to its widest cell, the separator row of dashes. */
function render(rows: readonly (readonly string[])[]): string[] {
  const width = (i: number): number => Math.max(3, ...rows.map((r) => (r[i] ?? '').length))
  const widths = (rows[0] ?? []).map((_, i) => width(i))
  const line = (cells: readonly string[]): string => `| ${cells.map((c, i) => c.padEnd(widths[i] ?? 0)).join(' | ')} |`
  return [line(rows[0] ?? []), line(widths.map((w) => '-'.repeat(w))), ...rows.slice(1).map(line)]
}

/** The references of a `ran by` cell: its atoms and tests (`path` or `path#text`), and whether it says `not yet`. */
function references(cell: string): { atoms: string[]; tests: string[]; notYet: boolean } {
  const spans = [...cell.matchAll(/`([^`]+)`/g)].map((m) => (m[1] as string).trim())
  return {
    atoms: spans.filter((s) => s.startsWith('atom:')).map((s) => s.slice(5).trim()),
    tests: spans.filter((s) => s.startsWith('test:')).map((s) => s.slice(5).trim()),
    notYet: NOT_YET.test(cell.replaceAll(/`[^`]*`/g, '')),
  }
}

function judge(
  ran: string,
  code: AtlasCode,
): { status: AtlasStatus; problems: string[]; atoms: string[]; tests: string[] } {
  const { atoms, tests, notYet } = references(ran)
  const problems: string[] = []
  for (const a of atoms) {
    const runs = code.atoms.get(a)
    if (runs === undefined) problems.push(`names the atom '${a}', which does not exist`)
    else if (!runs) problems.push(`names the atom '${a}', which does not run under EP`)
  }
  for (const t of tests) {
    const [path = '', ...rest] = t.split('#')
    const needle = rest.length === 0 ? undefined : rest.join('#')
    if (!code.fileHas(path, needle)) {
      problems.push(
        needle === undefined
          ? `names the test file '${path}', which does not exist`
          : `names '${path}', which does not hold '${needle}'`,
      )
    }
  }
  if (atoms.length + tests.length === 0 && !notYet) {
    problems.push('names no atom or test in `ran by` and does not say `not yet`')
  }
  if (atoms.length + tests.length > 0 && notYet) problems.push('says `not yet` and also names what runs it')
  const status: AtlasStatus = problems.length > 0 ? 'broken' : notYet ? 'planned' : 'runs'
  return { status, problems, atoms, tests }
}

/**
 * Read the atlas against the code.
 *
 * @param source - The text of `ep-priors.md`.
 * @param code - What the code says: the atoms that run and the test files that hold a text.
 * @returns The rows with their verdicts, the text with the status cells set, and the disagreements.
 */
export function readAtlas(source: string, code: AtlasCode): AtlasReading {
  const lines = source.split('\n')
  const out: string[] = []
  const rows: AtlasRow[] = []
  const problems: string[] = []
  const unfixable: string[] = []
  let i = 0
  while (i < lines.length) {
    if (!(lines[i] as string).startsWith('|')) {
      out.push(lines[i] as string)
      i++
      continue
    }
    let j = i
    while (j < lines.length && (lines[j] as string).startsWith('|')) j++
    const block = lines.slice(i, j)
    const header = cellsOf(block[0] as string)
    const ranBy = header.indexOf('ran by')
    const statusAt = header.indexOf('status')
    if (ranBy < 0 || statusAt < 0) {
      out.push(...block)
      i = j
      continue
    }
    const table: string[][] = [header]
    block.slice(2).forEach((l, k) => {
      const cells = cellsOf(l)
      const lineNo = i + 2 + k + 1
      const label = cells
        .slice(0, 3)
        .map((c) => c.replaceAll('`', ''))
        .join(' / ')
      const verdict = judge(cells[ranBy] ?? '', code)
      const written = (cells[statusAt] ?? '').replaceAll('*', '')
      const own = [...verdict.problems]
      if (written !== verdict.status && verdict.status !== 'broken') {
        own.push(`status says '${written}', the code says '${verdict.status}'`)
      }
      for (const p of verdict.problems) unfixable.push(`ep-priors.md:${lineNo} (${label}): ${p}`)
      for (const p of own) problems.push(`ep-priors.md:${lineNo} (${label}): ${p}`)
      rows.push({ line: lineNo, label, ...verdict, written, problems: own })
      cells[statusAt] = verdict.status
      table.push(cells)
    })
    out.push(...render(table))
    i = j
  }
  return { rows, text: out.join('\n'), problems, unfixable }
}
