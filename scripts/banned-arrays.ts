/**
 * The check behind hand-built arrays in SHOWN code, as pure functions so it can be tested on fixture text
 * (`scripts/banned-arrays.test.ts`). The code a page shows (a `// region` ... `// endregion` block in the sources of
 * `SHOWN_ROOTS`) is what the owner reads, so it reads as tensor code: no `new Array`, `.fill(`, `Array.from(`, typed
 * arrays, `.push(` or `for (` loops. A line the page hides (`// hide` at its end) is not shown and is not checked.
 *
 * There is no allow-list and no marker comment: a shown line that trips the check is rewritten with tensor operations.
 * Everywhere else the rule is the one in CLAUDE.md ("No hand-built arrays"), kept by review, not by a scan.
 */

/** The roots searched for the code a page shows (the files with `// region` markers), as path prefixes. */
export const SHOWN_ROOTS: readonly string[] = ['packages/engine/src/lib/', 'apps/pocs/src/', 'examples/']

/** One occurrence the scan reports. */
export interface ArrayProblem {
  /** 1-based line. */
  readonly line: number
  readonly message: string
}

const TYPED = 'Float64|Float32|Int32|Int16|Int8|Uint32|Uint16|Uint8Clamped|Uint8'

const SHOWN_RULES: readonly { readonly pattern: RegExp; readonly message: string }[] = [
  { pattern: /\bnew Array\b/, message: 'new Array in shown code: build numeric data with zeros/ones/full/arange' },
  { pattern: /\.fill\(/, message: '.fill( in shown code: build numeric data with full/zeros/ones' },
  {
    pattern: /\bArray\.from\(/,
    message: 'Array.from( in shown code: use tensor ops (toFlat(t) is already a number[])',
  },
  { pattern: new RegExp(`\\bnew (${TYPED})Array\\(`), message: 'a typed array in shown code: use a tensor' },
  { pattern: /\.push\(/, message: '.push( in shown code: build the values with tensor ops' },
  { pattern: /\bfor \(/, message: 'a for loop in shown code: express it with tensor ops' },
]

const HIDE = /\/\/\s*hide\s*$/
const TEST_FILE = /\.test(-d)?\.tsx?$/

/** The source of a line with its strings emptied and its comment cut, so that only code is matched. */
function codeOf(line: string): string {
  const noStrings = line.replace(/'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`/g, '""')
  const comment = noStrings.indexOf('//')
  return comment >= 0 ? noStrings.slice(0, comment) : noStrings
}

const regionMarker = (line: string, word: string): boolean =>
  new RegExp(`^\\s*(//\\s*${word}\\b.*|\\{/\\*\\s*${word}\\b.*\\*/\\})\\s*$`).test(line)

/** The 0-based indices of the lines inside a `// region` ... `// endregion` block (the markers themselves excluded). */
export function regionLines(lines: readonly string[]): ReadonlySet<number> {
  const inside = new Set<number>()
  let open = false
  lines.forEach((line, i) => {
    if (regionMarker(line, 'region')) open = true
    else if (regionMarker(line, 'endregion')) open = false
    else if (open) inside.add(i)
  })
  return inside
}

/** Whether a path can hold shown code (a source file under `SHOWN_ROOTS`, not a test). */
export const mayShow = (path: string): boolean => SHOWN_ROOTS.some((p) => path.startsWith(p)) && !TEST_FILE.test(path)

/**
 * The hand-built arrays in the shown code of a source file.
 *
 * @param path - The path relative to the repository root (decides whether the file can hold shown code).
 * @param text - The file.
 * @returns The problems, in line order.
 */
export function arrayProblems(path: string, text: string): ArrayProblem[] {
  if (!mayShow(path)) return []
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  const region = regionLines(lines)
  const problems: ArrayProblem[] = []
  lines.forEach((raw, i) => {
    if (!region.has(i) || HIDE.test(raw)) return
    const trimmed = raw.trim()
    if (trimmed === '' || trimmed.startsWith('//') || trimmed.startsWith('*') || trimmed.startsWith('/*')) return
    const code = codeOf(raw)
    for (const r of SHOWN_RULES) if (r.pattern.test(code)) problems.push({ line: i + 1, message: r.message })
  })
  return problems
}
