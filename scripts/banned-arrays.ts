/**
 * The rule behind the hand-built arrays check of `scripts/banned.ts`, as pure functions so it can be tested on fixture text
 * (`scripts/banned-arrays.test.ts`). Numeric data is created and transformed with the tensor library, never by hand
 * (`new Array<number>(n).fill(c)`, `Array.from({ length: n }, ...)`, a `Float64Array` as a value container, a loop that
 * fills an array element by element). The scan flags, in the EP code paths (`ARRAY_PATHS`):
 *
 * - `new Array(` and `new Array<T>(`;
 * - `.fill(`;
 * - `Array.from({` (an array built from a length);
 * - `new Float64Array(` and the other typed arrays (`Float32`, `Int32`, `Int16`, `Int8`, `Uint32`, `Uint16`, `Uint8`).
 *
 * In every piece of code a page shows (a `// region` ... `// endregion` block in the sources of `SHOWN_ROOTS`, the code the
 * owner reads) it also flags `Array.from(`, `.push(` and `for (` loops: shown code reads as tensor code.
 *
 * An occurrence that is not a hand-built numeric array, or that is a kernel's working buffer, is allowed by a marker
 * comment on the line above it (or at the end of its own line): `// banned-ok: <category>: <reason>`, with a category from
 * `CATEGORIES` and a reason in words. A marker covers the next code line and the lines directly after it that are flagged
 * too (a run of buffers). Shown code carries no marker that would show up on the page: its marker ends with ` // hide`,
 * which the pages drop. A marker that covers nothing is reported as stale.
 */

/** The code paths the scan covers, as path prefixes relative to the repository root (the test files in them are skipped). */
export const ARRAY_PATHS: readonly string[] = [
  'packages/engine/src/lib/probprog/',
  'packages/engine/src/lib/message-passing/',
  'apps/pocs/src/examples/',
  'apps/pocs/src/pocs/expectation-propagation/',
  'apps/pocs/src/shared/ep/',
]

/** The roots searched for the code a page shows (the files with `// region` markers), as path prefixes. */
export const SHOWN_ROOTS: readonly string[] = ['packages/engine/src/lib/', 'apps/pocs/src/', 'examples/']

/** The categories of a `banned-ok` marker, each with what it covers. */
export const CATEGORIES: Readonly<Record<string, string>> = {
  kernel: 'the working buffer or the input of a performance-critical kernel (quadrature, grids, factorisations)',
  reference: 'a reference implementation written out in plain arithmetic to check the library against',
  'non-numeric': 'a list of things that are not numbers (strings, objects, tensors, specs, factors, labels)',
  sentinel: 'a constant array standing for a value of a type (a sentinel, an empty vector, a lookup table of the API)',
  'per-key draw': 'one draw from a key of its own per element, which a single vectorised draw would change',
  interop: 'the boundary of a third-party or DOM API that wants a plain array or a typed array',
  'TODO tensor': 'a hand-built numeric array that a tensor operation expresses; kept visible until it is converted',
}

/** One occurrence the scan reports. */
export interface ArrayProblem {
  /** 1-based line. */
  readonly line: number
  readonly message: string
}

const TYPED = 'Float64|Float32|Int32|Int16|Int8|Uint32|Uint16|Uint8Clamped|Uint8'

/** The patterns of the code paths. */
const PATH_RULES: readonly { readonly pattern: RegExp; readonly message: string }[] = [
  { pattern: /\bnew Array\b/, message: 'new Array: build numeric data with zeros/ones/full/arange/linspace' },
  { pattern: /\.fill\(/, message: '.fill(: build numeric data with full/zeros/ones' },
  { pattern: /\bArray\.from\(\s*\{/, message: 'Array.from({ length }): use arange/linspace and tensor ops' },
  {
    pattern: new RegExp(`\\bnew (${TYPED})Array\\(`),
    message: 'a typed array as a value container: use a tensor (or mark a kernel buffer)',
  },
]

/** The extra patterns of shown code. */
const SHOWN_RULES: readonly { readonly pattern: RegExp; readonly message: string }[] = [
  { pattern: /\bArray\.from\(/, message: 'Array.from( in shown code: toFlat(t) already gives a number[]' },
  { pattern: /\.push\(/, message: '.push( in shown code: build the values with tensor ops' },
  { pattern: /\bfor \(/, message: 'a for loop in shown code: express it with tensor ops' },
]

const MARKER = /\/\/\s*banned-ok:\s*(.*?)\s*(\/\/\s*hide\s*)?$/
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

/** Whether a path is one of the guarded code paths and not a test. */
export function guarded(path: string): boolean {
  return ARRAY_PATHS.some((p) => path.startsWith(p)) && !TEST_FILE.test(path) && !path.includes('/testing/')
}

/** Whether a path can hold shown code (a source file under `SHOWN_ROOTS`, not a test). */
export const mayShow = (path: string): boolean => SHOWN_ROOTS.some((p) => path.startsWith(p)) && !TEST_FILE.test(path)

/**
 * The hand-built arrays in a source file.
 *
 * @param path - The path relative to the repository root (decides which rules apply).
 * @param text - The file.
 * @returns The problems, in line order.
 */
export function arrayProblems(path: string, text: string): ArrayProblem[] {
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  const inPaths = guarded(path)
  const region = mayShow(path) ? regionLines(lines) : new Set<number>()
  if (!inPaths && region.size === 0) return []
  const problems: ArrayProblem[] = []
  // The marker comment waiting for its line (its 1-based line), and whether the line before is covered (a run).
  let pending: number | undefined
  let covered = false
  let block = false
  lines.forEach((raw, i) => {
    const n = i + 1
    const trimmed = raw.trim()
    const marker = MARKER.exec(raw)
    if (marker !== null) {
      const body = marker[1] ?? ''
      const category = Object.keys(CATEGORIES).find((c) => body.startsWith(`${c}:`))
      const reason = category === undefined ? '' : body.slice(category.length + 1).trim()
      if (category === undefined)
        problems.push({
          line: n,
          message: `banned-ok needs a category (${Object.keys(CATEGORIES).join(', ')}): ${body}`,
        })
      else if (reason.length < 3) problems.push({ line: n, message: 'banned-ok needs a reason after the category' })
      if (region.has(i) && marker[2] === undefined)
        problems.push({ line: n, message: 'a banned-ok marker inside shown code must end with // hide' })
      // a marker alone on its line, or after the `?` or `:` the formatter puts before a comment in a ternary
      if (/^([?:(&|,]+\s*)?\/\//.test(trimmed)) {
        pending = n
        covered = false
        return
      }
    }
    if (block) {
      if (raw.includes('*/')) block = false
      return
    }
    if (trimmed.startsWith('/*')) {
      block = !raw.includes('*/')
      return
    }
    if (trimmed.startsWith('*') || trimmed.startsWith('//')) return
    if (trimmed === '') {
      if (pending !== undefined)
        problems.push({ line: pending, message: 'banned-ok marker that no flagged line follows' })
      pending = undefined
      covered = false
      return
    }
    const code = codeOf(raw)
    const hits: string[] = []
    if (inPaths) for (const r of PATH_RULES) if (r.pattern.test(code)) hits.push(r.message)
    // a hidden line is not shown: only the lines of a region that the pages keep
    if (region.has(i) && !HIDE.test(raw)) for (const r of SHOWN_RULES) if (r.pattern.test(code)) hits.push(r.message)
    if (hits.length === 0) {
      if (pending !== undefined)
        problems.push({ line: pending, message: 'banned-ok marker that no flagged line follows' })
      if (marker !== null) problems.push({ line: n, message: 'banned-ok marker on a line that is not flagged' })
      pending = undefined
      covered = false
      return
    }
    const ok = marker !== null || pending !== undefined || covered
    if (!ok) for (const m of hits) problems.push({ line: n, message: m })
    covered = ok
    pending = undefined
  })
  if (pending !== undefined) problems.push({ line: pending, message: 'banned-ok marker that no flagged line follows' })
  return problems.sort((a, b) => a.line - b.line)
}
