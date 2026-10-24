/**
 * Examples and what running them reports. An example is the code of one `@example` block; its expectations are the
 * `// =>` markers in that code (docs/structure.md section 2, "Docstrings").
 */

/** Where something sits in the source: a repository-relative file and a 1-based line. */
export interface SourceLocation {
  readonly file: string
  readonly line: number
}

/** One runnable example: the code of an `@example` block, with the title written on the tag's line, if any. */
export interface Example {
  /** The title on the `@example` line (`@example Fitting a line`), or `''`. */
  readonly title: string
  /** The code, without the fence and the comment frame. */
  readonly code: string
  /** Where the code's first line sits in the source, when it came from a file. */
  readonly source?: SourceLocation
}

/** The kinds of `// =>` marker: a value, `figure`, `trace`, or `throws Code`. */
export type ExpectationKind = 'value' | 'figure' | 'trace' | 'throws'

/**
 * One `// =>` marker of an example: the expression it follows (which may span several lines, ending on `line`) and
 * what it should give.
 */
export interface Expectation {
  readonly kind: ExpectationKind
  /** The 0-based line of the example on which the expression starts. */
  readonly startLine: number
  /** The 0-based line of the example that carries the marker's expression end. */
  readonly line: number
  /** The expression's source text (several lines joined by newlines). */
  readonly expression: string
  /** For `value`: the expected value as written (a literal or an expression). For `throws`: the code, or `''`. */
  readonly expected: string
}

/** How one expectation fared. */
export type ExpectationStatus = 'pass' | 'fail' | 'error' | 'skipped'

/** An expectation with its outcome: the value the expression gave, formatted, and why it failed. */
export interface ExpectationResult extends Expectation {
  readonly status: ExpectationStatus
  /** The value the expression gave, formatted for display (or the error it threw). */
  readonly actual: string
  /** Why it failed, when it did. */
  readonly message: string
}

/** An error that stopped an example, with the example line it happened on when that is known. */
export interface ExampleError {
  readonly message: string
  /** The 0-based example line, when the stack names it. */
  readonly line?: number
}

/** What running one example reports. */
export interface ExampleResult {
  /** True when the example ran to the end and every expectation passed. */
  readonly ok: boolean
  readonly expectations: readonly ExpectationResult[]
  /** What the example wrote to its `console`, one entry per call. */
  readonly logs: readonly string[]
  /** The error that stopped the example (a syntax error, a failed import, an unexpected throw, a time-out). */
  readonly error?: ExampleError
  /** Wall time in milliseconds. */
  readonly ms: number
}
