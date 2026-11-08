/**
 * The agent index: the short list of the main things in the engine library, read from `@index` tags in the source
 * (docs/structure.md section 7). The grammar is `@index <kind> <description>`; the kinds are a closed vocabulary.
 */

/** The kinds an `@index` tag may name, in the order the index lists them. */
export const INDEX_KINDS = [
  'function',
  'estimator',
  'algorithm',
  'structure',
  'table',
  'module',
  'dataset',
  'distribution',
  'kernel',
  'agent',
] as const

/** What an indexed thing is: one of {@link INDEX_KINDS}. */
export type IndexKind = (typeof INDEX_KINDS)[number]

/** A parsed `@index` tag: the kind and a one-line description. */
export interface IndexTag {
  readonly kind: IndexKind
  readonly description: string
}

/** A problem found in an `@index` tag, with where it is. */
export interface IndexProblem {
  readonly severity: 'error' | 'warning'
  /** Repository-relative path. */
  readonly file: string
  /** 1-based line of the tag. */
  readonly line: number
  readonly message: string
}

/** One entry of the index: a file or a symbol, keyed by its import path plus its name. */
export interface IndexEntry {
  readonly kind: IndexKind
  /** The symbol's name, or the file's base name (`blas.ts`) for a file entry. */
  readonly name: string
  /** `file` for a tag in a file's opening paragraph, `symbol` for a tag on a declaration. */
  readonly level: 'file' | 'symbol'
  /** The entry point to import from, such as `omai-engine/numerics/linalg`. */
  readonly importPath: string
  /** The source file, relative to the package root (`src/lib/numerics/linalg/lu.ts`). */
  readonly path: string
  /** 1-based line of the declaration (1 for a file). */
  readonly line: number
  readonly description: string
  /** The first folder under `lib/`, such as `numerics`. */
  readonly domain: string
  /** The file's folder relative to `lib/`, such as `numerics/linalg`. */
  readonly folder: string
}
