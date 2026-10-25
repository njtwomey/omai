/**
 * The precompiled docs catalogue as files (`apps/site/plugins/docs-data.ts` writes them, the docs pages fetch them):
 * a small index for the tree, a search index fetched when someone searches, one file per source file with its full
 * symbol docs, and one small file per module with its types' declarations, for the hover on a type name. Every file is named by a hash of its content, so a browser may keep it forever.
 */
import type { PackageDoc, SymbolDoc, SymbolKind } from 'whitebox-meta/schema'

/** The URL folder (below the site's base) the files are served from. */
export const DOCS_DATA = 'docs-data/'

/** A source file in the index: path below `packages/engine/src`, entry point, opening sentence, data file, exports. */
export interface WireFile {
  readonly path: string
  readonly module: string
  readonly summary: string
  /** The name of its data file (`files/<hash>.json`, a {@link WireFileData}). */
  readonly data: string
  /** Its exports in source order, as `[name, kind]`. */
  readonly symbols: readonly (readonly [string, SymbolKind])[]
}

/** A folder with its summary (from the doc comment that opens its `index.ts`); an entry point's names its module. */
export interface WireFolder {
  readonly path: string
  readonly module?: string
  readonly summary: string
  /** The name of its types file (`types/<hash>.json`, a {@link WireTypes}), when it exports types. */
  readonly types?: string
}

/** The type names a file's docs use, each resolved (through the file's imports) to `[module, name]`. */
export type WireRefs = Readonly<Record<string, readonly [string, string]>>

/** A type as its hover shows it. */
export interface WireType {
  /** Its file, below `packages/engine/src`. */
  readonly file: string
  readonly kind: SymbolKind
  /** Its declaration (long ones cut). */
  readonly signature: string
  /** Its one-sentence summary. */
  readonly summary: string
}

/** The types (interfaces, type aliases, classes, enums) one module exports, by name. */
export type WireTypes = Readonly<Record<string, WireType>>

/** The index: what the tree, folder pages, file headers and `{@link}`s need. */
export interface WireIndex {
  /** A hash of the whole catalogue. */
  readonly version: string
  /** The name of the search index (a {@link WireSearch}). */
  readonly search: string
  readonly packages: readonly PackageDoc[]
  readonly folders: readonly WireFolder[]
  readonly files: readonly WireFile[]
}

/** The one-sentence summary of every export, parallel to the index: `symbols[i][j]` for `files[i].symbols[j]`. */
export interface WireSearch {
  readonly symbols: readonly (readonly string[])[]
}

/** One source file's documentation: its full opening comment and its exports' docs in source order. */
export interface WireFileData {
  readonly description: string
  readonly symbols: readonly SymbolDoc[]
  /** The type names its docs use, resolved. */
  readonly refs: WireRefs
}

/** What the generator writes last, naming the current index. */
export interface WireManifest {
  readonly version: string
  readonly index: string
  /** Every file of this version (index, search and data files), relative to the output folder. */
  readonly files: readonly string[]
}
