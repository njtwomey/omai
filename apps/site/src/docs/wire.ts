/**
 * The precompiled docs catalogue as files (`apps/site/plugins/docs-data.ts` writes them, the docs pages fetch them):
 * a small index for the tree, a search index fetched when someone searches, and one file per source file with its
 * full symbol docs. Every file is named by a hash of its content, so a browser may keep it forever.
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

/** An entry-point folder with its summary. */
export interface WireFolder {
  readonly path: string
  readonly module: string
  readonly summary: string
}

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
}

/** What the generator writes last, naming the current index. */
export interface WireManifest {
  readonly version: string
  readonly index: string
  /** Every file of this version (index, search and data files), relative to the output folder. */
  readonly files: readonly string[]
}
