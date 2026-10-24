/**
 * The `@concept` index: which symbols implement each concept, and which concepts each symbol names. The taxonomy
 * (docs/concepts.yaml) comes later; until then concepts are the ids the doc comments use.
 */
import type { Catalogue } from '../schema/catalogue.ts'

/** Concepts to symbols and back. */
export interface ConceptIndex {
  /** Concept id to the ids of the symbols tagged with it, sorted. */
  readonly symbols: Readonly<Record<string, readonly string[]>>
  /** Symbol id to its concept ids. */
  readonly concepts: Readonly<Record<string, readonly string[]>>
  /** Ids of symbols without a concept. */
  readonly untagged: readonly string[]
}

/**
 * Index the `@concept` tags of a catalogue.
 *
 * @param catalogue - The catalogue.
 * @returns The index.
 */
export function conceptIndex(catalogue: Catalogue): ConceptIndex {
  const symbols: Record<string, string[]> = {}
  const concepts: Record<string, readonly string[]> = {}
  const untagged: string[] = []
  for (const s of catalogue.symbols) {
    if (s.concepts.length === 0) {
      untagged.push(s.id)
      continue
    }
    concepts[s.id] = s.concepts
    for (const c of s.concepts) {
      const list = symbols[c] ?? []
      list.push(s.id)
      symbols[c] = list
    }
  }
  for (const list of Object.values(symbols)) list.sort()
  return { symbols, concepts, untagged }
}
