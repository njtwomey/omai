/**
 * Recipe metadata without the recipe: a module id ending in `?recipe` (a gallery recipe)
 * loads as only its `export const recipe = { ... }` literal, so the index pages and the contents can list every recipe
 * without importing any recipe's code. The registries glob the files with this query eagerly and import the code itself
 * lazily, when a page is opened.
 */
import { readFileSync } from 'node:fs'
import type { Plugin } from 'vite'

const QUERY = /[?&]recipe(?:&|=|$)/

/**
 * The `recipe` literal of a recipe file's source, as JavaScript: from the line `export const recipe` to the first line
 * that starts with `}` (the format every recipe uses, and the one `snippetOf` strips from the shown code).
 *
 * @param source - The recipe file's source.
 * @param file - The file, for the error.
 * @returns The literal, `{ ... }`.
 */
export function recipeLiteral(source: string, file: string): string {
  const lines = source.replace(/\r\n/g, '\n').split('\n')
  const start = lines.findIndex((l) => /^export const recipe\b/.test(l))
  let end = start
  while (start >= 0 && end < lines.length && !/^}/.test(lines[end] as string)) end++
  if (start < 0 || end >= lines.length) throw new Error(`${file}: no \`export const recipe = { ... }\` block`)
  const text = lines.slice(start, end + 1).join('\n')
  return text
    .replace(/^export const recipe\s*(?::\s*[\w.]+\s*)?=\s*/, '')
    .replace(/\s*satisfies\s+[\w.]+\s*;?\s*$/, '')
    .replace(/;\s*$/, '')
}

/** Serves `<recipe file>?recipe` as just the file's `recipe` export. */
export function recipeMeta(): Plugin {
  return {
    name: 'mdml-recipe-meta',
    enforce: 'pre',
    load(id) {
      const cut = id.indexOf('?')
      if (cut < 0 || !QUERY.test(id.slice(cut))) return null
      const file = id.slice(0, cut)
      this.addWatchFile(file)
      return { code: `export const recipe = ${recipeLiteral(readFileSync(file, 'utf8'), file)}\n`, map: null }
    },
  }
}
