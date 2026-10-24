/**
 * The code shown beside a recipe: the recipe file's own source (imported with `?raw`, so the code and the figure never
 * drift apart), trimmed to what a reader needs.
 *
 * - `export const recipe = { ... }` (the gallery's metadata) is dropped, with the import of its type.
 * - With `// region` ... `// endregion` blocks (or `{/* region *\/}` in JSX), only the imports and those blocks are
 *   shown, joined by `// ...`; otherwise the whole file is.
 * - A line ending in `// hide` is dropped.
 *
 * @param source - The recipe file.
 * @returns The snippet.
 */
export function snippetOf(source: string): string {
  const lines = source.replace(/\r\n/g, '\n').split('\n')
  const imports: string[] = []
  let i = 0
  for (; i < lines.length; i++) {
    const line = lines[i] as string
    if (line.trim() === '') continue
    if (!line.startsWith('import ')) break
    const statement = [line]
    while (
      !/from\s+['"][^'"]+['"]\s*;?\s*$|^import\s+['"][^'"]+['"]/.test(statement.at(-1) as string) &&
      i + 1 < lines.length
    )
      statement.push(lines[++i] as string)
    if (!/['"][^'"]*\/recipe(\.ts)?['"]/.test(statement.join('\n'))) imports.push(...statement)
  }
  let rest = lines.slice(i)
  const start = rest.findIndex((l) => /^export const recipe\b/.test(l))
  if (start >= 0) {
    let end = start
    while (end < rest.length && !/^}/.test(rest[end] as string)) end++
    rest = [...rest.slice(0, start), ...rest.slice(end + 1)]
  }
  rest = rest.filter((l) => !/\/\/\s*hide\s*$/.test(l))
  const regions: string[][] = []
  let open: string[] | null = null
  for (const l of rest) {
    if (marker(l, 'region')) open = []
    else if (marker(l, 'endregion')) {
      if (open !== null) regions.push(open)
      open = null
    } else open?.push(l)
  }
  const body = regions.length > 0 ? regions.map(dedent).join('\n\n// ...\n\n') : rest.join('\n')
  return tidy([imports.join('\n'), body].filter((s) => s.trim() !== '').join('\n\n'))
}

const marker = (line: string, word: string): boolean =>
  new RegExp(`^\\s*(//\\s*${word}\\b.*|\\{/\\*\\s*${word}\\b.*\\*/\\})\\s*$`).test(line)

function dedent(block: readonly string[]): string {
  const indents = block.filter((l) => l.trim() !== '').map((l) => (/^ */.exec(l)?.[0] ?? '').length)
  const cut = indents.length > 0 ? Math.min(...indents) : 0
  return block.map((l) => l.slice(cut)).join('\n')
}

const tidy = (s: string): string => `${s.replace(/\n{3,}/g, '\n\n').trim()}\n`
