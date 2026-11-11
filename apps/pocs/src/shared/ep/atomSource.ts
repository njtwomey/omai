import type { AtomProblem, EpAtom } from 'omai-engine/probprog/ep'
import { snippetOf } from '../../../../site/src/shell/snippet.ts'

/** The atoms' source files, as text (`?raw`): the code shown is the code that runs. */
const SOURCES = import.meta.glob(['../../../../../packages/engine/src/lib/probprog/ep/ep-atoms*.ts', '!**/*.test.ts'], {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

/** The data a problem binds, as it would be written: a line per key, a long array summarised by its length. */
function dataText(data: AtomProblem['data']): string {
  const round = (v: unknown): unknown =>
    typeof v === 'number' ? Number(v.toPrecision(4)) : Array.isArray(v) ? v.map(round) : v
  const entries = Object.entries(data as Record<string, unknown>).map(([k, v]) => {
    const text = JSON.stringify(round(v)).replace(/,/g, ', ')
    return `  ${k}: ${text.length > 70 && Array.isArray(v) ? `[... ${v.length} entries]` : text},`
  })
  return `{\n${entries.join('\n')}\n}`
}

/**
 * The code of an atom's page, assembled from its own source: the region of the atoms' files marked with its name (its model),
 * the data the page binds, and the lines of the scaffold's `atomRun` marked `inference` (compile, EP, read the beliefs). Empty
 * when the atom has no region.
 *
 * @param atom - The atom.
 * @param problem - The problem its knobs made (for the data).
 * @returns The code, and the file its model is in.
 */
export function atomSource(
  atom: EpAtom,
  problem: AtomProblem | undefined,
): { readonly code: string; readonly file: string } {
  let model = ''
  let file = ''
  for (const [path, text] of Object.entries(SOURCES)) {
    const found = path.endsWith('/ep-atoms.ts') ? '' : snippetOf(text, { region: atom.name })
    if (found.trim() !== '') {
      model = found
      file = path.split('/').pop() as string
      break
    }
  }
  if (model === '') return { code: '', file: '' }
  const scaffold = Object.entries(SOURCES).find(([path]) => path.endsWith('/ep-atoms.ts'))?.[1] ?? ''
  const inference = snippetOf(scaffold, { region: 'inference' })
  const data = problem === undefined ? '' : `\n// the data the page binds\nconst data = ${dataText(problem.data)}\n`
  return { code: `${model}${data}\n// what the page runs on it\n${inference}`, file }
}
