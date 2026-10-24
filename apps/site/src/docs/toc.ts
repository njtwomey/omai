/**
 * The docs table of contents: the engine's source tree as it is on disk. `whitebox-engine`, then `lib` and `methods`,
 * then each folder's subfolders and files (alphabetical), then each file's exports in source order. Names are the
 * folder and file names.
 */
import { type Docs, docsPath, symbolPath } from './data.ts'

/** A node of the table of contents. */
export interface TocNode {
  readonly id: string
  readonly title: string
  readonly path?: string
  readonly children?: readonly TocNode[]
}

function folderNode(docs: Docs, path: string): TocNode {
  const folder = docs.folder(path)
  const children: TocNode[] = [
    ...(folder?.folders ?? []).map((p) => folderNode(docs, p)),
    ...(folder?.files ?? []).map((f) => ({
      id: f.path,
      title: f.path.slice(f.path.lastIndexOf('/') + 1),
      path: docsPath(f.path),
      ...(f.symbols.length === 0
        ? {}
        : {
            children: f.symbols.map((s) => ({
              id: `${f.path}#${s.name}`,
              title: s.name,
              path: symbolPath(f.path, s.name),
            })),
          }),
    })),
  ]
  return {
    id: path === '' ? 'whitebox-engine' : path,
    title: folder?.name ?? path,
    path: docsPath(path),
    ...(children.length > 0 ? { children } : {}),
  }
}

/**
 * The tree, rooted at the package.
 *
 * @param docs - The docs index.
 * @returns One root node, `whitebox-engine`.
 */
export function docsToc(docs: Docs): TocNode[] {
  return [folderNode(docs, '')]
}
