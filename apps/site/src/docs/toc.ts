/**
 * The docs table of contents: the engine's source tree as it is on disk. `lib` and `methods`, then each folder's
 * subfolders and files (alphabetical), then each file's exports in source order. Names are the folder and file names,
 * each node marked with what it is (a folder, a file, or the kind of the export); the package's front page
 * (`mdml-engine`) sits above the tree.
 */
import type { TocNode } from '../shell/toc.ts'
import { type Docs, docsPath, symbolPath } from './data.ts'

/** The docs' front page, above the tree. */
export const DOCS_HOME: { readonly title: string; readonly path: string } = { title: 'mdml-engine', path: '/docs' }

function folderNode(docs: Docs, path: string): TocNode {
  const folder = docs.folder(path)
  const children: TocNode[] = [
    ...(folder?.folders ?? []).map((p) => folderNode(docs, p)),
    ...(folder?.files ?? []).map((f) => ({
      id: f.path,
      title: f.path.slice(f.path.lastIndexOf('/') + 1),
      kind: 'file' as const,
      path: docsPath(f.path),
      ...(f.symbols.length === 0
        ? {}
        : {
            children: f.symbols.map((s) => ({
              id: `${f.path}#${s.name}`,
              title: s.name,
              kind: s.kind,
              path: symbolPath(f.path, s.name),
            })),
          }),
    })),
  ]
  return {
    id: path,
    title: folder?.name ?? path,
    kind: 'folder',
    path: docsPath(path),
    ...(children.length > 0 ? { children } : {}),
  }
}

/**
 * The tree: `lib` and `methods` with everything below them.
 *
 * @param docs - The docs index.
 * @returns The top-level nodes.
 */
export function docsToc(docs: Docs): TocNode[] {
  return (docs.folder('')?.folders ?? []).map((p) => folderNode(docs, p))
}
