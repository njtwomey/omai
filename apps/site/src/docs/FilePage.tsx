import type { ReactNode } from 'react'
import { Link } from '../shell/Link.tsx'
import { Crumbs } from './Crumbs.tsx'
import { DocText } from './DocText.tsx'
import { baseName, type Docs, type IndexFile, symbolPath, useFileSymbols } from './data.ts'
import { SymbolBody } from './SymbolPage.tsx'

/** A source file: its opening comment, an index of its exports, then each export's documentation in source order. */
export function FilePage(props: { readonly docs: Docs; readonly file: IndexFile }): ReactNode {
  const { docs, file } = props
  const symbols = useFileSymbols(file)
  const prose = [file.summary, file.description].filter((t) => t !== '').join('\n\n')
  return (
    <div className="space-y-6">
      <header>
        <Crumbs path={file.path} />
        <h1 className="font-mono text-2xl font-semibold">{baseName(file.path)}</h1>
        <p className="mt-1 font-mono text-xs text-neutral-500">
          import {'{ … }'} from '{file.module}'
        </p>
        <p className="mt-2 flex flex-wrap gap-x-3 gap-y-1 font-mono text-xs">
          {file.symbols.map((s) => (
            <Link
              key={s.name}
              to={symbolPath(file.path, s.name)}
              className="text-sky-700 hover:underline dark:text-sky-400"
            >
              {s.name}
            </Link>
          ))}
        </p>
      </header>
      {prose === '' ? null : <DocText text={prose} docs={docs} near={file.module} />}
      {file.symbols.length === 0 ? <p className="text-sm text-neutral-500">No public exports.</p> : null}
      {symbols === null ? (
        file.symbols.length === 0 ? null : (
          <p className="text-sm text-neutral-500">Loading…</p>
        )
      ) : (
        <div className="space-y-12">
          {symbols.map((s) => (
            <section key={s.id} className="space-y-3 border-t border-neutral-200 pt-4 dark:border-neutral-800">
              <h2 className="font-mono text-lg font-semibold">
                <Link to={symbolPath(file.path, s.name)} className="hover:underline">
                  {s.name}
                </Link>{' '}
                <span className="text-xs font-normal text-neutral-500">{s.kind}</span>
              </h2>
              <SymbolBody docs={docs} symbol={s} />
            </section>
          ))}
        </div>
      )}
    </div>
  )
}
