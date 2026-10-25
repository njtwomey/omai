import type { ReactNode } from 'react'
import { Link } from '../shell/Link.tsx'
import { Crumbs } from './Crumbs.tsx'
import { DocText } from './DocText.tsx'
import { baseName, type Docs, type IndexFile, symbolPath, useFileData } from './data.ts'
import { SymbolBody } from './SymbolPage.tsx'
import { RefsContext } from './TypeHover.tsx'

const NO_REFS = {}

/** A source file: its opening comment, an index of its exports, then each export's documentation in source order. */
export function FilePage(props: {
  readonly docs: Docs
  readonly file: IndexFile
  readonly nav?: ReactNode
}): ReactNode {
  const { docs, file } = props
  const data = useFileData(file)
  const symbols = data?.symbols ?? null
  const prose = [file.summary, data?.description ?? ''].filter((t) => t !== '').join('\n\n')
  return (
    <RefsContext.Provider value={data?.refs ?? NO_REFS}>
      <div className="space-y-6">
        <header>
          <div className="flex items-start justify-between gap-4">
            <Crumbs path={file.path} />
            {props.nav}
          </div>
          <h1 className="font-mono text-2xl font-semibold">{baseName(file.path)}</h1>
          <p className="mt-1 font-mono text-xs text-muted">
            import {'{ … }'} from '{file.module}'
          </p>
          <p className="mt-2 flex flex-wrap gap-x-3 gap-y-1 font-mono text-xs">
            {file.symbols.map((s) => (
              <Link key={s.name} to={symbolPath(file.path, s.name)} className="text-accent hover:underline">
                {s.name}
              </Link>
            ))}
          </p>
        </header>
        {prose === '' ? null : <DocText text={prose} docs={docs} near={file.module} />}
        {file.symbols.length === 0 ? <p className="text-sm text-muted">No public exports.</p> : null}
        {symbols === null ? (
          file.symbols.length === 0 ? null : (
            <p className="text-sm text-muted">Loading…</p>
          )
        ) : (
          <div className="space-y-12">
            {symbols.map((s) => (
              <section key={s.id} className="space-y-3 border-t border-line pt-4">
                <h2 className="font-mono text-lg font-semibold">
                  <Link to={symbolPath(file.path, s.name)} className="hover:underline">
                    {s.name}
                  </Link>{' '}
                  <span className="text-xs font-normal text-muted">{s.kind}</span>
                </h2>
                <SymbolBody docs={docs} symbol={s} />
              </section>
            ))}
          </div>
        )}
      </div>
    </RefsContext.Provider>
  )
}
