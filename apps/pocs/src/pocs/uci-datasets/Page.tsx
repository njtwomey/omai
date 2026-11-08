import { type DatasetEntry, datasetRegistry, loadDataset } from 'omai-engine/datasets'
import { type ReactNode, useDeferredValue, useMemo, useState } from 'react'
import { type Frame, type FrameColumn, summarise, toFrame } from './frame.ts'

// The classic datasets from the UCI repository: vendored, real, and sourced from it.
const isUci = (e: DatasetEntry): boolean =>
  e.form === 'vendored' &&
  e.real &&
  /UCI (Machine Learning Repository|Statlog)/.test(`${e.source ?? ''} ${e.licence ?? ''}`)

const ENTRIES = datasetRegistry.entries.filter(isUci).sort((a, b) => a.title.localeCompare(b.title))
const PAGE = 100

export default function UciDatasetsPage(): ReactNode {
  const [query, setQuery] = useState('')
  const [name, setName] = useState<string>(ENTRIES[0]?.name ?? '')
  const q = useDeferredValue(query)
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w !== '')
  const shown = ENTRIES.filter((e) =>
    q.every((w) => `${e.title} ${e.description} ${e.kind} ${e.name}`.toLowerCase().includes(w)),
  )
  const entry = ENTRIES.find((e) => e.name === name)
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[18rem_minmax(0,1fr)]">
      <nav className="flex flex-col gap-2">
        <input
          type="search"
          aria-label="Search the datasets"
          placeholder={`Search ${ENTRIES.length} datasets`}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="h-9 rounded-md border border-line bg-bg px-3 text-sm text-fg outline-none focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50"
        />
        <ul className="flex max-h-[70vh] flex-col overflow-y-auto rounded-md border border-line">
          {shown.map((e) => (
            <li key={e.name}>
              <button
                type="button"
                onClick={() => setName(e.name)}
                className={`flex w-full cursor-pointer items-baseline justify-between gap-2 border-0 px-3 py-1.5 text-left text-sm ${e.name === name ? 'bg-surface font-medium text-fg' : 'bg-transparent text-muted hover:bg-surface hover:text-fg'}`}
              >
                <span className="truncate">{e.title}</span>
                <span className="shrink-0 text-xs">{e.size ?? ''}</span>
              </button>
            </li>
          ))}
          {shown.length === 0 && <li className="px-3 py-2 text-sm text-muted">Nothing matches.</li>}
        </ul>
      </nav>
      {entry === undefined ? null : <Dataset key={entry.name} entry={entry} />}
    </div>
  )
}

function Dataset(props: { readonly entry: DatasetEntry }): ReactNode {
  const { entry } = props
  const frame = useMemo(() => {
    try {
      return toFrame(loadDataset(entry).data)
    } catch {
      return undefined
    }
  }, [entry])
  return (
    <div className="flex min-w-0 flex-col gap-6">
      <Provenance entry={entry} frame={frame} />
      {frame === undefined ? (
        <p className="text-sm text-muted">This dataset is not a table of columns, so there is nothing to list.</p>
      ) : (
        <>
          <Columns frame={frame} />
          <Rows frame={frame} />
        </>
      )}
    </div>
  )
}

function Provenance(props: { readonly entry: DatasetEntry; readonly frame: Frame | undefined }): ReactNode {
  const { entry, frame } = props
  const meta = frame?.meta
  const missing = frame === undefined ? 0 : frame.columns.reduce((n, c) => n + summarise(c).missing, 0)
  const facts: [string, string][] = [
    ['rows', String(frame?.rows ?? entry.size ?? '')],
    ['columns', frame === undefined ? '' : String(frame.columns.length)],
    ['missing cells', frame === undefined ? '' : String(missing)],
    ['task', entry.kind],
  ]
  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-lg font-semibold tracking-tight">{entry.title}</h2>
      <p className="max-w-3xl text-sm text-muted">{entry.description}</p>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
        {facts.map(([k, v]) => (
          <div key={k}>
            <dt className="text-xs text-muted">{k}</dt>
            <dd className="font-medium">{v}</dd>
          </div>
        ))}
      </dl>
      <dl className="flex max-w-3xl flex-col gap-1 text-sm">
        {meta?.citation && <Row k="citation" v={meta.citation} />}
        {(meta?.licence ?? entry.licence) && <Row k="licence" v={(meta?.licence ?? entry.licence) as string} />}
        {(meta?.source ?? entry.source) && <Row k="source" v={(meta?.source ?? entry.source) as string} />}
        {meta?.url && (
          <div className="flex gap-3">
            <dt className="w-20 shrink-0 text-xs text-muted">page</dt>
            <dd className="min-w-0 truncate">
              <a className="underline" href={meta.url} target="_blank" rel="noreferrer">
                {meta.url}
              </a>
            </dd>
          </div>
        )}
        {meta?.sha256 && <Row k="sha-256" v={meta.sha256} mono />}
      </dl>
    </section>
  )
}

function Row(props: { readonly k: string; readonly v: string; readonly mono?: boolean }): ReactNode {
  return (
    <div className="flex gap-3">
      <dt className="w-20 shrink-0 text-xs text-muted">{props.k}</dt>
      <dd className={`min-w-0 break-words ${props.mono ? 'font-mono text-xs' : ''}`}>{props.v}</dd>
    </div>
  )
}

const fmt = (v: number): string =>
  Number.isInteger(v) ? String(v) : Math.abs(v) >= 1000 ? v.toFixed(0) : Number(v.toPrecision(4)).toString()

function Columns(props: { readonly frame: Frame }): ReactNode {
  return (
    <section className="flex flex-col gap-2">
      <h3 className="text-sm font-medium">Columns</h3>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {props.frame.columns.map((c) => (
          <ColumnCard key={c.name} column={c} />
        ))}
      </div>
    </section>
  )
}

function ColumnCard(props: { readonly column: FrameColumn }): ReactNode {
  const { column } = props
  const s = useMemo(() => summarise(column), [column])
  const bars = (s.bins ?? []).map((count, i) => ({ id: `${column.name}/${i}`, count }))
  const peak = Math.max(1, ...(s.bins ?? []), ...(s.levels?.map((l) => l[1]) ?? []))
  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-line p-3">
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-sm font-medium">{column.name}</span>
        <span className="shrink-0 text-xs text-muted">
          {column.target ? 'target · ' : ''}
          {column.kind === 'numeric' ? 'number' : 'label'}
          {s.missing > 0 ? ` · ${s.missing} missing` : ''}
        </span>
      </div>
      {s.bins !== undefined && s.range !== undefined && (
        <>
          <div className="flex h-10 items-end gap-px" role="img" aria-label={`histogram of ${column.name}`}>
            {bars.map((bar) => (
              <div
                key={bar.id}
                className="flex-1 rounded-t-sm bg-(--wb-accent)/70"
                style={{ height: `${(100 * bar.count) / peak}%` }}
              />
            ))}
          </div>
          <div className="flex justify-between text-xs text-muted">
            <span>{fmt(s.range[0])}</span>
            <span>mean {fmt(s.range[1])}</span>
            <span>{fmt(s.range[2])}</span>
          </div>
        </>
      )}
      {s.levels !== undefined && (
        <div className="flex flex-col gap-0.5 text-xs">
          {s.levels.map(([level, count]) => (
            <div key={level} className="flex items-center gap-2">
              <span className="w-24 shrink-0 truncate text-muted">{level}</span>
              <div
                className="h-2 rounded-sm bg-(--wb-accent)/70"
                style={{ width: `${(100 * count) / peak}%`, maxWidth: '60%' }}
              />
              <span className="text-muted">{count}</span>
            </div>
          ))}
          {(s.levelCount ?? 0) > s.levels.length && (
            <span className="text-muted">and {(s.levelCount ?? 0) - s.levels.length} more levels</span>
          )}
        </div>
      )}
    </div>
  )
}

function Rows(props: { readonly frame: Frame }): ReactNode {
  const { frame } = props
  const [shown, setShown] = useState(PAGE)
  const last = Math.min(shown, frame.rows)
  return (
    <section className="flex min-w-0 flex-col gap-2">
      <h3 className="text-sm font-medium">
        Rows 1 to {last} of {frame.rows}
      </h3>
      <div className="max-h-[60vh] overflow-auto rounded-md border border-line">
        <table className="w-full border-collapse text-left text-xs">
          <thead className="sticky top-0 bg-surface text-muted">
            <tr>
              <th className="px-2 py-1 font-normal">#</th>
              {frame.columns.map((c) => (
                <th key={c.name} className={`whitespace-nowrap px-2 py-1 font-normal ${c.target ? 'text-fg' : ''}`}>
                  {c.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: last }, (_, k) => k).map((i) => (
              <tr key={i} className="border-t border-line">
                <td className="px-2 py-0.5 text-muted">{i + 1}</td>
                {frame.columns.map((c) => {
                  const v = c.values[i]
                  const missing = v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v))
                  return (
                    <td
                      key={c.name}
                      className={`whitespace-nowrap px-2 py-0.5 tabular-nums ${missing ? 'text-muted' : ''}`}
                    >
                      {missing ? '·' : typeof v === 'number' ? fmt(v) : v}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {last < frame.rows && (
        <button
          type="button"
          onClick={() => setShown((n) => n + PAGE)}
          className="w-fit cursor-pointer rounded-md border border-line bg-bg px-3 py-1.5 text-xs text-fg hover:bg-surface"
        >
          Show {Math.min(PAGE, frame.rows - last)} more
        </button>
      )}
    </section>
  )
}
