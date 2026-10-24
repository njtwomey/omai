import type { ReactNode } from 'react'
import type { Example, MemberDoc, ModuleDoc, ParamDoc, SymbolDoc } from 'whitebox-meta/schema'
import { DocInline, DocText } from './DocText.tsx'
import type { Docs } from './data.ts'
import { Breadcrumbs } from './ModulePage.tsx'

function Code(props: { readonly code: string }): ReactNode {
  return (
    <pre className="overflow-x-auto rounded border border-neutral-200 bg-neutral-50 p-3 font-mono text-xs leading-relaxed dark:border-neutral-800 dark:bg-neutral-900">
      {props.code}
    </pre>
  )
}

function Params(props: {
  readonly params: readonly ParamDoc[]
  readonly docs: Docs
  readonly near: string
}): ReactNode {
  if (props.params.length === 0) return null
  return (
    <table className="w-full text-left text-sm">
      <thead className="text-neutral-500">
        <tr>
          <th className="py-1 pr-4 font-medium">Parameter</th>
          <th className="py-1 pr-4 font-medium">Type</th>
          <th className="py-1 font-medium">Description</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-neutral-200 dark:divide-neutral-800">
        {props.params.map((p) => (
          <tr key={p.name} className="align-top">
            <td className="py-1.5 pr-4 font-mono whitespace-nowrap">
              {p.name}
              {p.optional ? <span className="text-neutral-500">?</span> : null}
            </td>
            <td className="py-1.5 pr-4 font-mono text-xs text-neutral-600 dark:text-neutral-400">
              {p.type}
              {p.defaultValue === '' ? null : <span className="text-neutral-500"> = {p.defaultValue}</span>}
            </td>
            <td className="py-1.5">
              <DocInline text={p.description} docs={props.docs} near={props.near} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Examples(props: { readonly examples: readonly Example[] }): ReactNode {
  if (props.examples.length === 0) return null
  return (
    <div className="space-y-3">
      {props.examples.map((e, i) => (
        <div key={`${e.source?.line ?? i}`}>
          {e.title === '' ? null : <h4 className="mb-1 text-sm font-medium">{e.title}</h4>}
          <Code code={e.code} />
        </div>
      ))}
    </div>
  )
}

function Section(props: { readonly title: string; readonly children: ReactNode }): ReactNode {
  return (
    <section>
      <h2 className="mb-2 text-lg font-semibold">{props.title}</h2>
      {props.children}
    </section>
  )
}

function Member(props: { readonly member: MemberDoc; readonly docs: Docs; readonly near: string }): ReactNode {
  const m = props.member
  return (
    <div className="space-y-2 border-t border-neutral-200 pt-3 dark:border-neutral-800">
      <h3 className="font-mono text-sm font-semibold">
        {m.static ? 'static ' : ''}
        {m.name}
      </h3>
      <Code code={m.signature} />
      {m.summary === '' ? null : (
        <DocText
          text={[m.summary, m.description].filter((t) => t !== '').join('\n\n')}
          docs={props.docs}
          near={props.near}
          className="space-y-2 text-sm"
        />
      )}
      <Params params={m.params} docs={props.docs} near={props.near} />
      {m.returns === '' ? null : (
        <p className="text-sm">
          <span className="font-medium">Returns</span>{' '}
          <DocInline text={m.returns} docs={props.docs} near={props.near} />
        </p>
      )}
      <Examples examples={m.examples} />
    </div>
  )
}

/** A symbol: signature, doc comment, parameters, returns, throws, examples, members and where it is defined. */
export function SymbolPage(props: {
  readonly docs: Docs
  readonly module: ModuleDoc
  readonly symbol: SymbolDoc
}): ReactNode {
  const { docs, module, symbol: s } = props
  const near = module.id
  return (
    <article className="space-y-6">
      <header>
        <Breadcrumbs docs={docs} module={module} last={s.name} />
        <h1 className="font-mono text-2xl font-semibold">
          {s.name} <span className="text-sm font-normal text-neutral-500">{s.kind}</span>
        </h1>
        {s.deprecated === undefined ? null : (
          <p className="mt-2 text-sm text-amber-700 dark:text-amber-400">Deprecated: {s.deprecated}</p>
        )}
      </header>
      <Code code={s.signature} />
      <DocText text={[s.summary, s.description].filter((t) => t !== '').join('\n\n')} docs={docs} near={near} />
      {s.params.length === 0 ? null : (
        <Section title="Parameters">
          <Params params={s.params} docs={docs} near={near} />
        </Section>
      )}
      {s.returns === '' ? null : (
        <Section title="Returns">
          <DocText text={s.returns} docs={docs} near={near} />
        </Section>
      )}
      {s.throws.length === 0 ? null : (
        <Section title="Throws">
          <ul className="list-disc pl-5">
            {s.throws.map((t) => (
              <li key={t}>
                <DocInline text={t} docs={docs} near={near} />
              </li>
            ))}
          </ul>
        </Section>
      )}
      {s.examples.length === 0 ? null : (
        <Section title={s.examples.length === 1 ? 'Example' : 'Examples'}>
          <Examples examples={s.examples} />
        </Section>
      )}
      {s.members.length === 0 ? null : (
        <Section title="Members">
          <div className="space-y-4">
            {s.members.map((m) => (
              <Member key={`${m.static ? 's' : ''}${m.kind}:${m.name}`} member={m} docs={docs} near={near} />
            ))}
          </div>
        </Section>
      )}
      <footer className="space-y-1 border-t border-neutral-200 pt-3 text-sm text-neutral-500 dark:border-neutral-800">
        {s.concepts.length === 0 ? null : <p>Concepts: {s.concepts.join(', ')}</p>}
        {s.see.map((x) => (
          <p key={x}>
            See <DocInline text={x} docs={docs} near={near} />
          </p>
        ))}
        {s.references.map((x) => (
          <p key={x}>Reference: {x}</p>
        ))}
        {s.aliases.length === 0 ? null : <p>Also exported as {s.aliases.join(', ')}</p>}
        <p className="font-mono">
          {s.source.file}:{s.source.line}
        </p>
      </footer>
    </article>
  )
}
