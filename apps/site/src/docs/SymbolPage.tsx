import type { Example, MemberDoc, ParamDoc, SymbolDoc } from 'omai-meta/schema'
import type { ReactNode } from 'react'
import { CodeBlock } from '../shell/CodeBlock.tsx'
import { highlight } from '../shell/highlight.ts'
import { Crumbs } from './Crumbs.tsx'
import { DocInline, DocText } from './DocText.tsx'
import { baseName, type Docs, type IndexFile } from './data.ts'
import { LiveExample } from './LiveExample.tsx'
import { TypeRef } from './TypeHover.tsx'

/** A signature, its type names linking their pages with their definitions on hover. */
function Code(props: { readonly code: string; readonly docs: Docs; readonly near: string }): ReactNode {
  const { docs, near } = props
  return (
    <CodeBlock
      code={props.code}
      title="signature"
      renderType={(name, key) => <TypeRef key={key} docs={docs} name={name} near={near} code className="tok-type" />}
    />
  )
}

/** A type expression (a parameter's type), highlighted, its type names linked as in signatures. */
function TypeText(props: { readonly text: string; readonly docs: Docs; readonly near: string }): ReactNode {
  const { docs, near } = props
  return (
    <>
      {highlight(props.text).map((t, i) =>
        t.kind === undefined ? (
          t.text
        ) : t.kind === 'type' ? (
          // biome-ignore lint/suspicious/noArrayIndexKey: tokens are a fixed sequence for this text
          <TypeRef key={i} docs={docs} name={t.text} near={near} code className="tok-type" />
        ) : (
          // biome-ignore lint/suspicious/noArrayIndexKey: tokens are a fixed sequence for this text
          <span key={i} className={`tok-${t.kind}`}>
            {t.text}
          </span>
        ),
      )}
    </>
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
      <thead className="text-muted">
        <tr>
          <th className="py-1 pr-4 font-medium">Parameter</th>
          <th className="py-1 pr-4 font-medium">Type</th>
          <th className="py-1 font-medium">Description</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-line">
        {props.params.map((p) => (
          <tr key={p.name} className="align-top">
            <td className="py-1.5 pr-4 font-mono whitespace-nowrap">
              {p.name}
              {p.optional ? <span className="text-muted">?</span> : null}
            </td>
            <td className="site-code py-1.5 pr-4 font-mono text-xs text-fg-secondary">
              <TypeText text={p.type} docs={props.docs} near={props.near} />
              {p.defaultValue === '' ? null : <span className="text-muted"> = {p.defaultValue}</span>}
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
        <LiveExample key={`${e.source?.line ?? i}`} example={e} />
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
    <div className="space-y-2 border-t border-line pt-3">
      <h3 className="font-mono text-sm font-semibold">
        {m.static ? 'static ' : ''}
        {m.name}
      </h3>
      <Code code={m.signature} docs={props.docs} near={props.near} />
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

/** A symbol's documentation: signature, doc comment, parameters, returns, throws, examples, members, source. */
export function SymbolBody(props: { readonly docs: Docs; readonly symbol: SymbolDoc }): ReactNode {
  const { docs, symbol: s } = props
  const near = s.module
  return (
    <div className="space-y-6">
      {s.deprecated === undefined ? null : <p className="text-sm text-danger">Deprecated: {s.deprecated}</p>}
      <Code code={s.signature} docs={docs} near={near} />
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
      <footer className="space-y-1 border-t border-line pt-3 text-sm text-muted">
        <p className="font-mono">
          import {'{'} {s.name} {'}'} from '{s.module}'
        </p>
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
    </div>
  )
}

/** A symbol's page: where it sits in the tree, its name and kind, then its documentation. */
export function SymbolPage(props: {
  readonly docs: Docs
  readonly file: IndexFile
  readonly symbol: SymbolDoc
  readonly nav?: ReactNode
}): ReactNode {
  const { docs, file, symbol } = props
  return (
    <article className="space-y-4">
      <header>
        <div className="flex items-start justify-between gap-4">
          <Crumbs path={file.path} last />
          {props.nav}
        </div>
        <h1 className="font-mono text-2xl font-semibold">
          {symbol.name}{' '}
          <span className="text-sm font-normal text-muted">
            {symbol.kind} in {baseName(file.path)}
          </span>
        </h1>
      </header>
      <SymbolBody docs={docs} symbol={symbol} />
    </article>
  )
}
