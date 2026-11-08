import type { ReactNode } from 'react'
import { Link } from '../shell/Link.tsx'
import { docsPath } from './data.ts'

/** The folders above a path of the source tree (and, optionally, the path itself), as links. */
export function Crumbs(props: { readonly path: string; readonly last?: boolean }): ReactNode {
  const parts = props.path === '' ? [] : props.path.split('/')
  const upto = props.last === true ? parts.length : parts.length - 1
  return (
    <nav className="mb-2 font-mono text-sm text-muted">
      <Link to="/docs" className="hover:underline">
        omai-engine
      </Link>
      {parts.slice(0, upto).map((p, i) => {
        const at = parts.slice(0, i + 1).join('/')
        return (
          <span key={at}>
            {' / '}
            <Link to={docsPath(at)} className="hover:underline">
              {p}
            </Link>
          </span>
        )
      })}
    </nav>
  )
}
