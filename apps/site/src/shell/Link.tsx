import type { MouseEvent, ReactNode } from 'react'
import { href, navigate } from './router.ts'

/** An in-site link: a real anchor (open in a new tab works) that navigates without reloading on a plain click. */
export function Link(props: {
  readonly to: string
  readonly className?: string
  readonly children: ReactNode
}): ReactNode {
  const onClick = (e: MouseEvent<HTMLAnchorElement>): void => {
    if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return
    e.preventDefault()
    navigate(props.to)
  }
  return (
    <a href={href(props.to)} onClick={onClick} className={props.className}>
      {props.children}
    </a>
  )
}
