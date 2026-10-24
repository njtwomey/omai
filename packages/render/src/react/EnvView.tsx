import { type ReactElement, type ReactNode, useMemo } from 'react'
import type { Env } from 'whitebox-engine/rl'
import type { Scene } from 'whitebox-engine/scene'
import { cx } from './controls/classes.ts'
import { SceneView } from './SceneView.tsx'

/** Props of {@link EnvView}. */
export interface EnvViewProps<S, O, A> {
  readonly env: Env<S, O, A>
  /** The current state (the page keeps it, so other figures can read it). */
  readonly state: S
  /** Drawn width (default the frame's own). */
  readonly width?: number
  /** Draw something over the frame. */
  readonly decorate?: (frame: Scene, state: S) => Scene
  /** Content beside the frame (controls, readouts). */
  readonly children?: ReactNode
  readonly className?: string
}

/**
 * One consistent view of any environment's state: its `render` as a scene, optionally decorated, with room for
 * controls. Drive it with {@link useEnvStepper} to reset and step by hand.
 *
 * @param props - The environment, its state, width, decoration and extra content.
 * @returns The view.
 */
export function EnvView<S, O, A>(props: EnvViewProps<S, O, A>): ReactElement {
  const { env, state, width, decorate, children, className } = props
  const frame = useMemo(() => {
    const f = env.render(state)
    return decorate === undefined ? f : decorate(f, state)
  }, [env, state, decorate])
  return (
    <div className={cx('flex flex-wrap items-start gap-4', className)}>
      <SceneView scene={frame} {...(width === undefined ? {} : { width })} />
      {children}
    </div>
  )
}
