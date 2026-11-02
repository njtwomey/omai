/**
 * A scene as one framed figure: a `FigureFrame` the size of the scene, shrunk to fit a maximum box while keeping its
 * aspect, holding a `SceneView`, and redrawn at the modal's size when expanded.
 */

import type { ReactElement } from 'react'
import { FigureFrame, type FigureFrameProps } from './FigureFrame.tsx'
import { SceneView, type SceneViewProps } from './SceneView.tsx'

/** Props of {@link SceneFrame}. */
export interface SceneFrameProps
  extends Pick<SceneViewProps, 'scene' | 'mode' | 'onNodeClick' | 'onNodeHover' | 'highlight' | 'children'>,
    Pick<FigureFrameProps, 'title' | 'caption' | 'playback' | 'exportFormats' | 'className'> {
  /** The largest drawn width and height (default 720 by 720); the scene keeps its aspect and never grows. */
  readonly maxWidth?: number
  readonly maxHeight?: number
  /** The export's file name (default `scene`). */
  readonly filename?: string
}

/**
 * The drawn size of a scene in a box: its own size, shrunk (never grown) to fit, keeping its aspect.
 *
 * @param width - The scene's width.
 * @param height - The scene's height.
 * @param maxWidth - The box's width.
 * @param maxHeight - The box's height.
 * @returns The drawn width and height in whole pixels.
 * @example
 * ```ts
 * import { fitScene } from 'mdml-render/react'
 * fitScene(1000, 500, 720, 720) // => { width: 720, height: 360 }
 * fitScene(300, 200, 720, 720) // => { width: 300, height: 200 }
 * ```
 */
export function fitScene(
  width: number,
  height: number,
  maxWidth: number,
  maxHeight: number,
): { readonly width: number; readonly height: number } {
  const k = Math.min(1, maxWidth / Math.max(1, width), maxHeight / Math.max(1, height))
  return { width: Math.round(width * k), height: Math.round(height * k) }
}

/**
 * A scene as one figure: a {@link FigureFrame} (expand, export) the size of the scene, shrunk to fit `maxWidth` by
 * `maxHeight`, holding a {@link SceneView}; in the modal the scene is drawn at the modal's size. Pass `playback` to
 * record an animated scene as a GIF or WebM.
 *
 * @param props - The scene, its view's handlers, the largest size and the frame's text and export settings.
 * @returns The figure.
 * @index function React component: a Scene in a frame with expand and export: `<SceneFrame scene={scene} />`
 */
export function SceneFrame(props: SceneFrameProps): ReactElement {
  const { scene, maxWidth = 720, maxHeight = 720, filename = 'scene', title, caption, playback, exportFormats } = props
  const size = fitScene(scene.width, scene.height, maxWidth, maxHeight)
  return (
    <FigureFrame
      width={size.width}
      height={size.height}
      filename={filename}
      {...(title === undefined ? {} : { title })}
      {...(caption === undefined ? {} : { caption })}
      {...(playback === undefined ? {} : { playback })}
      {...(exportFormats === undefined ? {} : { exportFormats })}
      {...(props.className === undefined ? {} : { className: props.className })}
    >
      {(frame) => (
        <SceneView
          scene={scene}
          width={frame.width}
          height={frame.height}
          {...(props.mode === undefined ? {} : { mode: props.mode })}
          {...(props.onNodeClick === undefined ? {} : { onNodeClick: props.onNodeClick })}
          {...(props.onNodeHover === undefined ? {} : { onNodeHover: props.onNodeHover })}
          {...(props.highlight === undefined ? {} : { highlight: props.highlight })}
        >
          {props.children}
        </SceneView>
      )}
    </FigureFrame>
  )
}
