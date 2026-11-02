import { type ReactNode, useEffect, useRef } from 'react'

/**
 * Draws an image of interleaved red, green, blue values in [0, 1] on a canvas, one canvas pixel per image pixel and
 * scaled up without smoothing, so the pixels of a toy-sized image stay visible. Children are drawn over the image
 * (for example an SVG sharing its coordinates).
 *
 * @param props - The image's size, its values (row 0 first), a caption and an optional overlay.
 * @returns The image with its caption.
 */
export function PixelCanvas(props: {
  readonly width: number
  readonly height: number
  readonly rgb: readonly number[] | undefined
  readonly caption: string
  readonly children?: ReactNode
}): ReactNode {
  const { width, height, rgb, caption, children } = props
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const context = ref.current?.getContext('2d')
    if (context === null || context === undefined || rgb === undefined) return
    const image = context.createImageData(width, height)
    for (let i = 0; i < width * height; i++) {
      for (let c = 0; c < 3; c++)
        image.data[4 * i + c] = Math.round(255 * Math.min(1, Math.max(0, rgb[3 * i + c] ?? 0)))
      image.data[4 * i + 3] = 255
    }
    context.putImageData(image, 0, 0)
  }, [rgb, width, height])
  return (
    <figure className="flex min-w-0 flex-col gap-1">
      <div className="relative w-full" style={{ aspectRatio: `${width} / ${height}` }}>
        <canvas
          ref={ref}
          width={width}
          height={height}
          className="h-full w-full rounded border border-line"
          style={{ imageRendering: 'pixelated' }}
        />
        {children}
      </div>
      <figcaption className="text-xs text-muted">{caption}</figcaption>
    </figure>
  )
}
