import { add, clip, concat, floor, full, mul, pad, reshape, sliceAxis, tensor, toFlat } from 'omai-engine/tensor'
import { type ReactNode, useEffect, useRef } from 'react'

/**
 * Draws an image of interleaved red, green, blue values in [0, 1] on a canvas, one canvas pixel per image pixel and
 * scaled up without smoothing, so the pixels of a toy-sized image stay visible (`size` fixes the drawn width). Children are drawn over the image
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
  /** The drawn width in CSS pixels (default: the width of the container); use a whole multiple of `width`. */
  readonly size?: number
}): ReactNode {
  const { width, height, rgb, caption, children, size } = props
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const context = ref.current?.getContext('2d')
    if (context === null || context === undefined || rgb === undefined) return
    const image = context.createImageData(width, height)
    // The pixels as bytes: the colours clamped to [0, 1] and scaled to 255, an opaque alpha after each.
    const wanted = width * height * 3
    const given = tensor(rgb as number[])
    // Cut or padded with black to the size of the image.
    const values = pad(sliceAxis(given, 0, 0, Math.min(rgb.length, wanted)), [[0, Math.max(0, wanted - rgb.length)]])
    const bytes = floor(add(mul(clip(reshape(values, [-1, 3]), 0, 1), 255), 0.5))
    image.data.set(toFlat(concat([bytes, full([width * height, 1], 255)], 1)))
    context.putImageData(image, 0, 0)
  }, [rgb, width, height])
  return (
    <figure className="flex min-w-0 flex-col gap-1" style={size === undefined ? {} : { width: size }}>
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
