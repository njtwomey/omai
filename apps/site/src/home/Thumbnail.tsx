import { type ReactNode, useState } from 'react'
import { useColourMode } from '../shell/colour-mode.ts'

/** The thumbnail of a gallery recipe: its first figure, saved by `make thumbs`, in the page's colour mode. */
export function Thumbnail(props: { readonly section: string; readonly slug: string }): ReactNode {
  const mode = useColourMode()
  const [missing, setMissing] = useState(false)
  if (missing) return <span className="text-xs text-muted">no preview</span>
  return (
    <img
      src={`${import.meta.env.BASE_URL}thumbs/${props.section}/${props.slug}-${mode}.webp`}
      alt=""
      loading="lazy"
      onError={() => setMissing(true)}
      className="size-full object-contain p-1.5 transition-transform duration-300 group-hover:scale-[1.03]"
    />
  )
}
