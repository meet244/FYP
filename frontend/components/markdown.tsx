'use client'

/**
 * Small markdown renderer for note bodies and chat answers.
 *
 * Deliberately not `react-markdown`: the backend emits a narrow, predictable
 * subset (headings, lists, tables, emphasis, code) and chat answers additionally
 * carry `[n]` citation markers that must become interactive rather than text.
 * A full CommonMark pipeline would need custom renderers for that anyway.
 */
import * as React from 'react'
import { cn } from '@/lib/utils'

interface MarkdownProps {
  children: string
  className?: string
  /** Renders `[n]` markers as clickable chips. Chat only — notes have no citations. */
  onCitation?: (n: number) => void
}

/** Inline pass: `**bold**`, `*italic*`, `` `code` ``, then citation markers. */
function inline(text: string, onCitation?: (n: number) => void): React.ReactNode[] {
  const out: React.ReactNode[] = []
  const pattern = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`|\[\d+\])/g
  let last = 0
  let key = 0

  for (const m of text.matchAll(pattern)) {
    const i = m.index!
    if (i > last) out.push(text.slice(last, i))
    const tok = m[0]

    if (tok.startsWith('**')) {
      out.push(<strong key={key++} className="font-semibold text-foreground">{tok.slice(2, -2)}</strong>)
    } else if (tok.startsWith('`')) {
      out.push(
        <code key={key++} className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.85em] text-foreground">
          {tok.slice(1, -1)}
        </code>
      )
    } else if (tok.startsWith('[')) {
      const n = Number(tok.slice(1, -1))
      out.push(
        onCitation ? (
          <button
            key={key++}
            type="button"
            onClick={() => onCitation(n)}
            title={`Jump to source ${n}`}
            className="mx-0.5 inline-flex h-[1.15em] min-w-[1.15em] items-center justify-center rounded-[0.3em] bg-primary/15 px-1 align-super text-[0.7em] font-bold text-primary transition-colors hover:bg-primary/30"
          >
            {n}
          </button>
        ) : (
          <sup key={key++} className="text-primary">{tok}</sup>
        )
      )
    } else {
      out.push(<em key={key++} className="italic">{tok.slice(1, -1)}</em>)
    }
    last = i + tok.length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

function Table({ rows, onCitation }: { rows: string[]; onCitation?: (n: number) => void }) {
  const cells = (row: string) =>
    row.replace(/^\||\|$/g, '').split('|').map((c) => c.trim())
  const [head, ...body] = rows.filter((r) => !/^\|?[\s:|-]+\|?$/.test(r))

  return (
    <div className="my-3 overflow-x-auto rounded-lg border border-border/60">
      <table className="w-full text-sm">
        <thead className="bg-muted/50">
          <tr>
            {cells(head).map((c, i) => (
              <th key={i} className="px-3 py-2 text-left font-semibold text-foreground">
                {inline(c, onCitation)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, r) => (
            <tr key={r} className="border-t border-border/60">
              {cells(row).map((c, i) => (
                <td key={i} className="px-3 py-2 align-top text-muted-foreground">
                  {inline(c, onCitation)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Markdown({ children, className, onCitation }: MarkdownProps) {
  const blocks: React.ReactNode[] = []
  const lines = (children ?? '').split('\n')
  let i = 0
  let key = 0

  while (i < lines.length) {
    const line = lines[i]

    // fenced code
    if (line.trimStart().startsWith('```')) {
      const buf: string[] = []
      i++
      while (i < lines.length && !lines[i].trimStart().startsWith('```')) buf.push(lines[i++])
      i++
      blocks.push(
        <pre key={key++} className="my-3 overflow-x-auto rounded-lg border border-border/60 bg-muted/40 p-3">
          <code className="font-mono text-xs leading-relaxed text-foreground">{buf.join('\n')}</code>
        </pre>
      )
      continue
    }

    // table
    if (line.trim().startsWith('|') && line.includes('|', 1)) {
      const buf: string[] = []
      while (i < lines.length && lines[i].trim().startsWith('|')) buf.push(lines[i++])
      blocks.push(<Table key={key++} rows={buf} onCitation={onCitation} />)
      continue
    }

    // heading
    const h = /^(#{1,4})\s+(.*)$/.exec(line)
    if (h) {
      const level = h[1].length
      const size = ['text-lg', 'text-base', 'text-sm', 'text-sm'][level - 1]
      blocks.push(
        <p key={key++} className={cn('mt-4 mb-1.5 font-bold text-foreground first:mt-0', size)}>
          {inline(h[2], onCitation)}
        </p>
      )
      i++
      continue
    }

    // blockquote
    if (line.trimStart().startsWith('> ')) {
      const buf: string[] = []
      while (i < lines.length && lines[i].trimStart().startsWith('> ')) {
        buf.push(lines[i].trimStart().slice(2))
        i++
      }
      blocks.push(
        <blockquote key={key++} className="my-3 border-l-2 border-primary/40 pl-3 text-muted-foreground italic">
          {inline(buf.join(' '), onCitation)}
        </blockquote>
      )
      continue
    }

    // lists
    const bullet = /^\s*[-*+]\s+(.*)$/
    const ordered = /^\s*(\d+)[.)]\s+(.*)$/
    if (bullet.test(line) || ordered.test(line)) {
      const isOrdered = ordered.test(line)
      const items: string[] = []
      while (i < lines.length && (bullet.test(lines[i]) || ordered.test(lines[i]))) {
        const m = isOrdered ? ordered.exec(lines[i]) : bullet.exec(lines[i])
        items.push(isOrdered ? m![2] : m![1])
        i++
      }
      const Tag = isOrdered ? 'ol' : 'ul'
      blocks.push(
        <Tag
          key={key++}
          className={cn(
            'my-2 space-y-1.5 pl-5 text-muted-foreground',
            isOrdered ? 'list-decimal' : 'list-disc'
          )}
        >
          {items.map((it, n) => (
            <li key={n} className="leading-relaxed marker:text-primary/60">
              {inline(it, onCitation)}
            </li>
          ))}
        </Tag>
      )
      continue
    }

    // horizontal rule
    if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) {
      blocks.push(<hr key={key++} className="my-4 border-border/60" />)
      i++
      continue
    }

    // paragraph — greedily consume until a blank line or a block opener
    if (line.trim()) {
      const buf: string[] = []
      while (
        i < lines.length &&
        lines[i].trim() &&
        !/^(#{1,4}\s|\s*[-*+]\s|\s*\d+[.)]\s|>\s|```|\|)/.test(lines[i])
      ) {
        buf.push(lines[i].trim())
        i++
      }
      blocks.push(
        <p key={key++} className="my-2 leading-relaxed text-muted-foreground first:mt-0">
          {inline(buf.join(' '), onCitation)}
        </p>
      )
      continue
    }

    i++
  }

  return <div className={cn('text-sm', className)}>{blocks}</div>
}
