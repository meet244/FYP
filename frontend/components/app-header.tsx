'use client'

import Link from 'next/link'
import { Moon, Sun, Waves } from 'lucide-react'
import { useTheme } from 'next-themes'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { HealthBadge } from '@/components/health-badge'
import { cn } from '@/lib/utils'

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])

  return (
    <Button
      variant="ghost"
      size="icon"
      className="h-8 w-8 text-muted-foreground"
      onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
      aria-label="Toggle theme"
    >
      {/* Render a stable icon until mounted — the server cannot know the theme. */}
      {mounted && resolvedTheme === 'dark' ? (
        <Sun className="h-4 w-4" />
      ) : (
        <Moon className="h-4 w-4" />
      )}
    </Button>
  )
}

export function AppHeader({
  children,
  className,
}: {
  children?: React.ReactNode
  className?: string
}) {
  return (
    <header
      className={cn(
        'sticky top-0 z-40 border-b border-border/60 bg-background/80 backdrop-blur-xl',
        className
      )}
    >
      <div className="flex h-14 items-center gap-3 px-4 sm:px-6">
        <Link href="/" className="flex shrink-0 items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-accent shadow-md shadow-primary/25">
            <Waves className="h-4 w-4 text-white" />
          </div>
          <span className="text-base font-bold tracking-tight text-foreground">
            Class<span className="text-primary">Scribe</span>
          </span>
        </Link>

        <div className="min-w-0 flex-1">{children}</div>

        <div className="flex shrink-0 items-center gap-1.5">
          <HealthBadge />
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
