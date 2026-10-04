'use client'

import Link from 'next/link'
import { Moon, Sun } from 'lucide-react'
import { useTheme } from 'next-themes'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { HealthBadge } from '@/components/health-badge'
import { cn } from '@/lib/utils'

export function ThemeToggle() {
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
      {mounted && resolvedTheme === 'dark' ? (
        <Sun className="h-3.5 w-3.5" />
      ) : (
        <Moon className="h-3.5 w-3.5" />
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
        'sticky top-0 z-40 border-b border-border bg-background/90 backdrop-blur-md',
        className
      )}
    >
      <div className="flex h-12 items-center gap-3 px-4 sm:px-6">
        <Link href="/" className="flex shrink-0 items-center gap-2.5">
          <span className="flex h-7 w-7 items-center justify-center border border-foreground/20 font-display text-[13px] leading-none text-foreground">
            CS
          </span>
          <span className="hidden font-display text-[17px] leading-none tracking-tight text-foreground sm:inline">
            ClassScribe
          </span>
        </Link>

        <div className="min-w-0 flex-1">{children}</div>

        <div className="flex shrink-0 items-center gap-0.5">
          <HealthBadge />
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
