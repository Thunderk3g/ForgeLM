"use client"

import type React from "react"
import { Progress } from "@/components/ui/progress"

interface ProgressBarProps {
  progress: number
  label?: string
  className?: string
  showPercentage?: boolean
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ progress, label, className = "", showPercentage = true }) => {
  const clampedProgress = Math.max(0, Math.min(100, progress))

  return (
    <div className={`w-full space-y-2 ${className}`}>
      {label && (
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium leading-none">{label}</span>
          {showPercentage && (
            <span className="text-sm font-medium text-muted-foreground">{clampedProgress.toFixed(1)}%</span>
          )}
        </div>
      )}
      <Progress value={clampedProgress} className="h-2" />
    </div>
  )
}
