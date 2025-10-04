import React, { useState, useCallback } from "react"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { RotateCcw, Copy, Check, Type, Info } from "lucide-react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface TextAreaFieldProps {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  rows?: number
  maxLength?: number
  defaultValue?: string
  description?: string
  disabled?: boolean
  className?: string
  showWordCount?: boolean
  showCopy?: boolean
}

export function TextAreaField({
  label,
  value,
  onChange,
  placeholder = "",
  rows = 3,
  maxLength,
  defaultValue = "",
  description,
  disabled = false,
  className = "",
  showWordCount = true,
  showCopy = true
}: TextAreaFieldProps) {
  const [copied, setCopied] = useState(false)

  const handleReset = useCallback(() => {
    onChange(defaultValue)
  }, [defaultValue, onChange])

  const handleCopy = useCallback(async () => {
    if (!value) return
    
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      console.error("Failed to copy text:", error)
    }
  }, [value])

  const wordCount = value.trim() ? value.trim().split(/\s+/).length : 0
  const charCount = value.length

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Label className="text-sm font-medium">{label}</Label>
          {description && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="h-3 w-3 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-xs">
                  <p className="text-xs">{description}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>

        <div className="flex items-center space-x-2">
          {showCopy && value && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              disabled={disabled}
              className="h-6 w-6 p-0 opacity-50 hover:opacity-100"
            >
              {copied ? (
                <Check className="h-3 w-3 text-green-500" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
            </Button>
          )}
          
          {defaultValue !== undefined && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleReset}
              disabled={disabled || value === defaultValue}
              className="h-6 w-6 p-0 opacity-50 hover:opacity-100"
            >
              <RotateCcw className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>

      {/* Textarea */}
      <div className="relative">
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          rows={rows}
          maxLength={maxLength}
          disabled={disabled}
          className={`resize-none ${className}`}
        />
        
        {/* Character limit indicator */}
        {maxLength && (
          <div className="absolute bottom-2 right-2 text-xs text-muted-foreground bg-background/80 px-1 rounded">
            {charCount}/{maxLength}
          </div>
        )}
      </div>

      {/* Footer */}
      {showWordCount && (
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              <Type className="h-3 w-3" />
              <span>{wordCount} words</span>
            </div>
            <span>{charCount} characters</span>
          </div>
          
          {value && (
            <div className="flex items-center space-x-1">
              {value.includes("You are") && (
                <span className="text-green-500">System prompt detected</span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Quick templates */}
      {!value && !disabled && (
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onChange("You are a helpful assistant.")}
            className="text-xs h-6 px-2"
          >
            Default Assistant
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onChange("You are an expert in your field. Provide detailed, accurate responses.")}
            className="text-xs h-6 px-2"
          >
            Expert Mode
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onChange("You are a creative assistant focused on generating imaginative content.")}
            className="text-xs h-6 px-2"
          >
            Creative
          </Button>
        </div>
      )}
    </div>
  )
}