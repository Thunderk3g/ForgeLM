import React, { useState, useCallback } from "react"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { RotateCcw, Info } from "lucide-react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface ParameterSliderProps {
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  defaultValue?: number
  description?: string
  showInput?: boolean
  disabled?: boolean
  className?: string
  inputClassName?: string
}

export function ParameterSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  defaultValue,
  description,
  showInput = false,
  disabled = false,
  className = "",
  inputClassName = ""
}: ParameterSliderProps) {
  const [inputValue, setInputValue] = useState(value.toString())
  const [isInputFocused, setIsInputFocused] = useState(false)

  // Update input value when prop value changes (unless input is focused)
  React.useEffect(() => {
    if (!isInputFocused) {
      setInputValue(value.toString())
    }
  }, [value, isInputFocused])

  const handleSliderChange = useCallback((values: number[]) => {
    const newValue = values[0]
    onChange(newValue)
  }, [onChange])

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value)
  }, [])

  const handleInputBlur = useCallback(() => {
    setIsInputFocused(false)
    const numValue = parseFloat(inputValue)
    
    if (isNaN(numValue)) {
      setInputValue(value.toString())
      return
    }

    const clampedValue = Math.max(min, Math.min(max, numValue))
    const steppedValue = Math.round(clampedValue / step) * step
    
    setInputValue(steppedValue.toString())
    onChange(steppedValue)
  }, [inputValue, value, min, max, step, onChange])

  const handleInputFocus = useCallback(() => {
    setIsInputFocused(true)
  }, [])

  const handleInputKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleInputBlur()
      e.currentTarget.blur()
    }
  }, [handleInputBlur])

  const handleReset = useCallback(() => {
    if (defaultValue !== undefined) {
      onChange(defaultValue)
    }
  }, [defaultValue, onChange])

  const formatValue = (val: number): string => {
    if (step >= 1) {
      return val.toString()
    }
    return val.toFixed(Math.max(0, -Math.floor(Math.log10(step))))
  }

  return (
    <div className={`space-y-3 ${className}`}>
      {label && (
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
            {!showInput && (
              <span className="text-xs text-muted-foreground font-mono min-w-[3rem] text-right">
                {formatValue(value)}
              </span>
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
      )}

      <div className="space-y-2">
        {/* Slider */}
        <div className="px-2">
          <Slider
            value={[value]}
            onValueChange={handleSliderChange}
            min={min}
            max={max}
            step={step}
            disabled={disabled}
            className="w-full"
          />
        </div>

        {/* Range indicators */}
        <div className="flex justify-between text-xs text-muted-foreground px-2">
          <span>{formatValue(min)}</span>
          <span>{formatValue(max)}</span>
        </div>

        {/* Input field */}
        {showInput && (
          <div className="flex items-center space-x-2">
            <Input
              type="number"
              value={inputValue}
              onChange={handleInputChange}
              onBlur={handleInputBlur}
              onFocus={handleInputFocus}
              onKeyDown={handleInputKeyDown}
              min={min}
              max={max}
              step={step}
              disabled={disabled}
              className={`w-full text-sm font-mono ${inputClassName}`}
              placeholder={formatValue(value)}
            />
          </div>
        )}
      </div>
    </div>
  )
}