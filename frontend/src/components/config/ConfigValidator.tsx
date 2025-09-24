import React, { useState, useEffect, useCallback } from "react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { 
  CheckCircle, 
  AlertTriangle, 
  AlertCircle, 
  Info,
  Zap,
  Clock,
  MemoryStick,
  Loader2,
  RefreshCw
} from "lucide-react"
import type { DualModelConfig } from "@/lib/types"
import { validateSessionConfig } from "@/lib/api"

interface ConfigValidatorProps {
  config: DualModelConfig
  onValidationResults?: (results: ValidationResults) => void
  autoValidate?: boolean
  className?: string
}

interface ValidationResults {
  is_valid: boolean
  warnings: string[]
  errors: string[]
  estimated_performance?: {
    base_model_speed_factor: number
    finetuned_model_speed_factor: number
  }
  memory_requirements?: {
    base_model_mb: number
    finetuned_model_mb: number
    total_mb: number
  }
}

export function ConfigValidator({
  config,
  onValidationResults,
  autoValidate = true,
  className = ""
}: ConfigValidatorProps) {
  const [validation, setValidation] = useState<ValidationResults | null>(null)
  const [isValidating, setIsValidating] = useState(false)
  const [lastValidated, setLastValidated] = useState<string | null>(null)

  const validateConfig = useCallback(async () => {
    try {
      setIsValidating(true)
      const results = await validateSessionConfig(config.session_id, config) as ValidationResults
      setValidation(results)
      setLastValidated(new Date().toISOString())
      
      if (onValidationResults) {
        onValidationResults(results)
      }
    } catch (error) {
      console.error("Validation failed:", error)
      setValidation({
        is_valid: false,
        warnings: [],
        errors: ["Validation service unavailable"]
      })
    } finally {
      setIsValidating(false)
    }
  }, [config, onValidationResults])

  // Auto-validate when config changes
  useEffect(() => {
    if (autoValidate) {
      const timer = setTimeout(() => {
        validateConfig()
      }, 1000) // Debounce validation

      return () => clearTimeout(timer)
    }
  }, [config, autoValidate, validateConfig])

  const getStatusIcon = () => {
    if (isValidating) return <Loader2 className="h-4 w-4 animate-spin" />
    if (!validation) return <Info className="h-4 w-4" />
    if (!validation.is_valid) return <AlertCircle className="h-4 w-4 text-red-500" />
    if (validation.warnings.length > 0) return <AlertTriangle className="h-4 w-4 text-yellow-500" />
    return <CheckCircle className="h-4 w-4 text-green-500" />
  }

  const getStatusText = () => {
    if (isValidating) return "Validating..."
    if (!validation) return "Click to validate"
    if (!validation.is_valid) return "Invalid configuration"
    if (validation.warnings.length > 0) return `Valid with ${validation.warnings.length} warnings`
    return "Configuration valid"
  }

  const getStatusColor = () => {
    if (isValidating) return "border-blue-200 bg-blue-50"
    if (!validation) return "border-gray-200 bg-gray-50"
    if (!validation.is_valid) return "border-red-200 bg-red-50"
    if (validation.warnings.length > 0) return "border-yellow-200 bg-yellow-50"
    return "border-green-200 bg-green-50"
  }

  const formatMemorySize = (mb: number): string => {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)}GB`
    }
    return `${mb}MB`
  }

  const getSpeedFactorColor = (factor: number): string => {
    if (factor >= 1.2) return "text-green-600"
    if (factor >= 0.8) return "text-yellow-600"
    return "text-red-600"
  }

  const getSpeedFactorText = (factor: number): string => {
    if (factor >= 1.2) return "Fast"
    if (factor >= 0.8) return "Normal"
    return "Slow"
  }

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Validation Status Header */}
      <div className={`rounded-lg border p-3 ${getStatusColor()}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {getStatusIcon()}
            <span className="text-sm font-medium">{getStatusText()}</span>
          </div>
          
          <div className="flex items-center space-x-2">
            {lastValidated && (
              <span className="text-xs text-muted-foreground">
                {new Date(lastValidated).toLocaleTimeString()}
              </span>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={validateConfig}
              disabled={isValidating}
              className="h-6 w-6 p-0"
            >
              <RefreshCw className={`h-3 w-3 ${isValidating ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>

        {/* Performance Indicators */}
        {validation?.estimated_performance && (
          <div className="mt-2 grid grid-cols-2 gap-4 text-xs">
            <div className="flex items-center space-x-2">
              <Clock className="h-3 w-3 text-muted-foreground" />
              <span className="text-muted-foreground">Base:</span>
              <span className={getSpeedFactorColor(validation.estimated_performance.base_model_speed_factor)}>
                {getSpeedFactorText(validation.estimated_performance.base_model_speed_factor)}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <Clock className="h-3 w-3 text-muted-foreground" />
              <span className="text-muted-foreground">Fine-tuned:</span>
              <span className={getSpeedFactorColor(validation.estimated_performance.finetuned_model_speed_factor)}>
                {getSpeedFactorText(validation.estimated_performance.finetuned_model_speed_factor)}
              </span>
            </div>
          </div>
        )}

        {/* Memory Requirements */}
        {validation?.memory_requirements && (
          <div className="mt-2 flex items-center space-x-4 text-xs">
            <div className="flex items-center space-x-2">
              <MemoryStick className="h-3 w-3 text-muted-foreground" />
              <span className="text-muted-foreground">Memory:</span>
              <span className="font-mono">
                {formatMemorySize(validation.memory_requirements.total_mb)}
              </span>
            </div>
            <div className="text-muted-foreground">
              ({formatMemorySize(validation.memory_requirements.base_model_mb)} + {formatMemorySize(validation.memory_requirements.finetuned_model_mb)})
            </div>
          </div>
        )}
      </div>

      {/* Validation Messages */}
      {validation && (
        <div className="space-y-2">
          {/* Errors */}
          {validation.errors.map((error, index) => (
            <Alert key={`error-${index}`} variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="text-xs">{error}</AlertDescription>
            </Alert>
          ))}

          {/* Warnings */}
          {validation.warnings.slice(0, 3).map((warning, index) => (
            <Alert key={`warning-${index}`} className="border-yellow-200 bg-yellow-50">
              <AlertTriangle className="h-4 w-4 text-yellow-600" />
              <AlertDescription className="text-xs text-yellow-800">
                {warning}
              </AlertDescription>
            </Alert>
          ))}

          {/* Show count if there are more warnings */}
          {validation.warnings.length > 3 && (
            <div className="text-center">
              <Badge variant="outline" className="text-xs">
                +{validation.warnings.length - 3} more warnings
              </Badge>
            </div>
          )}
        </div>
      )}

      {/* Quick Optimization Suggestions */}
      {validation && validation.warnings.length > 0 && (
        <div className="rounded-lg border border-blue-200 bg-blue-50 p-3">
          <div className="flex items-center space-x-2 mb-2">
            <Zap className="h-4 w-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-800">Quick Fixes</span>
          </div>
          <div className="space-y-1 text-xs text-blue-700">
            {validation.warnings.some(w => w.includes("temperature")) && (
              <div>• Lower temperature for more consistent results</div>
            )}
            {validation.warnings.some(w => w.includes("max_tokens")) && (
              <div>• Reduce max_tokens for better performance</div>
            )}
            {validation.warnings.some(w => w.includes("do_sample")) && (
              <div>• Enable sampling or adjust temperature</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}