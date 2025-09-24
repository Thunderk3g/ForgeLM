import React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Link2, Link2Off } from "lucide-react"
import { ParameterSlider } from "./ParameterSlider"
import type { ModelSpecificConfig, GenerationConfig } from "@/lib/types"

interface ModelConfigSectionProps {
  title: string
  modelType: "base" | "finetuned"
  config: ModelSpecificConfig
  onConfigChange: (updates: Partial<ModelSpecificConfig>) => void
  syncEnabled?: boolean
  color?: "blue" | "green" | "gray"
  className?: string
}

const colorClasses = {
  blue: "border-blue-200 bg-blue-50/50 dark:bg-blue-950/20",
  green: "border-green-200 bg-green-50/50 dark:bg-green-950/20", 
  gray: "border-gray-200 bg-gray-50/50 dark:bg-gray-950/20"
}

const badgeClasses = {
  blue: "bg-blue-100 text-blue-800 border-blue-200",
  green: "bg-green-100 text-green-800 border-green-200",
  gray: "bg-gray-100 text-gray-800 border-gray-200"
}

export function ModelConfigSection({
  title,
  modelType,
  config,
  onConfigChange,
  syncEnabled = false,
  color = "gray",
  className = ""
}: ModelConfigSectionProps) {
  
  const handleGenerationConfigChange = (updates: Partial<GenerationConfig>) => {
    onConfigChange({
      generation_config: { ...config.generation_config, ...updates }
    })
  }

  const handleToggleEnabled = (enabled: boolean) => {
    onConfigChange({ enabled })
  }

  return (
    <Card className={`${colorClasses[color]} ${className}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <CardTitle className="text-sm font-medium">{title}</CardTitle>
            <Badge variant="outline" className={badgeClasses[color]}>
              {modelType}
            </Badge>
            {syncEnabled && (
              <Link2 className="h-3 w-3 text-muted-foreground" />
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            <Switch
              checked={config.enabled}
              onCheckedChange={handleToggleEnabled}
              size="sm"
            />
            <Label className="text-xs text-muted-foreground">Enabled</Label>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4" style={{ opacity: config.enabled ? 1 : 0.5 }}>
        {/* Core Parameters */}
        <div className="space-y-4">
          <ParameterSlider
            label="Temperature"
            value={config.generation_config.temperature}
            onChange={(value) => handleGenerationConfigChange({ temperature: value })}
            min={0.1}
            max={2.0}
            step={0.1}
            defaultValue={0.7}
            description="Controls randomness in generation. Higher values make output more creative but less consistent."
            disabled={!config.enabled}
          />

          <ParameterSlider
            label="Max Tokens"
            value={config.generation_config.max_tokens}
            onChange={(value) => handleGenerationConfigChange({ max_tokens: value })}
            min={50}
            max={4096}
            step={1}
            defaultValue={1024}
            description="Maximum number of tokens to generate in the response"
            disabled={!config.enabled}
            showInput
          />

          <ParameterSlider
            label="Top-p (Nucleus)"
            value={config.generation_config.top_p}
            onChange={(value) => handleGenerationConfigChange({ top_p: value })}
            min={0.1}
            max={1.0}
            step={0.05}
            defaultValue={0.9}
            description="Nucleus sampling parameter. Lower values make output more focused."
            disabled={!config.enabled}
          />

          <ParameterSlider
            label="Top-k"
            value={config.generation_config.top_k || 50}
            onChange={(value) => handleGenerationConfigChange({ top_k: value })}
            min={1}
            max={100}
            step={1}
            defaultValue={50}
            description="Limits sampling to the top-k most likely tokens"
            disabled={!config.enabled}
          />
        </div>

        {/* Advanced Parameters */}
        <div className="pt-2 border-t space-y-4">
          <ParameterSlider
            label="Presence Penalty"
            value={config.generation_config.presence_penalty}
            onChange={(value) => handleGenerationConfigChange({ presence_penalty: value })}
            min={-2.0}
            max={2.0}
            step={0.1}
            defaultValue={0.0}
            description="Penalty for tokens that have already appeared in the text"
            disabled={!config.enabled}
          />

          <ParameterSlider
            label="Frequency Penalty"
            value={config.generation_config.frequency_penalty}
            onChange={(value) => handleGenerationConfigChange({ frequency_penalty: value })}
            min={-2.0}
            max={2.0}
            step={0.1}
            defaultValue={0.0}
            description="Penalty based on token frequency in the text"
            disabled={!config.enabled}
          />

          <ParameterSlider
            label="Repetition Penalty"
            value={config.generation_config.repetition_penalty}
            onChange={(value) => handleGenerationConfigChange({ repetition_penalty: value })}
            min={0.5}
            max={2.0}
            step={0.05}
            defaultValue={1.0}
            description="Transformers-specific repetition penalty"
            disabled={!config.enabled}
          />
        </div>

        {/* Sampling Options */}
        <div className="pt-2 border-t space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-sm">Do Sample</Label>
              <p className="text-xs text-muted-foreground">Use sampling vs greedy decoding</p>
            </div>
            <Switch
              checked={config.generation_config.do_sample}
              onCheckedChange={(value) => handleGenerationConfigChange({ do_sample: value })}
              disabled={!config.enabled}
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <Label className="text-sm">Early Stopping</Label>
              <p className="text-xs text-muted-foreground">Stop generation early when possible</p>
            </div>
            <Switch
              checked={config.generation_config.early_stopping}
              onCheckedChange={(value) => handleGenerationConfigChange({ early_stopping: value })}
              disabled={!config.enabled}
            />
          </div>
        </div>

        {/* Performance Indicator */}
        <div className="pt-2 border-t">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Estimated Speed</span>
            <span className={`font-mono ${
              config.generation_config.max_tokens > 1000 ? 'text-yellow-600' :
              config.generation_config.temperature > 1.5 ? 'text-yellow-600' : 
              'text-green-600'
            }`}>
              {config.generation_config.max_tokens > 1000 || config.generation_config.temperature > 1.5 
                ? 'Slower' : 'Fast'}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}