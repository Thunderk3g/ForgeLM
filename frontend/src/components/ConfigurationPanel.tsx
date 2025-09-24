import React, { useState, useEffect, useCallback } from "react"
import { 
  Settings, 
  ChevronLeft, 
  ChevronRight, 
  RotateCcw, 
  Save, 
  Sliders,
  MessageSquare,
  Zap,
  Copy,
  CheckCircle,
  AlertTriangle,
  Info
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Separator } from "@/components/ui/separator"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"

// Configuration components
import { ParameterSlider } from "./config/ParameterSlider"
import { TextAreaField } from "./config/TextAreaField"
import { PresetSelector } from "./config/PresetSelector"
import { ModelConfigSection } from "./config/ModelConfigSection"
import { ConfigValidator } from "./config/ConfigValidator"

// Types
import type { 
  DualModelConfig, 
  GenerationConfig, 
  ModelSpecificConfig,
  ModelPair 
} from "../lib/types"

interface ConfigurationPanelProps {
  sessionId: string
  isOpen: boolean
  onToggle: () => void
  modelPair: ModelPair | null
  currentConfig: DualModelConfig | null
  onConfigChange: (config: DualModelConfig) => void
  onApplyConfig: (config: DualModelConfig) => Promise<void>
  onSavePreset: (name: string, description: string) => Promise<void>
  isApplyingConfig?: boolean
  className?: string
}

export function ConfigurationPanel({
  sessionId,
  isOpen,
  onToggle,
  modelPair,
  currentConfig,
  onConfigChange,
  onApplyConfig,
  onSavePreset,
  isApplyingConfig = false,
  className = ""
}: ConfigurationPanelProps) {
  const [activeTab, setActiveTab] = useState<"generation" | "prompts" | "advanced">("generation")
  const [syncParameters, setSyncParameters] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  const [validationResults, setValidationResults] = useState<any>(null)
  const [showPresetDialog, setShowPresetDialog] = useState(false)
  
  // Create default config if none provided
  const createDefaultConfig = useCallback((): DualModelConfig => ({
    session_id: sessionId,
    base_model: {
      system_prompt: "",
      context_length: 2048,
      generation_config: {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        max_tokens: 1024,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repetition_penalty: 1.0,
        do_sample: true,
        early_stopping: false,
        seed: null,
        pad_token_id: null,
        eos_token_id: null
      },
      model_id: modelPair?.base_model_name || null,
      enabled: true
    },
    finetuned_model: {
      system_prompt: "",
      context_length: 2048,
      generation_config: {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        max_tokens: 1024,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repetition_penalty: 1.0,
        do_sample: true,
        early_stopping: false,
        seed: null,
        pad_token_id: null,
        eos_token_id: null
      },
      model_id: modelPair?.adapter_name || null,
      enabled: true
    },
    sync_parameters: false,
    config_name: null,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  }), [sessionId, modelPair])

  const [config, setConfig] = useState<DualModelConfig>(() => 
    currentConfig || createDefaultConfig()
  )

  // Update config when currentConfig changes
  useEffect(() => {
    if (currentConfig) {
      setConfig(currentConfig)
      setSyncParameters(currentConfig.sync_parameters)
    }
  }, [currentConfig])

  // Handle config changes
  const handleConfigUpdate = useCallback((updates: Partial<DualModelConfig>) => {
    const newConfig = { 
      ...config, 
      ...updates,
      updated_at: new Date().toISOString()
    }
    
    setConfig(newConfig)
    setHasUnsavedChanges(true)
    onConfigChange(newConfig)
  }, [config, onConfigChange])

  // Handle base model config changes
  const handleBaseModelUpdate = useCallback((updates: Partial<ModelSpecificConfig>) => {
    const newConfig = {
      ...config,
      base_model: { ...config.base_model, ...updates },
      updated_at: new Date().toISOString()
    }

    // Sync parameters if enabled
    if (syncParameters && updates.generation_config) {
      newConfig.finetuned_model.generation_config = { 
        ...newConfig.finetuned_model.generation_config, 
        ...updates.generation_config 
      }
    }

    setConfig(newConfig)
    setHasUnsavedChanges(true)
    onConfigChange(newConfig)
  }, [config, syncParameters, onConfigChange])

  // Handle fine-tuned model config changes
  const handleFinetunedModelUpdate = useCallback((updates: Partial<ModelSpecificConfig>) => {
    const newConfig = {
      ...config,
      finetuned_model: { ...config.finetuned_model, ...updates },
      updated_at: new Date().toISOString()
    }

    // Sync parameters if enabled
    if (syncParameters && updates.generation_config) {
      newConfig.base_model.generation_config = { 
        ...newConfig.base_model.generation_config, 
        ...updates.generation_config 
      }
    }

    setConfig(newConfig)
    setHasUnsavedChanges(true)
    onConfigChange(newConfig)
  }, [config, syncParameters, onConfigChange])

  // Handle sync toggle
  const handleSyncToggle = useCallback((checked: boolean) => {
    setSyncParameters(checked)
    
    if (checked) {
      // Sync finetuned config to base config
      const newConfig = {
        ...config,
        finetuned_model: {
          ...config.finetuned_model,
          generation_config: { ...config.base_model.generation_config }
        },
        sync_parameters: true,
        updated_at: new Date().toISOString()
      }
      setConfig(newConfig)
      onConfigChange(newConfig)
    } else {
      handleConfigUpdate({ sync_parameters: false })
    }
  }, [config, handleConfigUpdate, onConfigChange])

  // Reset to defaults
  const handleReset = useCallback(() => {
    const defaultConfig = createDefaultConfig()
    setConfig(defaultConfig)
    setSyncParameters(false)
    setHasUnsavedChanges(false)
    onConfigChange(defaultConfig)
  }, [createDefaultConfig, onConfigChange])

  // Apply configuration
  const handleApply = useCallback(async () => {
    try {
      await onApplyConfig(config)
      setHasUnsavedChanges(false)
    } catch (error) {
      console.error("Failed to apply configuration:", error)
    }
  }, [config, onApplyConfig])

  // Copy configuration to clipboard
  const handleCopyConfig = useCallback(async () => {
    try {
      const configJson = JSON.stringify(config, null, 2)
      await navigator.clipboard.writeText(configJson)
      // Show success toast
    } catch (error) {
      console.error("Failed to copy configuration:", error)
    }
  }, [config])

  return (
    <div className={`flex h-full w-full ${className}`}>
      {/* Configuration Panel */}
      <div className="w-full h-full overflow-hidden">
        <Card className="h-full border-0 shadow-none bg-transparent">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Settings className="h-5 w-5 text-primary" />
                  <CardTitle className="text-lg">Model Configuration</CardTitle>
                </div>
                <div className="flex items-center space-x-1">
                  {hasUnsavedChanges && (
                    <Badge variant="secondary" className="text-xs">
                      Unsaved
                    </Badge>
                  )}
                  <Button variant="ghost" size="sm" onClick={handleReset}>
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              
              <CardDescription>
                Configure generation parameters for both base and fine-tuned models
              </CardDescription>

              {modelPair && (
                <div className="flex flex-col space-y-2 text-xs">
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline" className="text-blue-600 border-blue-200">
                      Base
                    </Badge>
                    <span className="text-muted-foreground truncate">
                      {modelPair.base_model_name}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline" className="text-green-600 border-green-200">
                      Fine-tuned
                    </Badge>
                    <span className="text-muted-foreground truncate">
                      {modelPair.adapter_name}
                    </span>
                  </div>
                </div>
              )}

              {/* Sync Parameters Toggle */}
              <div className="flex items-center justify-between pt-2 border-t">
                <div className="flex items-center space-x-2">
                  <Switch
                    id="sync-params"
                    checked={syncParameters}
                    onCheckedChange={handleSyncToggle}
                  />
                  <Label htmlFor="sync-params" className="text-sm">
                    Sync Parameters
                  </Label>
                  <Info className="h-3 w-3 text-muted-foreground" />
                </div>
              </div>
            </CardHeader>

            <CardContent className="space-y-4 h-full overflow-auto pb-20">
              {/* Configuration Validation */}
              <ConfigValidator 
                config={config}
                onValidationResults={setValidationResults}
              />

              {/* Preset Selector */}
              <PresetSelector
                sessionId={sessionId}
                onPresetApply={(preset: any) => {
                  const newConfig: DualModelConfig = {
                    ...config,
                    base_model: {
                      ...config.base_model,
                      generation_config: preset.base_config,
                      system_prompt: preset.system_prompts?.base || config.base_model.system_prompt
                    },
                    finetuned_model: {
                      ...config.finetuned_model,
                      generation_config: preset.finetuned_config,
                      system_prompt: preset.system_prompts?.finetuned || config.finetuned_model.system_prompt
                    },
                    config_name: `Preset: ${preset.name}`,
                    updated_at: new Date().toISOString()
                  }
                  setConfig(newConfig)
                  onConfigChange(newConfig)
                  setHasUnsavedChanges(true)
                }}
              />

              <Separator />

              {/* Configuration Tabs */}
              <Tabs value={activeTab} onValueChange={(value: any) => setActiveTab(value)}>
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="generation" className="text-xs">
                    <Sliders className="h-3 w-3 mr-1" />
                    Parameters
                  </TabsTrigger>
                  <TabsTrigger value="prompts" className="text-xs">
                    <MessageSquare className="h-3 w-3 mr-1" />
                    Prompts
                  </TabsTrigger>
                  <TabsTrigger value="advanced" className="text-xs">
                    <Zap className="h-3 w-3 mr-1" />
                    Advanced
                  </TabsTrigger>
                </TabsList>

                {/* Generation Parameters Tab */}
                <TabsContent value="generation" className="space-y-4 mt-4">
                  <ModelConfigSection
                    title="Base Model"
                    modelType="base"
                    config={config.base_model}
                    onConfigChange={handleBaseModelUpdate}
                    syncEnabled={syncParameters}
                    color="blue"
                  />
                  
                  <ModelConfigSection
                    title="Fine-tuned Model"
                    modelType="finetuned"
                    config={config.finetuned_model}
                    onConfigChange={handleFinetunedModelUpdate}
                    syncEnabled={syncParameters}
                    color="green"
                  />
                </TabsContent>

                {/* System Prompts Tab */}
                <TabsContent value="prompts" className="space-y-4 mt-4">
                  <div className="space-y-4">
                    <TextAreaField
                      label="Base Model System Prompt"
                      value={config.base_model.system_prompt}
                      onChange={(value) => handleBaseModelUpdate({ system_prompt: value })}
                      placeholder="You are a helpful assistant..."
                      rows={4}
                      className="border-blue-200 focus:border-blue-400"
                    />
                    
                    <TextAreaField
                      label="Fine-tuned Model System Prompt"
                      value={config.finetuned_model.system_prompt}
                      onChange={(value) => handleFinetunedModelUpdate({ system_prompt: value })}
                      placeholder="You are a specialized assistant..."
                      rows={4}
                      className="border-green-200 focus:border-green-400"
                    />
                  </div>
                </TabsContent>

                {/* Advanced Settings Tab */}
                <TabsContent value="advanced" className="space-y-4 mt-4">
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label className="text-xs text-muted-foreground">Base Context Length</Label>
                        <ParameterSlider
                          label=""
                          value={config.base_model.context_length}
                          onChange={(value) => handleBaseModelUpdate({ context_length: value })}
                          min={512}
                          max={8192}
                          step={1024}
                          showInput
                        />
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">Fine-tuned Context Length</Label>
                        <ParameterSlider
                          label=""
                          value={config.finetuned_model.context_length}
                          onChange={(value) => handleFinetunedModelUpdate({ context_length: value })}
                          min={512}
                          max={8192}
                          step={1024}
                          showInput
                        />
                      </div>
                    </div>

                    {/* Advanced Generation Parameters */}
                    <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
                      <CollapsibleTrigger asChild>
                        <Button variant="ghost" className="w-full justify-between text-sm">
                          Advanced Parameters
                          <ChevronRight className={`h-4 w-4 transition-transform ${showAdvanced ? 'rotate-90' : ''}`} />
                        </Button>
                      </CollapsibleTrigger>
                      <CollapsibleContent className="space-y-4 mt-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <Label className="text-xs text-muted-foreground">Base Seed</Label>
                            <ParameterSlider
                              label=""
                              value={config.base_model.generation_config.seed || 0}
                              onChange={(value) => handleBaseModelUpdate({
                                generation_config: { 
                                  ...config.base_model.generation_config, 
                                  seed: value === 0 ? null : value 
                                }
                              })}
                              min={0}
                              max={9999}
                              step={1}
                              showInput
                            />
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground">Fine-tuned Seed</Label>
                            <ParameterSlider
                              label=""
                              value={config.finetuned_model.generation_config.seed || 0}
                              onChange={(value) => handleFinetunedModelUpdate({
                                generation_config: { 
                                  ...config.finetuned_model.generation_config, 
                                  seed: value === 0 ? null : value 
                                }
                              })}
                              min={0}
                              max={9999}
                              step={1}
                              showInput
                            />
                          </div>
                        </div>
                      </CollapsibleContent>
                    </Collapsible>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>

            {/* Action Buttons */}
            <div className="absolute bottom-0 left-0 right-0 p-4 border-t bg-background">
              <div className="flex items-center justify-between space-x-2">
                <div className="flex items-center space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCopyConfig}
                    className="text-xs"
                  >
                    <Copy className="h-3 w-3 mr-1" />
                    Copy
                  </Button>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowPresetDialog(true)}
                    className="text-xs"
                  >
                    <Save className="h-3 w-3 mr-1" />
                    Save Preset
                  </Button>
                </div>

                <Button
                  onClick={handleApply}
                  disabled={!hasUnsavedChanges || isApplyingConfig}
                  size="sm"
                  className="text-xs"
                >
                  {isApplyingConfig ? (
                    <>
                      <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1" />
                      Applying...
                    </>
                  ) : (
                    <>
                      <CheckCircle className="h-3 w-3 mr-1" />
                      Apply
                    </>
                  )}
                </Button>
              </div>

              {/* Validation Summary */}
              {validationResults && (
                <div className="mt-2 text-xs">
                  {validationResults.warnings?.length > 0 && (
                    <Alert className="py-2">
                      <AlertTriangle className="h-3 w-3" />
                      <AlertDescription className="text-xs">
                        {validationResults.warnings.slice(0, 2).join("; ")}
                        {validationResults.warnings.length > 2 && "..."}
                      </AlertDescription>
                    </Alert>
                  )}
                </div>
              )}
            </div>
          </Card>
      </div>
    </div>
  )
}