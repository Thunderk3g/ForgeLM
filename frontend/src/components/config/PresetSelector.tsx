import React, { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { 
  Palette, 
  Plus, 
  Zap, 
  Brain, 
  Code2, 
  BarChart3, 
  Lightbulb,
  Loader2,
  Check,
  AlertTriangle
} from "lucide-react"
import { toast } from "react-hot-toast"

// Import API functions
import { 
  listConfigPresets, 
  getConfigPreset, 
  createConfigPreset,
  applyPresetToSession 
} from "@/lib/api"

// Import types
import type { GenerationConfig } from "@/lib/types"

// Temporary workaround for ConfigPreset import issue
interface ConfigPreset {
  preset_id: string
  name: string
  description: string
  category: string
  base_config: GenerationConfig
  finetuned_config: GenerationConfig
  system_prompts: Record<string, string>
  is_built_in: boolean
  created_by: string | null
  created_at: string
  usage_count: number
}

interface PresetSelectorProps {
  sessionId: string
  onPresetApply: (preset: ConfigPreset) => void
  onPresetCreate?: (preset: ConfigPreset) => void
  className?: string
}

const presetIcons: Record<string, React.ReactNode> = {
  creative: <Lightbulb className="h-4 w-4" />,
  factual: <BarChart3 className="h-4 w-4" />,
  code: <Code2 className="h-4 w-4" />,
  general: <Brain className="h-4 w-4" />,
  default: <Zap className="h-4 w-4" />
}

const categoryColors: Record<string, string> = {
  creative: "bg-purple-100 text-purple-800 border-purple-200",
  factual: "bg-blue-100 text-blue-800 border-blue-200", 
  code: "bg-green-100 text-green-800 border-green-200",
  general: "bg-gray-100 text-gray-800 border-gray-200"
}

export function PresetSelector({ 
  sessionId, 
  onPresetApply, 
  onPresetCreate,
  className = "" 
}: PresetSelectorProps) {
  const [presets, setPresets] = useState<ConfigPreset[]>([])
  const [categories, setCategories] = useState<string[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string>("all")
  const [isLoading, setIsLoading] = useState(false)
  const [isApplying, setIsApplying] = useState<string | null>(null)
  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const [showPresetDetails, setShowPresetDetails] = useState<string | null>(null)

  // Create preset form state
  const [newPresetName, setNewPresetName] = useState("")
  const [newPresetDescription, setNewPresetDescription] = useState("")
  const [newPresetCategory, setNewPresetCategory] = useState("general")
  const [isCreating, setIsCreating] = useState(false)

  // Load presets on mount
  useEffect(() => {
    loadPresets()
  }, [])

  const loadPresets = async () => {
    try {
      setIsLoading(true)
      const response = await listConfigPresets()
      setPresets(response.presets)
      setCategories(["all", ...response.categories])
    } catch (error) {
      console.error("Failed to load presets:", error)
      toast.error("Failed to load configuration presets")
    } finally {
      setIsLoading(false)
    }
  }

  const handlePresetApply = async (presetId: string) => {
    try {
      setIsApplying(presetId)
      
      // Get preset details and apply to session
      const preset = await getConfigPreset(presetId)
      await applyPresetToSession(sessionId, presetId)
      
      onPresetApply(preset)
      toast.success(`Applied preset: ${preset.name}`)
    } catch (error) {
      console.error("Failed to apply preset:", error)
      toast.error("Failed to apply preset")
    } finally {
      setIsApplying(null)
    }
  }

  const handlePresetCreate = async () => {
    if (!newPresetName.trim()) {
      toast.error("Please enter a preset name")
      return
    }

    try {
      setIsCreating(true)
      
      // Create default configuration for new preset
      const defaultConfig: GenerationConfig = {
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
      }

      const newPreset = await createConfigPreset({
        name: newPresetName,
        description: newPresetDescription,
        category: newPresetCategory,
        base_config: defaultConfig,
        finetuned_config: defaultConfig,
        system_prompts: {
          base: "",
          finetuned: ""
        }
      })

      setPresets(prev => [...prev, newPreset])
      
      if (onPresetCreate) {
        onPresetCreate(newPreset)
      }

      toast.success(`Created preset: ${newPreset.name}`)
      
      // Reset form
      setNewPresetName("")
      setNewPresetDescription("")
      setNewPresetCategory("general")
      setShowCreateDialog(false)
      
    } catch (error) {
      console.error("Failed to create preset:", error)
      toast.error("Failed to create preset")
    } finally {
      setIsCreating(false)
    }
  }

  const filteredPresets = selectedCategory === "all" 
    ? presets 
    : presets.filter(preset => preset.category === selectedCategory)

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Palette className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium">Configuration Presets</span>
        </div>
        
        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm" className="text-xs">
              <Plus className="h-3 w-3 mr-1" />
              Create
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create Configuration Preset</DialogTitle>
              <DialogDescription>
                Save your current configuration as a preset for future use
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div>
                <Label htmlFor="preset-name">Name</Label>
                <Input
                  id="preset-name"
                  value={newPresetName}
                  onChange={(e) => setNewPresetName(e.target.value)}
                  placeholder="My Preset"
                />
              </div>
              <div>
                <Label htmlFor="preset-description">Description</Label>
                <Textarea
                  id="preset-description"
                  value={newPresetDescription}
                  onChange={(e) => setNewPresetDescription(e.target.value)}
                  placeholder="Describe when to use this preset..."
                  rows={3}
                />
              </div>
              <div>
                <Label htmlFor="preset-category">Category</Label>
                <Select value={newPresetCategory} onValueChange={setNewPresetCategory}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="general">General</SelectItem>
                    <SelectItem value="creative">Creative</SelectItem>
                    <SelectItem value="factual">Factual</SelectItem>
                    <SelectItem value="code">Code</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex justify-end space-x-2">
                <Button 
                  variant="outline" 
                  onClick={() => setShowCreateDialog(false)}
                  disabled={isCreating}
                >
                  Cancel
                </Button>
                <Button 
                  onClick={handlePresetCreate}
                  disabled={isCreating || !newPresetName.trim()}
                >
                  {isCreating ? (
                    <>
                      <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    <>
                      <Check className="h-3 w-3 mr-1" />
                      Create
                    </>
                  )}
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Category Filter */}
      {categories.length > 2 && (
        <Select value={selectedCategory} onValueChange={setSelectedCategory}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select category..." />
          </SelectTrigger>
          <SelectContent>
            {categories.map(category => (
              <SelectItem key={category} value={category}>
                {category === "all" ? "All Categories" : category.charAt(0).toUpperCase() + category.slice(1)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}

      {/* Presets Grid */}
      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : filteredPresets.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">
          <AlertTriangle className="h-8 w-8 mx-auto mb-2" />
          <p className="text-sm">No presets found</p>
          <p className="text-xs">Create your first preset to get started</p>
        </div>
      ) : (
        <div className="grid gap-2">
          {filteredPresets.map(preset => (
            <Card 
              key={preset.preset_id} 
              className="cursor-pointer hover:shadow-md transition-shadow"
              onClick={() => setShowPresetDetails(preset.preset_id)}
            >
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="flex-shrink-0">
                      {presetIcons[preset.category] || presetIcons.default}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-1">
                        <h3 className="text-sm font-medium truncate">
                          {preset.name}
                        </h3>
                        {preset.is_built_in && (
                          <Badge variant="outline" className="text-xs">
                            Built-in
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground truncate">
                        {preset.description || "No description"}
                      </p>
                      <div className="flex items-center space-x-2 mt-1">
                        <Badge 
                          variant="outline" 
                          className={`text-xs ${categoryColors[preset.category] || categoryColors.general}`}
                        >
                          {preset.category}
                        </Badge>
                        {preset.usage_count > 0 && (
                          <span className="text-xs text-muted-foreground">
                            Used {preset.usage_count} times
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <Button
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation()
                      handlePresetApply(preset.preset_id)
                    }}
                    disabled={isApplying === preset.preset_id}
                    className="text-xs"
                  >
                    {isApplying === preset.preset_id ? (
                      <>
                        <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                        Applying...
                      </>
                    ) : (
                      "Apply"
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Preset Details Dialog */}
      <Dialog 
        open={showPresetDetails !== null} 
        onOpenChange={(open) => !open && setShowPresetDetails(null)}
      >
        <DialogContent>
          {showPresetDetails && (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center space-x-2">
                  {presetIcons[filteredPresets.find(p => p.preset_id === showPresetDetails)?.category || 'default']}
                  <span>{filteredPresets.find(p => p.preset_id === showPresetDetails)?.name}</span>
                </DialogTitle>
                <DialogDescription>
                  {filteredPresets.find(p => p.preset_id === showPresetDetails)?.description}
                </DialogDescription>
              </DialogHeader>
              
              <div className="space-y-4">
                {/* Quick parameters preview */}
                {(() => {
                  const preset = filteredPresets.find(p => p.preset_id === showPresetDetails)
                  if (!preset) return null
                  
                  return (
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Temperature:</span>
                        <span className="ml-2 font-mono">{preset.base_config.temperature}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Max Tokens:</span>
                        <span className="ml-2 font-mono">{preset.base_config.max_tokens}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Top-p:</span>
                        <span className="ml-2 font-mono">{preset.base_config.top_p}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Top-k:</span>
                        <span className="ml-2 font-mono">{preset.base_config.top_k}</span>
                      </div>
                    </div>
                  )
                })()}
                
                <div className="flex justify-end space-x-2">
                  <Button 
                    variant="outline" 
                    onClick={() => setShowPresetDetails(null)}
                  >
                    Close
                  </Button>
                  <Button 
                    onClick={() => {
                      const preset = filteredPresets.find(p => p.preset_id === showPresetDetails)
                      if (preset) {
                        handlePresetApply(preset.preset_id)
                        setShowPresetDetails(null)
                      }
                    }}
                  >
                    Apply Preset
                  </Button>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}