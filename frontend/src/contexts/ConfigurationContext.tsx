import React, { createContext, useContext, useReducer, useCallback, useEffect } from "react"
import { toast } from "react-hot-toast"
import type { 
  DualModelConfig, 
  ModelCapabilities,
  ConfigValidationResponse,
  GenerationConfig,
  ModelSpecificConfig
} from "../lib/types"

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
import {
  getSessionConfig,
  updateSessionConfig,
  validateSessionConfig,
  getModelCapabilities,
  listConfigPresets,
  applyPresetToSession
} from "../lib/api"

// Configuration State
interface ConfigurationState {
  // Current configuration
  currentConfig: DualModelConfig | null
  
  // UI state
  isPanelOpen: boolean
  isApplyingConfig: boolean
  isSavingPreset: boolean
  isLoadingPresets: boolean
  
  // Validation
  validationResults: ConfigValidationResponse | null
  
  // Presets
  presets: ConfigPreset[]
  presetCategories: string[]
  
  // Model capabilities
  modelCapabilities: Record<string, ModelCapabilities>
  
  // Error state
  error: string | null
  
  // Last update timestamp
  lastUpdated: string | null
}

// Configuration Actions
type ConfigurationAction =
  | { type: 'SET_CONFIG'; payload: DualModelConfig }
  | { type: 'UPDATE_CONFIG'; payload: Partial<DualModelConfig> }
  | { type: 'SET_PANEL_OPEN'; payload: boolean }
  | { type: 'SET_APPLYING_CONFIG'; payload: boolean }
  | { type: 'SET_SAVING_PRESET'; payload: boolean }
  | { type: 'SET_LOADING_PRESETS'; payload: boolean }
  | { type: 'SET_VALIDATION_RESULTS'; payload: ConfigValidationResponse | null }
  | { type: 'SET_PRESETS'; payload: { presets: ConfigPreset[]; categories: string[] } }
  | { type: 'ADD_PRESET'; payload: ConfigPreset }
  | { type: 'SET_MODEL_CAPABILITIES'; payload: { modelId: string; capabilities: ModelCapabilities } }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_LAST_UPDATED'; payload: string }

// Configuration Context
interface ConfigurationContextType {
  state: ConfigurationState
  
  // Config management
  loadSessionConfig: (sessionId: string) => Promise<void>
  updateConfig: (updates: Partial<DualModelConfig>) => void
  applyConfig: (sessionId: string) => Promise<void>
  resetConfig: (sessionId: string) => void
  validateConfig: (config: DualModelConfig) => Promise<void>
  
  // Model config management
  updateBaseModelConfig: (updates: Partial<ModelSpecificConfig>) => void
  updateFinetunedModelConfig: (updates: Partial<ModelSpecificConfig>) => void
  syncParameters: (enabled: boolean) => void
  
  // Preset management
  loadPresets: (category?: string) => Promise<void>
  applyPreset: (sessionId: string, presetId: string) => Promise<void>
  saveCurrentAsPreset: (sessionId: string, name: string, description: string, category: string) => Promise<ConfigPreset | void>
  
  // Model capabilities
  loadModelCapabilities: (modelId: string) => Promise<void>
  
  // UI management
  togglePanel: () => void
  setPanelOpen: (open: boolean) => void
  
  // Utility functions
  createDefaultConfig: (sessionId: string) => DualModelConfig
  getParameterConstraints: (modelId: string) => Record<string, any> | null
  validateParameter: (param: string, value: any, modelId?: string) => boolean
}

// Default state
const initialState: ConfigurationState = {
  currentConfig: null,
  isPanelOpen: false,
  isApplyingConfig: false,
  isSavingPreset: false,
  isLoadingPresets: false,
  validationResults: null,
  presets: [],
  presetCategories: [],
  modelCapabilities: {},
  error: null,
  lastUpdated: null
}

// Reducer
function configurationReducer(state: ConfigurationState, action: ConfigurationAction): ConfigurationState {
  switch (action.type) {
    case 'SET_CONFIG':
      return {
        ...state,
        currentConfig: action.payload,
        lastUpdated: new Date().toISOString(),
        error: null
      }
      
    case 'UPDATE_CONFIG':
      if (!state.currentConfig) return state
      return {
        ...state,
        currentConfig: {
          ...state.currentConfig,
          ...action.payload,
          updated_at: new Date().toISOString()
        },
        lastUpdated: new Date().toISOString()
      }
      
    case 'SET_PANEL_OPEN':
      return { ...state, isPanelOpen: action.payload }
      
    case 'SET_APPLYING_CONFIG':
      return { ...state, isApplyingConfig: action.payload }
      
    case 'SET_SAVING_PRESET':
      return { ...state, isSavingPreset: action.payload }
      
    case 'SET_LOADING_PRESETS':
      return { ...state, isLoadingPresets: action.payload }
      
    case 'SET_VALIDATION_RESULTS':
      return { ...state, validationResults: action.payload }
      
    case 'SET_PRESETS':
      return {
        ...state,
        presets: action.payload.presets,
        presetCategories: action.payload.categories
      }
      
    case 'ADD_PRESET':
      return {
        ...state,
        presets: [...state.presets, action.payload]
      }
      
    case 'SET_MODEL_CAPABILITIES':
      return {
        ...state,
        modelCapabilities: {
          ...state.modelCapabilities,
          [action.payload.modelId]: action.payload.capabilities
        }
      }
      
    case 'SET_ERROR':
      return { ...state, error: action.payload }
      
    case 'SET_LAST_UPDATED':
      return { ...state, lastUpdated: action.payload }
      
    default:
      return state
  }
}

// Create context
const ConfigurationContext = createContext<ConfigurationContextType | undefined>(undefined)

// Configuration Provider
export function ConfigurationProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(configurationReducer, initialState)

  // Create default configuration
  const createDefaultConfig = useCallback((sessionId: string): DualModelConfig => {
    const defaultGenConfig: GenerationConfig = {
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

    const defaultModelConfig: ModelSpecificConfig = {
      system_prompt: "",
      context_length: 2048,
      generation_config: { ...defaultGenConfig },
      model_id: null,
      enabled: true
    }

    return {
      session_id: sessionId,
      base_model: { ...defaultModelConfig },
      finetuned_model: { ...defaultModelConfig },
      sync_parameters: false,
      config_name: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }
  }, [])

  // Load session configuration
  const loadSessionConfig = useCallback(async (sessionId: string) => {
    try {
      const config = await getSessionConfig(sessionId) as DualModelConfig
      dispatch({ type: 'SET_CONFIG', payload: config })
    } catch (error) {
      console.error("Failed to load session config:", error)
      // Create default config on failure
      const defaultConfig = createDefaultConfig(sessionId)
      dispatch({ type: 'SET_CONFIG', payload: defaultConfig })
    }
  }, [createDefaultConfig])

  // Update configuration
  const updateConfig = useCallback((updates: Partial<DualModelConfig>) => {
    dispatch({ type: 'UPDATE_CONFIG', payload: updates })
  }, [])

  // Apply configuration
  const applyConfig = useCallback(async (sessionId: string) => {
    if (!state.currentConfig) return

    try {
      dispatch({ type: 'SET_APPLYING_CONFIG', payload: true })
      await updateSessionConfig(sessionId, state.currentConfig)
      toast.success("Configuration applied successfully")
    } catch (error) {
      console.error("Failed to apply configuration:", error)
      toast.error("Failed to apply configuration")
      dispatch({ type: 'SET_ERROR', payload: "Failed to apply configuration" })
    } finally {
      dispatch({ type: 'SET_APPLYING_CONFIG', payload: false })
    }
  }, [state.currentConfig])

  // Reset configuration
  const resetConfig = useCallback((sessionId: string) => {
    const defaultConfig = createDefaultConfig(sessionId)
    dispatch({ type: 'SET_CONFIG', payload: defaultConfig })
    toast.success("Configuration reset to defaults")
  }, [createDefaultConfig])

  // Validate configuration
  const validateConfig = useCallback(async (config: DualModelConfig) => {
    try {
      const results = await validateSessionConfig(config.session_id, config) as ConfigValidationResponse
      dispatch({ type: 'SET_VALIDATION_RESULTS', payload: results })
    } catch (error) {
      console.error("Validation failed:", error)
      dispatch({ type: 'SET_VALIDATION_RESULTS', payload: null })
    }
  }, [])

  // Update base model configuration
  const updateBaseModelConfig = useCallback((updates: Partial<ModelSpecificConfig>) => {
    if (!state.currentConfig) return

    const newConfig = {
      ...state.currentConfig,
      base_model: { ...state.currentConfig.base_model, ...updates }
    }

    // Sync parameters if enabled
    if (state.currentConfig.sync_parameters && updates.generation_config) {
      newConfig.finetuned_model.generation_config = {
        ...newConfig.finetuned_model.generation_config,
        ...updates.generation_config
      }
    }

    dispatch({ type: 'SET_CONFIG', payload: newConfig })
  }, [state.currentConfig])

  // Update fine-tuned model configuration
  const updateFinetunedModelConfig = useCallback((updates: Partial<ModelSpecificConfig>) => {
    if (!state.currentConfig) return

    const newConfig = {
      ...state.currentConfig,
      finetuned_model: { ...state.currentConfig.finetuned_model, ...updates }
    }

    // Sync parameters if enabled
    if (state.currentConfig.sync_parameters && updates.generation_config) {
      newConfig.base_model.generation_config = {
        ...newConfig.base_model.generation_config,
        ...updates.generation_config
      }
    }

    dispatch({ type: 'SET_CONFIG', payload: newConfig })
  }, [state.currentConfig])

  // Toggle parameter sync
  const syncParameters = useCallback((enabled: boolean) => {
    if (!state.currentConfig) return

    const newConfig = {
      ...state.currentConfig,
      sync_parameters: enabled
    }

    if (enabled) {
      // Sync finetuned config to base config
      newConfig.finetuned_model.generation_config = {
        ...state.currentConfig.base_model.generation_config
      }
    }

    dispatch({ type: 'SET_CONFIG', payload: newConfig })
  }, [state.currentConfig])

  // Load presets
  const loadPresets = useCallback(async (category?: string) => {
    try {
      dispatch({ type: 'SET_LOADING_PRESETS', payload: true })
      const response = await listConfigPresets(category) as { presets: ConfigPreset[]; categories: string[] }
      dispatch({ 
        type: 'SET_PRESETS', 
        payload: { 
          presets: response.presets, 
          categories: response.categories 
        } 
      })
    } catch (error) {
      console.error("Failed to load presets:", error)
      toast.error("Failed to load configuration presets")
    } finally {
      dispatch({ type: 'SET_LOADING_PRESETS', payload: false })
    }
  }, [])

  // Apply preset
  const applyPreset = useCallback(async (sessionId: string, presetId: string) => {
    try {
      const config = await applyPresetToSession(sessionId, presetId) as DualModelConfig
      dispatch({ type: 'SET_CONFIG', payload: config })
      toast.success("Preset applied successfully")
    } catch (error) {
      console.error("Failed to apply preset:", error)
      toast.error("Failed to apply preset")
    }
  }, [])

  // Save current configuration as preset
  const saveCurrentAsPreset = useCallback(async (
    _sessionId: string, 
    name: string, 
    description: string, 
    category: string
  ) => {
    if (!state.currentConfig) return

    try {
      dispatch({ type: 'SET_SAVING_PRESET', payload: true })
      
      // Create preset from current configuration
      const presetData = {
        name,
        description,
        category,
        base_config: state.currentConfig.base_model.generation_config,
        finetuned_config: state.currentConfig.finetuned_model.generation_config,
        system_prompts: {
          base: state.currentConfig.base_model.system_prompt,
          finetuned: state.currentConfig.finetuned_model.system_prompt
        }
      }
      
      // Import createConfigPreset here to avoid circular dependency
      const { createConfigPreset } = await import('../lib/api')
      const newPreset = await createConfigPreset(presetData) as ConfigPreset
      
      // Add to local state
      dispatch({ type: 'ADD_PRESET', payload: newPreset })
      
      toast.success("Preset saved successfully")
      return newPreset
    } catch (error) {
      console.error("Failed to save preset:", error)
      toast.error("Failed to save preset")
      throw error
    } finally {
      dispatch({ type: 'SET_SAVING_PRESET', payload: false })
    }
  }, [state.currentConfig])

  // Load model capabilities
  const loadModelCapabilities = useCallback(async (modelId: string) => {
    try {
      const capabilities = await getModelCapabilities(modelId) as ModelCapabilities
      dispatch({ 
        type: 'SET_MODEL_CAPABILITIES', 
        payload: { modelId, capabilities } 
      })
    } catch (error) {
      console.error("Failed to load model capabilities:", error)
    }
  }, [])

  // UI management
  const togglePanel = useCallback(() => {
    dispatch({ type: 'SET_PANEL_OPEN', payload: !state.isPanelOpen })
  }, [state.isPanelOpen])

  const setPanelOpen = useCallback((open: boolean) => {
    dispatch({ type: 'SET_PANEL_OPEN', payload: open })
  }, [])

  // Utility functions
  const getParameterConstraints = useCallback((modelId: string): Record<string, any> | null => {
    return state.modelCapabilities[modelId]?.parameter_constraints || null
  }, [state.modelCapabilities])

  const validateParameter = useCallback((param: string, value: any, modelId?: string): boolean => {
    if (!modelId) return true
    
    const constraints = getParameterConstraints(modelId)
    if (!constraints || !constraints[param]) return true

    const constraint = constraints[param]
    if (constraint.min !== undefined && value < constraint.min) return false
    if (constraint.max !== undefined && value > constraint.max) return false
    
    return true
  }, [getParameterConstraints])

  // Auto-validate configuration when it changes
  useEffect(() => {
    if (state.currentConfig) {
      const timer = setTimeout(() => {
        validateConfig(state.currentConfig!)
      }, 1000)
      return () => clearTimeout(timer)
    }
  }, [state.currentConfig, validateConfig])

  const contextValue: ConfigurationContextType = {
    state,
    loadSessionConfig,
    updateConfig,
    applyConfig,
    resetConfig,
    validateConfig,
    updateBaseModelConfig,
    updateFinetunedModelConfig,
    syncParameters,
    loadPresets,
    applyPreset,
    saveCurrentAsPreset,
    loadModelCapabilities,
    togglePanel,
    setPanelOpen,
    createDefaultConfig,
    getParameterConstraints,
    validateParameter
  }

  return (
    <ConfigurationContext.Provider value={contextValue}>
      {children}
    </ConfigurationContext.Provider>
  )
}

// Hook to use configuration context
export function useConfiguration() {
  const context = useContext(ConfigurationContext)
  if (context === undefined) {
    throw new Error('useConfiguration must be used within a ConfigurationProvider')
  }
  return context
}