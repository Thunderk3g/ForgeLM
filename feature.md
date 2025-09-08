# LLMForge Data Visualization & Auto Data Generation Feature Development Prompt

## Context & Current Architecture Analysis

Based on the LLMForge repository analysis, you have a sophisticated fine-tuning platform with:

### Current Strengths:
- **Advanced ML Backend**: Unsloth optimization, multi-stage training (SFT→RM→PPO→DPO), advanced PEFT methods
- **React Frontend**: Modern UI with real-time training monitoring via SSE
- **Template System**: Flexible dataset formatting (Alpaca, ChatML, ShareGPT)
- **Research Analytics**: Basic comparison logging in `utils/research_logger.py`
- **Model Comparison**: Dual inference with performance metrics
- **Multi-Processing Training**: Non-blocking API with process isolation

### Current Gaps for Data Visualization:
- Limited visualization beyond basic training metrics
- No interactive data exploration tools
- Missing comprehensive analytics dashboards
- No automated dataset generation capabilities
- Research analytics are text-based without visual insights

## Feature Requirements Specification

### 1. Advanced Data Visualization System

#### A. Training Analytics Dashboard
Create a comprehensive training visualization system that enhances the existing real-time monitoring:

**Components Needed:**
- **Training Metrics Visualizer**: Extend the existing SSE streaming to include:
  - Loss curves with multiple smoothing options
  - Learning rate schedules with annotations
  - GPU memory usage over time
  - Gradient norms and parameter statistics
  - Comparison between different training runs
  - Interactive timeline with zoom/pan capabilities

- **Model Performance Analytics**:
  - Validation metrics over epochs
  - PEFT adapter efficiency visualization
  - Parameter count vs performance scatter plots
  - Training speed benchmarks (leverage existing Unsloth 2x speedup metrics)

- **Resource Utilization Monitor**:
  - Real-time GPU memory usage (integrate with existing `utils/memory.py`)
  - CPU utilization during multiprocessing training
  - Training throughput (samples/second)
  - Memory optimization impact visualization

#### B. Dataset Analytics & Exploration
Build on the existing template system and data preparation pipeline:

**Features Required:**
- **Dataset Composition Analysis**:
  - Token distribution histograms
  - Sequence length statistics
  - Template format coverage (Alpaca, ChatML, etc.)
  - Data quality metrics and outlier detection
  - Cross-template compatibility visualization

- **Interactive Data Explorer**:
  - Sample browser with filtering capabilities
  - Template preview with before/after formatting
  - Data augmentation impact visualization
  - Balance analysis for supervised tasks

#### C. Model Comparison Visualization
Enhance the existing comparison service (`services/comparison_service.py`):

**Advanced Analytics:**
- **Response Quality Analysis**:
  - Response length distributions
  - Similarity heatmaps between base and fine-tuned models
  - Performance timing comparisons
  - Cache hit rate visualizations (FastLibra optimization metrics)

- **Research Analytics Dashboard**:
  - Extend `utils/research_logger.py` with interactive visualizations
  - Session-based comparison trends
  - Model pair performance matrices
  - A/B testing result visualization

### 2. Automated Data Generation System

#### A. Template-Aware Data Generator
Integrate with the existing template system (`models/templates.py`):

**Core Capabilities:**
- **Multi-Format Generation**: Generate data in all supported formats (Alpaca, ChatML, ShareGPT, Qwen)
- **Task-Specific Generation**: 
  - Instruction-tuning datasets
  - Conversation datasets
  - Domain-specific training data
  - Code generation datasets
  - Mathematical reasoning datasets

- **Quality Control**:
  - Automatic data validation using existing template validators
  - Diversity metrics to ensure varied training examples
  - Length distribution control
  - Difficulty progression

#### B. Model-Driven Generation Pipeline
Leverage available models in the system:

**Generation Strategies:**
- **Self-Supervised Generation**: Use base models to generate instruction-response pairs
- **Iterative Refinement**: Use fine-tuned models to improve generated data quality
- **Multi-Model Consensus**: Generate data using multiple models and filter by agreement
- **Domain Adaptation**: Generate domain-specific data based on user requirements

## Implementation Architecture

### Frontend Components (React + TypeScript)

```typescript
// New Components to Add to frontend/src/components/

interface DataVisualizationProps {
  trainingMetrics: TrainingMetrics[];
  datasetStats: DatasetAnalytics;
  comparisonData: ComparisonMetrics[];
  realTimeUpdates: boolean;
}

interface DataGenerationProps {
  availableModels: ModelPair[];
  templateFormats: TemplateType[];
  generationConfig: GenerationConfig;
  onDataGenerated: (dataset: GeneratedDataset) => void;
}

// Integration Points:
// - Extend existing TrainingForm component
// - Enhance ComparisonResponse display
// - Add new routes to existing React Router setup
// - Integrate with current SSE streaming system
```

### Backend Integration Points

```python
# New Services to Add:

class VisualizationService:
    """Extends existing services with visualization data preparation"""
    def __init__(self, training_service, comparison_service, research_logger):
        # Integrate with existing services
        pass
    
    def get_training_analytics(self, task_id: str) -> TrainingAnalytics:
        # Enhance existing training status monitoring
        pass
    
    def get_dataset_analytics(self, dataset_path: str) -> DatasetAnalytics:
        # Analyze datasets using existing template system
        pass

class DataGenerationService:
    """New service for automated data generation"""
    def __init__(self, model_service, template_service):
        # Use existing ModelService and template system
        pass
    
    def generate_training_data(self, config: GenerationConfig) -> GeneratedDataset:
        # Generate data using available models
        pass
    
    def validate_generated_data(self, data: List[Dict]) -> ValidationReport:
        # Use existing template validation
        pass

# New Controllers:
class VisualizationController:
    """API endpoints for visualization data"""
    
class DataGenerationController:
    """API endpoints for data generation"""
```

### Database/Storage Integration

```python
# Extend existing research logging:
# - Add visualization metadata to research_logger.py
# - Store generated datasets in existing data structure
# - Cache visualization data for performance

# File Structure Extensions:
# data/
# ├── generated/           # New: Auto-generated datasets
# ├── visualizations/      # New: Cached visualization data
# ├── analytics/          # New: Processed analytics data
# └── existing structure...
```

## Detailed Feature Specifications

### 1. Training Visualization Enhancement

**Integration with Existing SSE System:**
```javascript
// Extend existing SSE handling in frontend
const enhancedSSEHandler = (eventData) => {
  const { type, data } = eventData;
  
  switch(type) {
    case 'training_metrics':
      updateTrainingCharts(data);
      break;
    case 'resource_usage':
      updateResourceCharts(data);
      break;
    case 'model_comparison':
      updateComparisonViz(data);
      break;
    // Extend existing cases...
  }
};
```

**Backend Enhancement:**
```python
# Extend existing training_controller.py SSE streaming
async def enhanced_stream_training_logs(self, task_id: str):
    async def generate_enhanced_logs():
        # Existing log streaming...
        
        # Add visualization data
        if should_send_metrics():
            viz_data = await self.visualization_service.get_training_viz_data(task_id)
            yield f"event: visualization\n"
            yield f"data: {json.dumps(viz_data)}\n\n"
        
        # Add resource monitoring
        resource_data = get_resource_metrics()
        yield f"event: resources\n"
        yield f"data: {json.dumps(resource_data)}\n\n"
```

### 2. Dataset Analytics Implementation

**Data Analysis Pipeline:**
```python
class DatasetAnalyzer:
    """Analyze datasets for visualization"""
    
    def analyze_dataset(self, dataset_path: str) -> DatasetAnalytics:
        # Load using existing dataset loading logic
        dataset = load_from_disk(dataset_path)
        
        return DatasetAnalytics(
            total_samples=len(dataset),
            token_distribution=self._analyze_tokens(dataset),
            length_statistics=self._analyze_lengths(dataset),
            template_coverage=self._analyze_templates(dataset),
            quality_metrics=self._analyze_quality(dataset)
        )
    
    def _analyze_tokens(self, dataset):
        # Use existing tokenizer from model_service
        pass
    
    def _analyze_templates(self, dataset):
        # Use existing template system
        pass
```

### 3. Auto Data Generation Pipeline

**Generation Strategy:**
```python
class AutoDataGenerator:
    """Generate training data using available models"""
    
    def __init__(self, model_service: ModelService, template_service: TemplateService):
        self.model_service = model_service
        self.template_service = template_service
    
    async def generate_instruction_dataset(self, config: GenerationConfig) -> GeneratedDataset:
        # Use existing model loading and inference
        model_pair = await self.model_service.get_best_available_pair()
        
        # Generate using existing comparison infrastructure
        for prompt_seed in config.prompt_seeds:
            response = await self.model_service.generate_comparison_responses(
                session_id=f"generation_{uuid.uuid4()}",
                messages=[{"role": "user", "content": prompt_seed}]
            )
            
            # Use existing template formatting
            formatted_data = await self.template_service.format_dataset(
                response, config.template_name
            )
            
            yield formatted_data
    
    def generate_conversation_dataset(self, config: GenerationConfig):
        # Multi-turn conversation generation
        pass
    
    def generate_domain_specific_dataset(self, domain: str, config: GenerationConfig):
        # Domain-adapted generation
        pass
```

### 4. Visualization Components

**Chart Library Integration:**
```typescript
// Use existing Recharts from the project
import { 
  LineChart, BarChart, ScatterPlot, HeatMap, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend 
} from 'recharts';

interface TrainingVisualizationProps {
  metrics: TrainingMetrics[];
  isRealTime: boolean;
}

const TrainingVisualization: React.FC<TrainingVisualizationProps> = ({ metrics, isRealTime }) => {
  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-2">Training Loss</h3>
        <LineChart width={400} height={200} data={metrics}>
          <XAxis dataKey="step" />
          <YAxis />
          <CartesianGrid strokeDasharray="3 3" />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="loss" stroke="#8884d8" />
          <Line type="monotone" dataKey="eval_loss" stroke="#82ca9d" />
        </LineChart>
      </div>
      
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-2">Learning Rate Schedule</h3>
        <LineChart width={400} height={200} data={metrics}>
          <XAxis dataKey="step" />
          <YAxis />
          <CartesianGrid strokeDasharray="3 3" />
          <Tooltip />
          <Line type="monotone" dataKey="learning_rate" stroke="#ff7300" />
        </LineChart>
      </div>
    </div>
  );
};
```

## Integration Strategy

### Phase 1: Foundation (Weeks 1-2)
1. **Extend Existing Research Logger**: Add visualization data collection to `utils/research_logger.py`
2. **Enhance SSE Streaming**: Extend training controller SSE to include visualization events
3. **Create Basic Visualization Components**: Build on existing React components

### Phase 2: Core Visualization (Weeks 3-4)
1. **Training Analytics Dashboard**: Real-time training metrics visualization
2. **Dataset Analytics**: Integrate with existing data preparation pipeline
3. **Model Comparison Visualization**: Enhance existing comparison service

### Phase 3: Data Generation (Weeks 5-6)
1. **Auto Data Generator**: Build on existing model service and template system
2. **Generation Pipeline**: Integrate with existing training workflow
3. **Quality Validation**: Use existing template validation system

### Phase 4: Advanced Features (Weeks 7-8)
1. **Interactive Analytics**: Advanced filtering and exploration
2. **Export Capabilities**: Integration with existing research export
3. **Performance Optimization**: Leverage existing caching and optimization

## Technical Considerations

### Performance & Scalability
- **Leverage Existing Caching**: Use FastLibra cache optimization for visualization data
- **Streaming Updates**: Build on existing SSE infrastructure
- **Memory Management**: Integrate with existing GPU memory management
- **Background Processing**: Use existing multiprocessing framework

### Security & Privacy
- **Data Privacy**: Respect existing research logging privacy measures
- **Access Control**: Integrate with existing API authentication
- **Resource Management**: Use existing resource limits and monitoring

### Compatibility
- **Template System**: Full compatibility with existing template formats
- **Model Support**: Work with all existing model pairs and PEFT methods
- **API Consistency**: Maintain existing API patterns and responses

## Success Metrics

### User Experience
- **Visualization Responsiveness**: <100ms chart updates during training
- **Data Generation Speed**: Generate 1000 samples in <5 minutes
- **Dashboard Load Time**: <2 seconds for complete analytics dashboard

### System Performance
- **Memory Overhead**: <10% additional memory usage for visualization
- **Training Impact**: <5% impact on training speed
- **API Responsiveness**: Maintain existing API performance during visualization

### Feature Adoption
- **Usage Analytics**: Track visualization feature usage
- **Data Quality**: Measure generated data effectiveness in training
- **User Feedback**: Collect feedback on visualization usefulness

This comprehensive feature set will transform LLMForge from a fine-tuning platform into a complete ML development environment with advanced analytics and automated data generation capabilities.