import { 
  BookOpen, 
  Home, 
  Settings, 
  Workflow, 
  Brain,
  FileText,
  Zap,
  Database,
  BarChart3,
  MessageSquare,
  Code2,
  Terminal,
  Lightbulb,
  AlertCircle,
  CheckCircle,
  ExternalLink
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface DocsPageProps {
  onNavigate: (page: "home" | "training" | "workflow" | "docs" | "chat" | "analytics") => void
}

const docSections = [
  {
    title: "Getting Started",
    icon: Lightbulb,
    items: [
      { title: "Quick Start Guide", description: "Set up your first fine-tuning job in 5 minutes" },
      { title: "Platform Overview", description: "Understanding the LLMForge interface and features" },
      { title: "System Requirements", description: "Hardware and software prerequisites" }
    ]
  },
  {
    title: "Model Training",
    icon: Zap,
    items: [
      { title: "Model Selection", description: "Choosing the right base model for your use case" },
      { title: "Data Preparation", description: "Formatting and preparing your training datasets" },
      { title: "Training Configuration", description: "Hyperparameters, LoRA settings, and optimization" },
      { title: "Monitoring Training", description: "Real-time metrics and progress tracking" }
    ]
  },
  {
    title: "Data Management",
    icon: Database,
    items: [
      { title: "Dataset Formats", description: "Supported formats: JSON, CSV, JSONL" },
      { title: "Data Validation", description: "Ensuring data quality and format compliance" },
      { title: "Custom Datasets", description: "Creating domain-specific training data" }
    ]
  },
  {
    title: "Model Comparison",
    icon: MessageSquare,
    items: [
      { title: "Chat Interface", description: "Interactive testing of trained models" },
      { title: "A/B Testing", description: "Comparing model versions side by side" },
      { title: "Performance Metrics", description: "Evaluation and benchmarking tools" }
    ]
  },
  {
    title: "Analytics & Insights",
    icon: BarChart3,
    items: [
      { title: "Training Metrics", description: "Loss curves, learning rates, and convergence analysis" },
      { title: "Resource Usage", description: "GPU utilization and memory consumption" },
      { title: "Export Data", description: "Downloading metrics and training logs" }
    ]
  },
  {
    title: "Advanced Features",
    icon: Code2,
    items: [
      { title: "API Integration", description: "Programmatic access to training functions" },
      { title: "Custom Configurations", description: "YAML-based training setups" },
      { title: "Batch Processing", description: "Running multiple training jobs" }
    ]
  }
]

const quickLinks = [
  { title: "Start Training", icon: Zap, route: "training" as const, description: "Begin fine-tuning immediately" },
  { title: "View Workflows", icon: Workflow, route: "workflow" as const, description: "Manage training pipelines" },
  { title: "Model Chat", icon: MessageSquare, route: "chat" as const, description: "Test your models" }
]

export default function DocsPage({ onNavigate }: DocsPageProps) {
  return (
    <div className="min-h-screen bg-background">
      {/* Navigation Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto max-w-7xl">
          <div className="flex h-16 items-center justify-between px-4">
            <div className="flex items-center space-x-3">
              <Brain className="h-8 w-8 text-primary" />
              <h1 className="text-2xl font-bold tracking-tight">LLMForge</h1>
            </div>

            <nav className="hidden md:flex items-center space-x-1">
              <Button variant="ghost" onClick={() => onNavigate("home")} className="text-sm">
                <Home className="mr-2 h-4 w-4" />
                Home
              </Button>
              <Button variant="ghost" onClick={() => onNavigate("training")} className="text-sm">
                <Settings className="mr-2 h-4 w-4" />
                Fine-tuning
              </Button>
              <Button variant="ghost" onClick={() => onNavigate("workflow")} className="text-sm">
                <Workflow className="mr-2 h-4 w-4" />
                Workflows
              </Button>
              <Button variant="default" className="text-sm">
                <BookOpen className="mr-2 h-4 w-4" />
                Documentation
              </Button>
            </nav>
          </div>
        </div>
      </header>

      <main className="container mx-auto max-w-7xl p-4 md:p-6 lg:p-8 space-y-8">
        {/* Page Header */}
        <div className="space-y-4">
          <div className="flex items-center space-x-3">
            <div className="rounded-lg bg-primary/10 p-2">
              <BookOpen className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Documentation</h1>
              <p className="text-muted-foreground">Complete guide to fine-tuning language models with LLMForge</p>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-4 gap-8">
          {/* Main Documentation Content */}
          <div className="lg:col-span-3 space-y-8">
            {/* Overview */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="h-5 w-5" />
                  <span>Platform Overview</span>
                </CardTitle>
                <CardDescription>
                  LLMForge is a comprehensive platform for fine-tuning large language models with enterprise-grade features
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="prose prose-sm max-w-none dark:prose-invert">
                  <p>
                    Our platform simplifies the complex process of fine-tuning language models by providing:
                  </p>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-primary" />
                      <span>Intuitive web interface for model training</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-primary" />
                      <span>Real-time monitoring and analytics</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-primary" />
                      <span>Automated data preparation and validation</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-primary" />
                      <span>Advanced comparison and testing tools</span>
                    </li>
                  </ul>
                </div>

                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    New to fine-tuning? Start with our <Button variant="link" className="p-0 h-auto" onClick={() => onNavigate("training")}>Quick Start Guide</Button> 
                    to get your first model training in minutes.
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>

            {/* Documentation Sections */}
            <div className="space-y-6">
              <h2 className="text-2xl font-bold tracking-tight">Documentation Sections</h2>
              
              <div className="grid md:grid-cols-2 gap-6">
                {docSections.map((section, index) => {
                  const Icon = section.icon
                  return (
                    <Card key={index} className="group hover:shadow-lg transition-all duration-200">
                      <CardHeader>
                        <CardTitle className="flex items-center space-x-3">
                          <div className="rounded-md bg-primary/10 p-2">
                            <Icon className="h-5 w-5 text-primary" />
                          </div>
                          <span>{section.title}</span>
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        {section.items.map((item, itemIndex) => (
                          <div key={itemIndex} className="space-y-1">
                            <div className="flex items-center justify-between">
                              <h4 className="text-sm font-medium">{item.title}</h4>
                              <ExternalLink className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                            </div>
                            <p className="text-xs text-muted-foreground">{item.description}</p>
                          </div>
                        ))}
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            </div>

            {/* Code Example */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Terminal className="h-5 w-5" />
                  <span>Example Configuration</span>
                </CardTitle>
                <CardDescription>Sample YAML configuration for training setup</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="bg-muted rounded-md p-4 font-mono text-sm">
                  <div className="space-y-1">
                    <div><span className="text-primary">model_name:</span> <span className="text-green-400">"microsoft/DialoGPT-medium"</span></div>
                    <div><span className="text-primary">max_seq_length:</span> <span className="text-blue-400">2048</span></div>
                    <div><span className="text-primary">learning_rate:</span> <span className="text-blue-400">2e-4</span></div>
                    <div><span className="text-primary">num_train_epochs:</span> <span className="text-blue-400">3</span></div>
                    <div><span className="text-primary">per_device_train_batch_size:</span> <span className="text-blue-400">4</span></div>
                    <div><span className="text-primary">lora_config:</span></div>
                    <div className="ml-4"><span className="text-primary">r:</span> <span className="text-blue-400">16</span></div>
                    <div className="ml-4"><span className="text-primary">lora_alpha:</span> <span className="text-blue-400">32</span></div>
                    <div className="ml-4"><span className="text-primary">lora_dropout:</span> <span className="text-blue-400">0.1</span></div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Quick Links */}
            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
                <CardDescription>Jump directly to key features</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {quickLinks.map((link, index) => {
                  const Icon = link.icon
                  return (
                    <Button
                      key={index}
                      variant="outline"
                      className="w-full justify-start text-left h-auto p-3"
                      onClick={() => onNavigate(link.route)}
                    >
                      <Icon className="mr-3 h-4 w-4 text-primary" />
                      <div>
                        <div className="font-medium">{link.title}</div>
                        <div className="text-xs text-muted-foreground">{link.description}</div>
                      </div>
                    </Button>
                  )
                })}
              </CardContent>
            </Card>

            {/* Version Info */}
            <Card>
              <CardHeader>
                <CardTitle>Platform Info</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span>Version</span>
                  <Badge variant="outline">v2.1.0</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span>API Status</span>
                  <Badge className="bg-primary/10 text-primary">Online</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Models</span>
                  <span className="text-muted-foreground">15+ supported</span>
                </div>
                <Separator />
                <div className="text-xs text-muted-foreground">
                  Built with Unsloth, Hugging Face, and modern AI frameworks
                </div>
              </CardContent>
            </Card>

            {/* Support */}
            <Card>
              <CardHeader>
                <CardTitle>Need Help?</CardTitle>
                <CardDescription>Get support and join the community</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button variant="outline" className="w-full justify-start">
                  <MessageSquare className="mr-2 h-4 w-4" />
                  Community Forum
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <ExternalLink className="mr-2 h-4 w-4" />
                  GitHub Repository
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <FileText className="mr-2 h-4 w-4" />
                  API Reference
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  )
}