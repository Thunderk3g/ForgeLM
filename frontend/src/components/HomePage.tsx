import { useState } from "react"
import { 
  Brain, 
  ArrowRight, 
  Zap, 
  MessageSquare, 
  BarChart3, 
  Settings,
  Workflow,
  BookOpen,
  Home,
  Code2,
  Database,
  Play,
  Users,
  Shield,
  Rocket
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface HomePageProps {
  onNavigate: (page: "home" | "training" | "workflow" | "docs" | "chat" | "analytics") => void
  apiStatus: "checking" | "online" | "offline"
}

const featureCards = [
  {
    icon: Zap,
    title: "Quick Start Training",
    description: "Get started with fine-tuning in minutes using our streamlined workflow",
    action: "Start Training",
    route: "training" as const,
    color: "text-primary",
    bgColor: "bg-primary/10"
  },
  {
    icon: Workflow,
    title: "Training Workflows",
    description: "Visualize and manage your complete fine-tuning pipeline",
    action: "View Workflows",
    route: "workflow" as const,
    color: "text-secondary",
    bgColor: "bg-secondary/10"
  },
  {
    icon: MessageSquare,
    title: "Model Comparison",
    description: "Test and compare different model versions side by side",
    action: "Compare Models",
    route: "chat" as const,
    color: "text-chart-1",
    bgColor: "bg-chart-1/10"
  },
  {
    icon: BarChart3,
    title: "Training Analytics",
    description: "Monitor performance, metrics, and training insights",
    action: "View Analytics",
    route: "analytics" as const,
    color: "text-chart-2",
    bgColor: "bg-chart-2/10"
  }
]

const statsCards = [
  { label: "Models Supported", value: "15+", icon: Database },
  { label: "Active Projects", value: "3", icon: Play },
  { label: "Training Sessions", value: "127", icon: Users },
  { label: "Success Rate", value: "98%", icon: Shield }
]

export default function HomePage({ onNavigate, apiStatus }: HomePageProps) {
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
              <Button variant="default" className="text-sm">
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
              <Button variant="ghost" onClick={() => onNavigate("analytics")} className="text-sm">
                <BarChart3 className="mr-2 h-4 w-4" />
                Analytics
              </Button>
              <Button variant="ghost" onClick={() => onNavigate("docs")} className="text-sm">
                <BookOpen className="mr-2 h-4 w-4" />
                Documentation
              </Button>
            </nav>

            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 rounded-md border px-3 py-1.5">
                <div className={`h-2 w-2 rounded-full ${
                  apiStatus === 'online' ? 'bg-primary' : 
                  apiStatus === 'offline' ? 'bg-destructive' : 
                  'bg-secondary'
                }`} />
                <span className="text-sm font-medium">
                  {apiStatus === 'checking' ? 'Checking...' : 'API ' + apiStatus}
                </span>
              </div>

              <Button onClick={() => onNavigate("training")}>
                Get Started
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto max-w-7xl p-4 md:p-6 lg:p-8 space-y-12">
        {/* Hero Section */}
        <div className="text-center space-y-6 py-12">
          <div className="space-y-4">
            <Badge className="mx-auto">
              <Rocket className="mr-2 h-3 w-3" />
              Fine-tune LLMs with Ease
            </Badge>
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
              Professional LLM
              <span className="text-primary block">Fine-tuning Platform</span>
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Train, compare, and deploy custom language models with our intuitive platform. 
              Get production-ready results with minimal configuration.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" onClick={() => onNavigate("training")} className="text-lg px-8 py-3">
              <Play className="mr-2 h-5 w-5" />
              Start Training Now
            </Button>
            <Button size="lg" variant="outline" onClick={() => onNavigate("workflow")} className="text-lg px-8 py-3">
              <Workflow className="mr-2 h-5 w-5" />
              View Workflows
            </Button>
          </div>
        </div>

        {/* Stats Section */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {statsCards.map((stat, index) => {
            const Icon = stat.icon
            return (
              <Card key={index} className="text-center">
                <CardHeader className="pb-3">
                  <Icon className="h-8 w-8 mx-auto text-primary" />
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="text-2xl font-bold">{stat.value}</div>
                  <p className="text-sm text-muted-foreground">{stat.label}</p>
                </CardContent>
              </Card>
            )
          })}
        </div>

        {/* Features Section */}
        <div className="space-y-8">
          <div className="text-center space-y-4">
            <h2 className="text-3xl font-bold tracking-tight">Everything You Need</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Complete toolkit for fine-tuning language models with professional-grade features
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {featureCards.map((feature, index) => {
              const Icon = feature.icon
              return (
                <Card key={index} className="group hover:shadow-lg transition-all duration-200 cursor-pointer border-border/50 hover:border-border"
                      onClick={() => onNavigate(feature.route)}>
                  <CardHeader className="pb-4">
                    <div className={`rounded-lg ${feature.bgColor} p-3 w-fit`}>
                      <Icon className={`h-6 w-6 ${feature.color}`} />
                    </div>
                    <CardTitle className="group-hover:text-primary transition-colors">{feature.title}</CardTitle>
                    <CardDescription>{feature.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <Button variant="ghost" className="w-full justify-between group-hover:bg-accent">
                      {feature.action}
                      <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
                    </Button>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </div>

        {/* Technology Stack */}
        <div className="space-y-8">
          <div className="text-center space-y-4">
            <h2 className="text-3xl font-bold tracking-tight">Built with Modern Tools</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Leveraging the latest in AI research and development frameworks
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <Card className="text-center">
              <CardHeader>
                <Code2 className="h-12 w-12 mx-auto text-primary" />
                <CardTitle>Unsloth Integration</CardTitle>
                <CardDescription>
                  2x faster training with memory optimization and LoRA adapters
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="text-center">
              <CardHeader>
                <Database className="h-12 w-12 mx-auto text-secondary" />
                <CardTitle>Hugging Face Models</CardTitle>
                <CardDescription>
                  Access to thousands of pre-trained models and datasets
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="text-center">
              <CardHeader>
                <BarChart3 className="h-12 w-12 mx-auto text-chart-1" />
                <CardTitle>Real-time Monitoring</CardTitle>
                <CardDescription>
                  Live metrics, loss tracking, and training insights
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center py-12 space-y-6">
          <h2 className="text-3xl font-bold tracking-tight">Ready to Get Started?</h2>
          <p className="text-lg text-muted-foreground max-w-xl mx-auto">
            Begin your fine-tuning journey with our step-by-step guided workflow
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" onClick={() => onNavigate("training")} className="text-lg px-8 py-3">
              <Zap className="mr-2 h-5 w-5" />
              Start Fine-tuning
            </Button>
            <Button size="lg" variant="outline" onClick={() => onNavigate("docs")} className="text-lg px-8 py-3">
              <BookOpen className="mr-2 h-5 w-5" />
              Read Documentation
            </Button>
          </div>
        </div>
      </main>
    </div>
  )
}