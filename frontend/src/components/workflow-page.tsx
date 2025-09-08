"use client"

import { useState } from "react"
import {
  Brain,
  ArrowRight,
  Database,
  Settings,
  BarChart3,
  MessageSquare,
  Play,
  Pause,
  CheckCircle,
  Clock,
  Home,
  Workflow,
  BookOpen,
  FileText,
  Zap,
  Download,
  Upload,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

interface WorkflowPageProps {
  onNavigate: (page: "home" | "training" | "workflow" | "docs") => void
}

const workflowSteps = [
  {
    id: 1,
    title: "Model Selection",
    description: "Choose your base model from our curated collection",
    icon: Brain,
    status: "completed" as const,
    duration: "2 min",
  },
  {
    id: 2,
    title: "Data Preparation",
    description: "Upload and preprocess your training dataset",
    icon: Database,
    status: "completed" as const,
    duration: "5 min",
  },
  {
    id: 3,
    title: "Configuration",
    description: "Set hyperparameters and training options",
    icon: Settings,
    status: "active" as const,
    duration: "3 min",
  },
  {
    id: 4,
    title: "Fine-tuning",
    description: "Train your model with real-time monitoring",
    icon: Zap,
    status: "pending" as const,
    duration: "30-120 min",
  },
  {
    id: 5,
    title: "Evaluation",
    description: "Test and compare model performance",
    icon: BarChart3,
    status: "pending" as const,
    duration: "10 min",
  },
  {
    id: 6,
    title: "Deployment",
    description: "Deploy your fine-tuned model for inference",
    icon: MessageSquare,
    status: "pending" as const,
    duration: "5 min",
  },
]

const activeProjects = [
  {
    id: 1,
    name: "Customer Support Bot",
    model: "Llama 3.1 8B",
    progress: 75,
    status: "training" as const,
    eta: "45 min remaining",
  },
  {
    id: 2,
    name: "Code Assistant",
    model: "CodeLlama 34B",
    progress: 100,
    status: "completed" as const,
    eta: "Completed 2h ago",
  },
  {
    id: 3,
    name: "Content Generator",
    model: "Mistral 7B",
    progress: 25,
    status: "training" as const,
    eta: "2h 15min remaining",
  },
]

export default function WorkflowPage({ onNavigate }: WorkflowPageProps) {
  const [selectedStep, setSelectedStep] = useState(3)

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-5 w-5 text-primary" />
      case "active":
        return <Play className="h-5 w-5 text-secondary" />
      case "training":
        return <Zap className="h-5 w-5 text-chart-2 animate-pulse" />
      default:
        return <Clock className="h-5 w-5 text-muted-foreground" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-primary/10 text-primary border-primary/20"
      case "active":
        return "bg-secondary/10 text-secondary border-secondary/20"
      case "training":
        return "bg-chart-2/10 text-chart-2 border-chart-2/20"
      default:
        return "bg-muted text-muted-foreground border-border"
    }
  }

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
              <Button variant="default" className="text-sm">
                <Workflow className="mr-2 h-4 w-4" />
                Workflows
              </Button>
              <Button variant="ghost" onClick={() => onNavigate("docs")} className="text-sm">
                <BookOpen className="mr-2 h-4 w-4" />
                Documentation
              </Button>
            </nav>

            <Button onClick={() => onNavigate("training")}>
              Start Training
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto max-w-7xl p-4 md:p-6 lg:p-8 space-y-8">
        {/* Page Header */}
        <div className="space-y-4">
          <div className="flex items-center space-x-3">
            <div className="rounded-lg bg-primary/10 p-2">
              <Workflow className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Training Workflows</h1>
              <p className="text-muted-foreground">Visualize and manage your model fine-tuning pipeline</p>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Workflow Steps */}
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Fine-tuning Pipeline</CardTitle>
                <CardDescription>Follow these steps to successfully fine-tune your language model</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {workflowSteps.map((step, index) => {
                  const Icon = step.icon
                  const isActive = step.id === selectedStep
                  const isCompleted = step.status === "completed"
                  const isPending = step.status === "pending"

                  return (
                    <div key={step.id} className="relative">
                      {/* Connection Line */}
                      {index < workflowSteps.length - 1 && (
                        <div className="absolute left-6 top-12 w-0.5 h-16 bg-border" />
                      )}

                      <div
                        className={`flex items-start space-x-4 p-4 rounded-lg border cursor-pointer transition-all ${
                          isActive
                            ? "border-secondary bg-secondary/5"
                            : isCompleted
                              ? "border-primary/20 bg-primary/5"
                              : "border-border hover:border-border/60"
                        }`}
                        onClick={() => setSelectedStep(step.id)}
                      >
                        <div
                          className={`rounded-full p-2 ${
                            isCompleted
                              ? "bg-primary text-primary-foreground"
                              : isActive
                                ? "bg-secondary text-secondary-foreground"
                                : "bg-muted text-muted-foreground"
                          }`}
                        >
                          <Icon className="h-4 w-4" />
                        </div>

                        <div className="flex-1 space-y-2">
                          <div className="flex items-center justify-between">
                            <h3 className="font-semibold">{step.title}</h3>
                            <div className="flex items-center space-x-2">
                              <Badge variant="outline" className="text-xs">
                                {step.duration}
                              </Badge>
                              {getStatusIcon(step.status)}
                            </div>
                          </div>
                          <p className="text-sm text-muted-foreground">{step.description}</p>

                          {isActive && (
                            <div className="flex space-x-2 pt-2">
                              <Button size="sm">
                                Continue
                                <ArrowRight className="ml-2 h-3 w-3" />
                              </Button>
                              <Button size="sm" variant="outline">
                                <FileText className="mr-2 h-3 w-3" />
                                Guide
                              </Button>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
                <CardDescription>Common workflow operations and shortcuts</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-4">
                  <Button variant="outline" className="justify-start h-auto p-4 bg-transparent">
                    <Upload className="mr-3 h-5 w-5 text-primary" />
                    <div className="text-left">
                      <div className="font-medium">Upload Dataset</div>
                      <div className="text-sm text-muted-foreground">Add training data</div>
                    </div>
                  </Button>

                  <Button variant="outline" className="justify-start h-auto p-4 bg-transparent">
                    <Download className="mr-3 h-5 w-5 text-secondary" />
                    <div className="text-left">
                      <div className="font-medium">Export Model</div>
                      <div className="text-sm text-muted-foreground">Download trained model</div>
                    </div>
                  </Button>

                  <Button variant="outline" className="justify-start h-auto p-4 bg-transparent">
                    <BarChart3 className="mr-3 h-5 w-5 text-chart-1" />
                    <div className="text-left">
                      <div className="font-medium">View Metrics</div>
                      <div className="text-sm text-muted-foreground">Training analytics</div>
                    </div>
                  </Button>

                  <Button variant="outline" className="justify-start h-auto p-4 bg-transparent">
                    <MessageSquare className="mr-3 h-5 w-5 text-chart-3" />
                    <div className="text-left">
                      <div className="font-medium">Test Model</div>
                      <div className="text-sm text-muted-foreground">Interactive chat</div>
                    </div>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Active Projects Sidebar */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Active Projects</CardTitle>
                <CardDescription>Monitor your ongoing training sessions</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {activeProjects.map((project) => (
                  <div key={project.id} className="space-y-3 p-4 border rounded-lg">
                    <div className="flex items-start justify-between">
                      <div className="space-y-1">
                        <h4 className="font-medium text-sm">{project.name}</h4>
                        <p className="text-xs text-muted-foreground">{project.model}</p>
                      </div>
                      {getStatusIcon(project.status)}
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span>Progress</span>
                        <span>{project.progress}%</span>
                      </div>
                      <Progress value={project.progress} className="h-2" />
                      <p className="text-xs text-muted-foreground">{project.eta}</p>
                    </div>

                    <div className="flex space-x-2">
                      <Button size="sm" variant="outline" className="flex-1 text-xs bg-transparent">
                        View
                      </Button>
                      {project.status === "training" && (
                        <Button size="sm" variant="outline" className="text-xs bg-transparent">
                          <Pause className="h-3 w-3" />
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>System Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm">GPU Utilization</span>
                  <Badge className={getStatusColor("training")}>85%</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Queue Length</span>
                  <Badge variant="outline">2 jobs</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">API Status</span>
                  <Badge className={getStatusColor("completed")}>Online</Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  )
}
