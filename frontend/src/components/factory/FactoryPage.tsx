// The Factory section: the dashboard's face of the factory service
// (hardware-aware planning, LLM-free classifiers and call screening). It
// follows the other pages' layout: a sticky header with the section
// navigation, a page title, then the work area, here split into four tabs.

import { BarChart3, BookOpen, Brain, Factory, Home, Settings, Workflow } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { factoryApi, describeError } from "@/lib/factory-api"
import { useFactoryQuery } from "@/hooks/useFactoryRequest"
import { CallReviewTab } from "./CallReviewTab"
import { ClassifiersTab } from "./ClassifiersTab"
import { HardwareTab } from "./HardwareTab"
import { PlanTab } from "./PlanTab"

type Page = "home" | "training" | "workflow" | "docs" | "analytics" | "factory"

interface FactoryPageProps {
  onNavigate: (page: Page) => void
}

const NAV: { page: Page; label: string; icon: typeof Home }[] = [
  { page: "home", label: "Home", icon: Home },
  { page: "training", label: "Fine-tuning", icon: Settings },
  { page: "workflow", label: "Workflows", icon: Workflow },
  { page: "factory", label: "Factory", icon: Factory },
  { page: "analytics", label: "Analytics", icon: BarChart3 },
  { page: "docs", label: "Documentation", icon: BookOpen },
]

const probeThisMachine = () => factoryApi.hardware(null)

function ServiceStatus({ loading, error }: { loading: boolean; error: unknown }) {
  // Derived from the hardware probe the page makes anyway: if that answered,
  // the service is up.
  const state = loading ? "checking" : error ? "offline" : "online"
  const text = loading ? "Checking..." : error ? `Factory ${describeError(error).title}` : "Factory online"
  return (
    <div className="flex items-center space-x-2 rounded-md border px-3 py-1.5" role="status">
      <div
        className={`h-2 w-2 rounded-full ${
          state === "online" ? "bg-chart-2" : state === "offline" ? "bg-destructive" : "bg-secondary"
        }`}
        aria-hidden="true"
      />
      <span className="text-sm font-medium">{text}</span>
    </div>
  )
}

export default function FactoryPage({ onNavigate }: FactoryPageProps) {
  const probe = useFactoryQuery("this-machine", probeThisMachine)
  const referenceProfiles = probe.data?.reference_profiles ?? []
  // Shared by the Classifiers and Call review tabs, so a classifier trained
  // in one is offered in the other without a reload.
  const classifiers = useFactoryQuery("all", factoryApi.listClassifiers)

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto max-w-7xl">
          <div className="flex h-16 items-center justify-between gap-2 px-4">
            <div className="flex items-center space-x-3">
              <Brain className="h-8 w-8 text-primary" aria-hidden="true" />
              <span className="text-2xl font-bold tracking-tight">LLMForge</span>
            </div>

            <nav className="hidden lg:flex items-center space-x-1" aria-label="Sections">
              {NAV.map(({ page, label, icon: Icon }) => (
                <Button
                  key={page}
                  variant={page === "factory" ? "default" : "ghost"}
                  onClick={page === "factory" ? undefined : () => onNavigate(page)}
                  aria-current={page === "factory" ? "page" : undefined}
                  className="text-sm"
                >
                  <Icon className="mr-2 h-4 w-4" aria-hidden="true" />
                  {label}
                </Button>
              ))}
            </nav>

            <div className="flex items-center gap-2">
              <ServiceStatus loading={probe.loading} error={probe.error} />
              {/* Below lg the section links are hidden, as on the other pages; keep a way home. */}
              <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => onNavigate("home")} aria-label="Home">
                <Home className="h-4 w-4" aria-hidden="true" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto max-w-7xl space-y-8 p-4 md:p-6 lg:p-8">
        <div className="flex items-center space-x-3">
          <div className="rounded-lg bg-primary/10 p-2">
            <Factory className="h-6 w-6 text-primary" aria-hidden="true" />
          </div>
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Model Factory</h1>
            <p className="text-muted-foreground">
              Plan from your hardware and data, train LLM-free classifiers, and screen calls for review.
            </p>
          </div>
          <Badge variant="outline" className="ml-auto hidden sm:inline-flex">
            CPU only, no LLM at inference
          </Badge>
        </div>

        {/* Every tab stays mounted (hidden when inactive) so a plan, a training
            result or a review survives a look at another tab. */}
        <Tabs defaultValue="hardware" className="gap-6">
          <TabsList className="grid h-auto w-full grid-cols-2 sm:grid-cols-4 lg:w-fit" aria-label="Factory tools">
            <TabsTrigger value="hardware" className="py-1.5">
              Hardware
            </TabsTrigger>
            <TabsTrigger value="plan" className="py-1.5">
              Plan
            </TabsTrigger>
            <TabsTrigger value="classifiers" className="py-1.5">
              Classifiers
            </TabsTrigger>
            <TabsTrigger value="review" className="py-1.5">
              Call review
            </TabsTrigger>
          </TabsList>
          <TabsContent value="hardware" forceMount className="data-[state=inactive]:hidden">
            <HardwareTab probe={probe} />
          </TabsContent>
          <TabsContent value="plan" forceMount className="data-[state=inactive]:hidden">
            <PlanTab referenceProfiles={referenceProfiles} />
          </TabsContent>
          <TabsContent value="classifiers" forceMount className="data-[state=inactive]:hidden">
            <ClassifiersTab list={classifiers} />
          </TabsContent>
          <TabsContent value="review" forceMount className="data-[state=inactive]:hidden">
            <CallReviewTab classifiers={classifiers} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
