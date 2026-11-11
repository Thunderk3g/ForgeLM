// Hardware tab: what the factory service sees on its own machine (GET
// /hardware), or any of the reference profiles the planner can plan for.
// Every planner answer depends on this profile, so it is shown in full:
// CPU features decide int8 serving, accelerators and their usability decide
// training, and the installed packages decide which candidates are runnable.

import { useState } from "react"
import { Cpu, HardDrive, MemoryStick, Package, Sparkles } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { factoryApi } from "@/lib/factory-api"
import { useFactoryQuery } from "@/hooks/useFactoryRequest"
import type { Accelerator, HardwareResponse } from "@/types/factory"
import { EmptyState, ErrorState, Field, LoadingState, Stat } from "./factory-shared"

// Radix Select cannot use "" as an item value, so the probed machine has a
// sentinel of its own.
const THIS_MACHINE = "__probe__"

const fetchProfile = (key: string) => factoryApi.hardware(key === THIS_MACHINE ? null : key)

function usability(a: Accelerator): { text: string; className: string } {
  if (a.torch_usable === true) return { text: "usable", className: "bg-chart-2/15 text-chart-2 border-chart-2/30" }
  if (a.torch_usable === false)
    return { text: "not usable by installed torch", className: "bg-destructive/15 text-destructive border-destructive/30" }
  return { text: "torch not checked", className: "text-muted-foreground" }
}

function ProfileView({ data }: { data: HardwareResponse }) {
  const p = data.profile
  const cap = data.generative_capacity
  const packages = Object.entries(p.packages).sort(([a], [b]) => a.localeCompare(b))
  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="h-5 w-5 text-primary" aria-hidden="true" />
            {p.cpu_model || "Unknown CPU"}
          </CardTitle>
          <CardDescription>
            {p.os} / {p.arch}, Python {p.python}
            {p.in_container && " (inside a container; memory reflects its limit)"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Device class" value={<Badge variant="secondary">{p.device_class}</Badge>} />
            <Stat label="Cores" value={`${p.physical_cores} physical / ${p.logical_cores} logical`} />
            <Stat label="Memory" value={`${p.ram_available_gb.toFixed(1)} of ${p.ram_total_gb.toFixed(1)} GiB free`} />
            <Stat label="int8 dot-product (VNNI)" value={p.has_vnni ? "yes" : "no"} />
          </dl>

          <section aria-labelledby="hw-flags" className="space-y-2">
            <h3 id="hw-flags" className="text-sm font-semibold">
              CPU features
            </h3>
            {p.cpu_flags.length ? (
              <ul className="flex flex-wrap gap-1.5">
                {p.cpu_flags.map((f) => (
                  <li key={f}>
                    <Badge variant="outline" className="font-mono">
                      {f}
                    </Badge>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted-foreground">None detected.</p>
            )}
          </section>

          <section aria-labelledby="hw-acc" className="space-y-2">
            <h3 id="hw-acc" className="flex items-center gap-2 text-sm font-semibold">
              <HardDrive className="h-4 w-4" aria-hidden="true" />
              Accelerators
            </h3>
            {p.accelerators.length ? (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Device</TableHead>
                      <TableHead>Vendor</TableHead>
                      <TableHead>Runtime</TableHead>
                      <TableHead>Memory</TableHead>
                      <TableHead>Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {p.accelerators.map((a, i) => {
                      const u = usability(a)
                      return (
                        <TableRow key={`${a.name}-${i}`}>
                          <TableCell className="whitespace-normal">
                            {a.count > 1 ? `${a.count} x ` : ""}
                            {a.name}
                            {a.notes.map((n) => (
                              <p key={n} className="text-xs text-muted-foreground">
                                {n}
                              </p>
                            ))}
                          </TableCell>
                          <TableCell>{a.vendor}</TableCell>
                          <TableCell className="font-mono">{a.runtime}</TableCell>
                          <TableCell>
                            {a.memory_gb.toFixed(1)} GiB{a.unified_memory ? " unified" : ""}
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline" className={u.className}>
                              {u.text}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      )
                    })}
                  </TableBody>
                </Table>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">None. Everything will run on the CPU.</p>
            )}
          </section>

          {p.notes.length > 0 && (
            <section aria-labelledby="hw-notes" className="space-y-2">
              <h3 id="hw-notes" className="text-sm font-semibold">
                Notes
              </h3>
              <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
                {p.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </section>
          )}
        </CardContent>
      </Card>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" aria-hidden="true" />
              Generative capacity
            </CardTitle>
            <CardDescription>The largest generative model this machine can fine-tune, by estimate.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-3xl font-bold">
              {cap.max_params_billion}B <span className="text-base font-normal text-muted-foreground">parameters</span>
            </p>
            <dl className="grid grid-cols-2 gap-4">
              <Stat label="Backend" value={<span className="font-mono">{cap.backend}</span>} />
              <Stat label="Device" value={<span className="font-mono">{cap.device}</span>} />
              <Stat label="Usable memory" value={`${cap.usable_memory_gb} GiB`} />
            </dl>
            <p className="text-sm text-muted-foreground">{cap.basis}</p>
            {cap.install_hint.length > 0 && (
              <p className="text-sm">
                Needs: <code className="rounded bg-muted px-1 py-0.5 text-xs">pip install {cap.install_hint.join(" ")}</code>
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Package className="h-5 w-5 text-primary" aria-hidden="true" />
              Installed backends
            </CardTitle>
            <CardDescription>Packages the planner checks for, and ONNX Runtime providers.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {packages.length ? (
              <ul className="grid grid-cols-1 gap-1 text-sm sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
                {packages.map(([name, version]) => (
                  <li key={name} className="flex justify-between gap-2 rounded border px-2 py-1">
                    <span className="font-mono">{name}</span>
                    <span className="text-muted-foreground">{version}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted-foreground">None of the optional backends is installed.</p>
            )}
            <div>
              <h3 className="mb-1 flex items-center gap-1.5 text-xs text-muted-foreground">
                <MemoryStick className="h-3.5 w-3.5" aria-hidden="true" />
                ONNX Runtime providers
              </h3>
              <p className="text-sm">{p.onnx_providers.length ? p.onnx_providers.join(", ") : "none"}</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export function HardwareTab({ referenceProfiles }: { referenceProfiles: string[] }) {
  const [view, setView] = useState(THIS_MACHINE)
  const { data, error, loading, reload } = useFactoryQuery(view, fetchProfile)
  // A response's own list is the freshest; the page-level list covers the
  // moment before the first response arrives.
  const profiles = data?.reference_profiles ?? referenceProfiles

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <Field
          id="hw-view"
          label="Profile"
          hint="The service's own machine, or a reference profile the planner can plan for."
          className="sm:max-w-sm"
        >
          <Select value={view} onValueChange={setView}>
            <SelectTrigger id="hw-view" className="w-full sm:w-72" aria-describedby="hw-view-hint">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={THIS_MACHINE}>This machine (probed)</SelectItem>
              {profiles.map((p) => (
                <SelectItem key={p} value={p}>
                  {p}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Button variant="outline" onClick={reload} disabled={loading}>
          {view === THIS_MACHINE ? "Probe again" : "Reload"}
        </Button>
      </div>

      {loading && !data && <LoadingState label="Reading the hardware profile..." />}
      {error !== null && <ErrorState error={error} onRetry={reload} />}
      {!loading && !error && !data && <EmptyState title="No profile loaded" />}
      {data && !error && (
        <div className={loading ? "opacity-60 transition-opacity" : undefined} aria-busy={loading}>
          <ProfileView data={data} />
        </div>
      )}
    </div>
  )
}
