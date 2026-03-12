// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { FileText, Loader2 } from "lucide-react"

interface LogGeneratorButtonProps {
  accessToken: string | undefined
  onSuccess?: (logGroup: string, logStream: string) => void
  onError?: (error: string) => void
}

interface LogGeneratorRequest {
  hours?: number
  rate?: number
  seed?: number
}

interface LogGeneratorResponse {
  message: string
  log_group: string
  log_stream: string
  log_count: number
  time_range: {
    start: string
    end: string
  }
}

export function LogGeneratorButton({
  accessToken,
  onSuccess,
  onError,
}: LogGeneratorButtonProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [hours, setHours] = useState(1)
  const [rate, setRate] = useState(20)
  const [apiUrl, setApiUrl] = useState<string | null>(null)

  // Load API URL from config
  useEffect(() => {
    fetch("/aws-exports.json")
      .then((res) => res.json())
      .then((config) => {
        setApiUrl(config.logGeneratorApiUrl || null)
      })
      .catch((err) => {
        console.error("Failed to load config:", err)
      })
  }, [])

  const handleGenerate = async () => {
    if (!apiUrl) {
      onError?.("Log Generator API URL not configured")
      return
    }

    if (!accessToken) {
      onError?.("Not authenticated")
      return
    }

    setIsGenerating(true)

    try {
      const request: LogGeneratorRequest = {
        hours,
        rate,
        seed: Math.floor(Math.random() * 10000),
      }

      const response = await fetch(`${apiUrl}/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify(request),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || `HTTP ${response.status}`)
      }

      const data = await response.json()
      // Lambda returns body as a JSON string, need to parse it
      const result: LogGeneratorResponse = typeof data.body === 'string' 
        ? JSON.parse(data.body) 
        : data

      onSuccess?.(result.log_group, result.log_stream)
      setIsOpen(false)
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error)
      onError?.(errorMessage)
    } finally {
      setIsGenerating(false)
    }
  }

  if (!apiUrl) {
    return null // Don't show button if API not configured
  }

  const estimatedLogs = hours * 60 * rate * 5

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="gap-2">
          <FileText className="h-4 w-4" />
          Generate Logs
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Generate Bank Logs</DialogTitle>
          <DialogDescription>
            Generate synthetic banking system logs for incident analysis
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <label htmlFor="hours" className="text-right text-sm font-medium">
              Duration (hours)
            </label>
            <input
              id="hours"
              type="number"
              min="1"
              max="24"
              value={hours}
              onChange={(e) => setHours(parseInt(e.target.value) || 1)}
              className="col-span-3 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              disabled={isGenerating}
            />
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <label htmlFor="rate" className="text-right text-sm font-medium">
              Rate (per min)
            </label>
            <input
              id="rate"
              type="number"
              min="1"
              max="100"
              value={rate}
              onChange={(e) => setRate(parseInt(e.target.value) || 20)}
              className="col-span-3 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              disabled={isGenerating}
            />
          </div>
          <div className="rounded-lg bg-muted p-3 text-sm text-muted-foreground">
            This will generate approximately{" "}
            <span className="font-semibold">{estimatedLogs.toLocaleString()}</span> log
            events across 5 banking services (auth, payments, accounts, trading,
            notifications).
          </div>
        </div>
        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => setIsOpen(false)}
            disabled={isGenerating}
          >
            Cancel
          </Button>
          <Button type="button" onClick={handleGenerate} disabled={isGenerating}>
            {isGenerating && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {isGenerating ? "Generating..." : "Generate"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
