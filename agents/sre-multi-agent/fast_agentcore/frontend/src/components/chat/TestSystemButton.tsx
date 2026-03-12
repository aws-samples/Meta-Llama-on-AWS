// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { PlayCircle } from "lucide-react"

interface TestSystemButtonProps {
  onTestSystem: () => void
  disabled?: boolean
}

export function TestSystemButton({
  onTestSystem,
  disabled = false,
}: TestSystemButtonProps) {
  const [isRunning, setIsRunning] = useState(false)

  const handleClick = () => {
    setIsRunning(true)
    onTestSystem()
    // Reset after a delay to allow the test to start
    setTimeout(() => setIsRunning(false), 2000)
  }

  return (
    <Button
      onClick={handleClick}
      variant="outline"
      size="sm"
      className="gap-2"
      disabled={disabled || isRunning}
    >
      <PlayCircle className="h-4 w-4" />
      Test System
    </Button>
  )
}
