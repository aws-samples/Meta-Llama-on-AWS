export interface WorkflowStep {
  step_name: string
  status: 'completed' | 'failed' | 'completed_with_warnings'
  duration: number
  output?: string
}

export interface ProcessingResult {
  success: boolean
  filename: string
  generated_code: string
  language: string
  processing_time: number
  workflow_steps: WorkflowStep[]
  timestamp: string
}