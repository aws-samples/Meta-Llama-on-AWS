'use client'

import { useEffect, useState } from 'react'
import { Loader2, FileText, Brain, Code, CheckCircle } from 'lucide-react'

const steps = [
  { id: 'parse', label: 'Parsing Document', icon: FileText },
  { id: 'analyze', label: 'Analyzing Content', icon: Brain },
  { id: 'generate', label: 'Generating Code', icon: Code },
  { id: 'validate', label: 'Validating Output', icon: CheckCircle },
]

export default function ProcessingStatus() {
  const [currentStep, setCurrentStep] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev + 1) % steps.length)
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
      <div className="flex items-center space-x-3 mb-4">
        <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
        <h3 className="text-lg font-semibold text-blue-900">
          Processing with LangGraph & Llama4
        </h3>
      </div>

      <div className="space-y-3">
        {steps.map((step, index) => {
          const Icon = step.icon
          const isActive = index === currentStep
          const isCompleted = index < currentStep

          return (
            <div
              key={step.id}
              className={`flex items-center space-x-3 p-3 rounded-lg transition-colors ${
                isActive ? 'bg-blue-100 border border-blue-300' : 
                isCompleted ? 'bg-green-50 border border-green-200' : 
                'bg-gray-50 border border-gray-200'
              }`}
            >
              <div className={`flex-shrink-0 ${
                isActive ? 'text-blue-600' : 
                isCompleted ? 'text-green-600' : 
                'text-gray-400'
              }`}>
                {isActive ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Icon className="w-5 h-5" />
                )}
              </div>
              
              <span className={`font-medium ${
                isActive ? 'text-blue-900' : 
                isCompleted ? 'text-green-900' : 
                'text-gray-600'
              }`}>
                {step.label}
              </span>

              {isActive && (
                <div className="flex-1 flex justify-end">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="mt-4 text-sm text-blue-700">
        <p>🚀 Powered by AWS SageMaker & Llama4</p>
      </div>
    </div>
  )
}