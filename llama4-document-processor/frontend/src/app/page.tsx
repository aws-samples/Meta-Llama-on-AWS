'use client'

import { useState } from 'react'
import DocumentUpload from '../components/DocumentUpload'
import CodeDisplay from '../components/CodeDisplay'
import ProcessingStatus from '../components/ProcessingStatus'
import { ProcessingResult } from '../types'

export default function Home() {
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<ProcessingResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileUpload = async (file: File) => {
    setIsProcessing(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('http://YOUR-EC2-IP:8001/process-document', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: ProcessingResult = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Document to Code Generator
          </h1>
          <p className="text-xl text-gray-600 mb-2">
            Powered by Llama4 and LangGraph on AWS
          </p>
          <p className="text-gray-500">
            Upload API documentation and get production-ready code instantly
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-6">Upload Documentation</h2>
            <DocumentUpload 
              onFileUpload={handleFileUpload}
              isProcessing={isProcessing}
            />
            
            {isProcessing && (
              <div className="mt-6">
                <ProcessingStatus />
              </div>
            )}

            {error && (
              <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-700">Error: {error}</p>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-6">Generated Code</h2>
            {result ? (
              <CodeDisplay result={result} />
            ) : (
              <div className="text-center text-gray-500 py-12">
                <p>Upload a document to see generated code here</p>
              </div>
            )}
          </div>
        </div>

        {/* Workflow Steps */}
        {result && result.workflow_steps && (
          <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">Processing Workflow</h3>
            <div className="space-y-3">
              {result.workflow_steps.map((step, index) => (
                <div key={index} className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    step.status === 'completed' ? 'bg-green-500' : 
                    step.status === 'failed' ? 'bg-red-500' : 'bg-yellow-500'
                  }`} />
                  <span className="font-medium">{step.step_name}</span>
                  <span className="text-gray-500">({step.duration.toFixed(2)}s)</span>
                  {step.output && (
                    <span className="text-sm text-gray-600">- {step.output}</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  )
}