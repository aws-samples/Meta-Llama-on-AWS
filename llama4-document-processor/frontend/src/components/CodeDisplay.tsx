'use client'

import { useState } from 'react'
import { Copy, Download, Check } from 'lucide-react'
import { ProcessingResult } from '../types'

interface CodeDisplayProps {
  result: ProcessingResult
}

export default function CodeDisplay({ result }: CodeDisplayProps) {
  const [copied, setCopied] = useState(false)

  const copyToClipboard = async () => {
    try {
      // Try modern clipboard API first
      if (navigator.clipboard) {
        await navigator.clipboard.writeText(result.generated_code)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
        return
      }
      
      // Fallback for HTTP/non-secure contexts
      const textArea = document.createElement('textarea')
      textArea.value = result.generated_code
      textArea.style.position = 'fixed'
      textArea.style.left = '-999999px'
      textArea.style.top = '-999999px'
      document.body.appendChild(textArea)
      textArea.focus()
      textArea.select()
      
      const successful = document.execCommand('copy')
      document.body.removeChild(textArea)
      
      if (successful) {
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      } else {
        throw new Error('Copy command failed')
      }
    } catch (err) {
      console.error('Failed to copy:', err)
      // Show user they can manually select and copy
      alert('Copy failed. Please select the code manually and use Ctrl+C')
    }
  }

  const downloadCode = () => {
    const extension = result.language.toLowerCase() === 'python' ? 'py' : 
                     result.language.toLowerCase() === 'javascript' ? 'js' : 'txt'
    const filename = `generated_code.${extension}`
    
    const blob = new Blob([result.generated_code], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            {result.filename}
          </h3>
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <span>Language: {result.language}</span>
            <span>Processing time: {result.processing_time.toFixed(2)}s</span>
          </div>
        </div>
        
        <div className="flex space-x-2">
          <button
            onClick={copyToClipboard}
            className="flex items-center space-x-1 px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
          >
            {copied ? (
              <>
                <Check className="w-4 h-4 text-green-600" />
                <span className="text-green-600">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-4 h-4" />
                <span>Copy</span>
              </>
            )}
          </button>
          
          <button
            onClick={downloadCode}
            className="flex items-center space-x-1 px-3 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
          >
            <Download className="w-4 h-4" />
            <span>Download</span>
          </button>
        </div>
      </div>

      {/* Code Display */}
      <div className="flex-1 bg-gray-900 rounded-lg overflow-hidden">
        <div className="p-4 h-full overflow-auto">
          <pre className="text-sm text-gray-100 whitespace-pre-wrap font-mono leading-relaxed">
            <code>{result.generated_code}</code>
          </pre>
        </div>
      </div>

      {/* Success Indicator */}
      {result.success && (
        <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-green-700 text-sm">
            ✅ Code generated successfully! Ready to use.
          </p>
        </div>
      )}
    </div>
  )
}
