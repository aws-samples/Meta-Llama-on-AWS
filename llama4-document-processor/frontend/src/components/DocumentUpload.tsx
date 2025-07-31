'use client'

import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, Loader2 } from 'lucide-react'

interface DocumentUploadProps {
  onFileUpload: (file: File) => void
  isProcessing: boolean
}

export default function DocumentUpload({ onFileUpload, isProcessing }: DocumentUploadProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileUpload(acceptedFiles[0])
    }
  }, [onFileUpload])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md']
    },
    multiple: false,
    disabled: isProcessing
  })

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
          ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center space-y-4">
          {isProcessing ? (
            <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />
          ) : (
            <Upload className="w-12 h-12 text-gray-400" />
          )}
          
          <div>
            <p className="text-lg font-medium text-gray-900">
              {isProcessing ? 'Processing document...' : 
               isDragActive ? 'Drop your document here' : 
               'Upload API Documentation'}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              {isProcessing ? 'Please wait while Llama4 generates your code' :
               'Drag & drop or click to select PDF, DOCX, TXT, or MD files'}
            </p>
          </div>
        </div>
      </div>

      <div className="mt-4 text-xs text-gray-500">
        <p className="font-medium mb-2">Supported formats:</p>
        <div className="flex flex-wrap gap-2">
          {[
            { ext: 'PDF', desc: 'API documentation' },
            { ext: 'DOCX', desc: 'Word documents' },
            { ext: 'TXT', desc: 'Plain text' },
            { ext: 'MD', desc: 'Markdown files' }
          ].map(({ ext, desc }) => (
            <div key={ext} className="flex items-center space-x-1 bg-gray-100 px-2 py-1 rounded">
              <FileText className="w-3 h-3" />
              <span className="font-mono">{ext}</span>
              <span>- {desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}