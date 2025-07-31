#!/bin/bash

# Frontend startup script

set -e

echo "🚀 Starting Llama4 Frontend..."

# Install dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install additional required packages
echo "📦 Installing additional packages..."
npm install lucide-react
npm install -D tailwindcss postcss autoprefixer

echo "✅ Starting Next.js development server..."
echo "🌐 Frontend will be available at: http://localhost:3000"
echo "🛑 Press Ctrl+C to stop the server"

npm run dev -- --hostname 0.0.0.0