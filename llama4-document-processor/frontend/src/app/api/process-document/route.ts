import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const response = await fetch('http://localhost:8001/process-document', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: 'Processing failed' }, { status: 500 });
  }
}