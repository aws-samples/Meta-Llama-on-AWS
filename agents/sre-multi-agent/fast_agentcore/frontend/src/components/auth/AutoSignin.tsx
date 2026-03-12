"use client"

import { ReactNode, useEffect, useState, PropsWithChildren } from "react"
import { useAuth } from "react-oidc-context"
import { Button } from "@/components/ui/button"

function AutoSigninContent({ children }: PropsWithChildren) {
  const auth = useAuth()
  const [signInError, setSignInError] = useState<string | null>(null)

  // Log auth state for debugging
  useEffect(() => {
    if (auth.error) {
      console.error("OIDC Auth error:", auth.error)
      setSignInError(auth.error.message)
    }
  }, [auth.error])

  const handleSignIn = async () => {
    setSignInError(null)
    try {
      await auth.signinRedirect()
    } catch (err: unknown) {
      console.error("signinRedirect failed:", err)
      const message = err instanceof Error ? err.message : String(err)
      setSignInError(message)

      // Fallback: build the authorize URL manually and redirect
      try {
        const response = await fetch("/aws-exports.json")
        const config = await response.json()
        const authority = config.authority
        const clientId = config.client_id
        const redirectUri = config.redirect_uri

        // Fetch OIDC discovery to get the authorization endpoint
        const discoveryResponse = await fetch(`${authority}/.well-known/openid-configuration`)
        const discovery = await discoveryResponse.json()
        const authEndpoint = discovery.authorization_endpoint

        const params = new URLSearchParams({
          client_id: clientId,
          response_type: "code",
          scope: "email openid profile",
          redirect_uri: redirectUri,
        })

        window.location.href = `${authEndpoint}?${params.toString()}`
      } catch (fallbackErr) {
        console.error("Fallback redirect also failed:", fallbackErr)
        setSignInError("Unable to redirect to sign-in page. Check browser console for details.")
      }
    }
  }

  if (auth.isLoading) {
    return <div className="flex items-center justify-center min-h-screen text-xl">Loading...</div>
  }

  if (!auth.isAuthenticated) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen gap-4">
        <p className="text-4xl">Please sign in</p>
        {signInError && (
          <p className="text-red-500 text-sm max-w-md text-center">{signInError}</p>
        )}
        <Button onClick={handleSignIn}>Sign In</Button>
      </div>
    )
  }

  return <>{children}</>
}

export function AutoSignin({ children }: { children: ReactNode }) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return null
  }

  return <AutoSigninContent>{children}</AutoSigninContent>
}
