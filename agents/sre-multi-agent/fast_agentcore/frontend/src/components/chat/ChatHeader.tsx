import { Button } from "@/components/ui/button"
import { Plus } from "lucide-react"
import { useAuth } from "@/hooks/useAuth"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { LogGeneratorButton } from "./LogGeneratorButton"
import { TestSystemButton } from "./TestSystemButton"

type ChatHeaderProps = {
  title?: string | undefined
  onNewChat: () => void
  canStartNewChat: boolean
  accessToken?: string
  onLogGeneratorSuccess?: (logGroup: string, logStream: string) => void
  onLogGeneratorError?: (error: string) => void
  onTestSystem?: () => void
}

export function ChatHeader({ 
  title, 
  onNewChat, 
  canStartNewChat,
  accessToken,
  onLogGeneratorSuccess,
  onLogGeneratorError,
  onTestSystem
}: ChatHeaderProps) {
  const { isAuthenticated, signOut } = useAuth()

  return (
    <header className="flex items-center justify-between p-4 border-b border-gray-700 w-full bg-gray-800">
      <div className="flex items-center">
        <h1 className="text-xl font-bold text-white">{title || "SRE Agent Powered by AgentCore and Meta Llama 3.3 70B"}</h1>
      </div>
      <div className="flex items-center gap-2">
        <LogGeneratorButton
          accessToken={accessToken}
          onSuccess={onLogGeneratorSuccess}
          onError={onLogGeneratorError}
        />
        {onTestSystem && (
          <TestSystemButton
            onTestSystem={onTestSystem}
            disabled={!canStartNewChat}
          />
        )}
        <Button onClick={onNewChat} variant="outline" className="gap-2" disabled={!canStartNewChat}>
          <Plus className="h-4 w-4" />
          New Chat
        </Button>
        {isAuthenticated && (
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline">Logout</Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Confirm Logout</AlertDialogTitle>
                <AlertDialogDescription>
                  Are you sure you want to log out? You will need to sign in again to access your
                  account.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={() => signOut()}>Confirm</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}
      </div>
    </header>
  )
}
