"use client";

import { useChatControls } from "@/app/MyRuntimeProvider";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { Button } from "@/components/ui/button";
import { PlugZapIcon, Trash2Icon } from "lucide-react";
import {
  useAui,
  AuiProvider,
  AuiConfig,
  Suggestions,
} from "@assistant-ui/react";

function ThreadWithSuggestions() {
  const aui = useAui();
  const config = AuiConfig({
    suggestions: Suggestions([
      {
        title: "Receiving goods",
        label: "from a supplier",
        prompt: "How do I receive goods from a supplier?",
      },
      {
        title: "Expired stock",
        label: "removal and reporting",
        prompt: "What is the process for handling expired stock?",
      },
      {
        title: "Supplementary orders",
        label: "when and how to raise them",
        prompt: "When and how do I create a supplementary order?",
      },
      {
        title: "Warehouse security",
        label: "access and visitor rules",
        prompt: "What are the security procedures for the warehouse?",
      },
    ]),
  });
  return (
    <AuiProvider extends={aui} config={config}>
      <Thread />
    </AuiProvider>
  );
}

function ChatToolbar() {
  const { isRunning, disconnect, clearChat } = useChatControls();
  return (
    <header className="flex items-center justify-between border-b px-4 py-2">
      <span className="text-sm font-medium">WMS SOP Assistant</span>
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={disconnect}
          disabled={!isRunning}
          title="Stop the current answer and close the connection"
        >
          <PlugZapIcon /> Disconnect
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={clearChat}
          title="Delete this conversation"
        >
          <Trash2Icon /> Clear chat
        </Button>
      </div>
    </header>
  );
}

export default function Home() {
  return (
    <main className="flex h-dvh flex-col">
      <ChatToolbar />
      <div className="min-h-0 flex-1">
        <ThreadWithSuggestions />
      </div>
    </main>
  );
}
