"use client";

import type { ThreadMessageLike } from "@assistant-ui/react";
import type { AppendMessage } from "@assistant-ui/react";
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
} from "@assistant-ui/react";
import { createContext, useContext, useState } from "react";
import { useChatSocket } from "@/hooks/use-chat-socket";

const convertMessage = (message: ThreadMessageLike) => {
  return message;
};

type ChatControls = {
  isRunning: boolean;
  disconnect: () => void;
  clearChat: () => void;
};

const ChatControlsContext = createContext<ChatControls | null>(null);

export function useChatControls(): ChatControls {
  const controls = useContext(ChatControlsContext);
  if (!controls) {
    throw new Error("useChatControls must be used inside MyRuntimeProvider");
  }
  return controls;
}

export function MyRuntimeProvider({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const [messages, setMessages] = useState<readonly ThreadMessageLike[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const { ask, cancel } = useChatSocket();

  const disconnect = () => {
    cancel();
    setIsRunning(false);
  };

  const clearChat = () => {
    disconnect();
    setMessages([]);
  };

  const onNew = async (message: AppendMessage) => {
    if (message.content.length !== 1 || message.content[0]?.type !== "text")
      throw new Error("Only text content is supported");

    const query = message.content[0].text;
    const assistantId = crypto.randomUUID();

    setMessages((cur) => [
      ...cur,
      { role: "user", content: [{ type: "text", text: query }] },
      { id: assistantId, role: "assistant", content: [{ type: "text", text: "" }] },
    ]);
    setIsRunning(true);

    let answer = "";
    const setAnswer = (text: string) =>
      setMessages((cur) =>
        cur.map((m) =>
          m.id === assistantId
            ? { ...m, content: [{ type: "text", text }] }
            : m,
        ),
      );
    
    ask(query, {
      onToken: (token) => {
        answer += token;
        setAnswer(answer);
      },
      onDone: () => setIsRunning(false),
      onError: (err) => {
        setAnswer(answer ? `${answer}\n\n_${err}_` : `⚠️ ${err}`);
        setIsRunning(false);
      },
    });
  };

  const runtime = useExternalStoreRuntime<ThreadMessageLike>({
    messages,
    setMessages,
    isRunning,
    onNew,
    onCancel: async () => disconnect(),
    convertMessage,
  });

  return (
    <ChatControlsContext.Provider value={{ isRunning, disconnect, clearChat }}>
      <AssistantRuntimeProvider runtime={runtime}>
        {children}
      </AssistantRuntimeProvider>
    </ChatControlsContext.Provider>
  );
}