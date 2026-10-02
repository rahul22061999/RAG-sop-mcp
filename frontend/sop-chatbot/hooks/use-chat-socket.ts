"use client"

import { useCallback, useEffect, useRef } from "react";

const WS_URL =  "ws://localhost:8000/ws/stream";


type ServerMessage =
  | { type: "token"; content: string }
  | { type: "done" }
  | { type: "error"; content: string };


export type ChatHandlers = {
  onToken: (token: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
};

export function useChatSocket() {
  const wsRef = useRef<WebSocket | null>(null);

  const cancel = useCallback(() => {
    const ws = wsRef.current;
    wsRef.current = null; // detach first so onclose doesn't report an error
    ws?.close();
  }, []);

  // Close any open socket when the component unmounts / the page is left.
  useEffect(() => cancel, [cancel]);

  const ask = useCallback(
    (query: string, handlers: ChatHandlers) => {
      cancel();
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      let finished = false;
      const finish = (fn: () => void) => {
        if (finished) return;
        finished = true;
        if (wsRef.current === ws) wsRef.current = null;
        fn();
        ws.close();
      };

      ws.onopen = () => ws.send(JSON.stringify({ query }));

      ws.onmessage = (event) => {
        const msg: ServerMessage = JSON.parse(event.data);
        if (msg.type === "token") handlers.onToken(msg.content);
        else if (msg.type === "done") finish(handlers.onDone);
        else finish(() => handlers.onError(msg.content));
      };

      ws.onerror = () =>
        finish(() => handlers.onError("Could not reach the SOP server."));

      ws.onclose = () => {
        // Ignore closes we caused (cancel / finish); report unexpected ones.
        if (wsRef.current === ws) {
          finish(() => handlers.onError("Connection closed unexpectedly."));
        }
      };
    },
    [cancel],
  );

  return { ask, cancel };
}