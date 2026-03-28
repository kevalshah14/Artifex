import { GoogleGenAI } from "@google/genai";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
} from "@assistant-ui/react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Thread } from "@/components/assistant-ui/thread";
import { MessageCircle, X } from "lucide-react";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });

const GeminiAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    const contents = messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({
        role: m.role === "assistant" ? "model" : "user",
        parts: m.content
          .filter((p) => p.type === "text")
          .map((p) => ({ text: p.type === "text" ? p.text : "" })),
      }));

    const response = await ai.models.generateContentStream({
      model: "gemini-2.5-flash",
      contents,
      config: { abortSignal },
    });

    let text = "";
    for await (const chunk of response) {
      text += chunk.text ?? "";
      yield { content: [{ type: "text" as const, text }] };
    }
  },
};

interface ChatPanelProps {
  isOpen: boolean;
  onClose: () => void;
  isDarkMode: boolean;
}

export function ChatPanel({ isOpen, onClose, isDarkMode }: ChatPanelProps) {
  const runtime = useLocalRuntime(GeminiAdapter);

  if (!isOpen) return null;

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <TooltipProvider>
        <div
          className={`absolute top-4 bottom-4 left-4 right-4 min-[660px]:right-auto min-[660px]:top-10 min-[660px]:left-10 min-[660px]:bottom-10 min-[660px]:w-[420px] rounded-[2.5rem] flex flex-col z-40 overflow-hidden shadow-2xl transition-all border ${
            isDarkMode
              ? "bg-slate-900/90 border-white/10 text-slate-100 shadow-slate-950/40"
              : "bg-white/90 border-white/60 text-slate-800 shadow-slate-200/40"
          }`}
          style={{ backdropFilter: "blur(20px) saturate(180%)" }}
        >
          {/* Header */}
          <div
            className={`px-7 py-5 border-b flex justify-between items-center shrink-0 ${
              isDarkMode
                ? "border-white/5 bg-white/5"
                : "border-slate-100 bg-white/40"
            }`}
          >
            <div className="flex items-center gap-3">
              <div
                className={`w-8 h-8 rounded-xl flex items-center justify-center ${
                  isDarkMode ? "bg-indigo-500/20" : "bg-indigo-50"
                }`}
              >
                <MessageCircle
                  className={`w-4 h-4 ${isDarkMode ? "text-indigo-400" : "text-indigo-600"}`}
                />
              </div>
              <div>
                <h2 className="text-sm font-bold leading-none">Chat</h2>
                <p
                  className={`text-[10px] mt-1 ${isDarkMode ? "text-slate-500" : "text-slate-400"}`}
                >
                  Gemini 2.5 Flash
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-200/20 rounded-full transition-colors text-slate-400"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Thread */}
          <div className={`flex-1 min-h-0 ${isDarkMode ? "dark" : ""}`}>
            <Thread />
          </div>
        </div>
      </TooltipProvider>
    </AssistantRuntimeProvider>
  );
}
