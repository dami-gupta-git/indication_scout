// Renders an agent/supervisor summary string as markdown. Summaries are
// multi-paragraph prose with **bold** section headers and bullet lists, so a
// raw <p> shows literal asterisks and collapses everything onto one line.

import ReactMarkdown from "react-markdown";

export function Markdown({ children }: { children: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown>{children}</ReactMarkdown>
    </div>
  );
}
