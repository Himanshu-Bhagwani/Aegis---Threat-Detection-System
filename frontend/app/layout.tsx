import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Apeilo — Threat Detection",
  description: "Multi-layer threat detection and identity management",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
