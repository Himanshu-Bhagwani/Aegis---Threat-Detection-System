import "./globals.css";
import { Inter } from "next/font/google";
import type { Metadata } from "next";
import type { ReactNode } from "react";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Aegis",
  description: "Aegis fraud and identity protection portal"
};

export default function RootLayout({
  children
}: {
  children: ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
