"use client";

import type { ReactNode } from "react";

import { Navigation } from "@/components/navigation";
import { Sidebar } from "@/components/sidebar";

type AppShellProps = {
  children: ReactNode;
};

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="app-shell">
      <Navigation />
      <div className="app-main">
        <Sidebar />
        <div className="app-content">{children}</div>
      </div>
    </div>
  );
}
