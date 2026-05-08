import type { ReactNode } from "react";
import Sidebar from "@/components/sidebar";
import Topbar  from "@/components/topbar";
import { ProfileProvider } from "@/contexts/ProfileContext";

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <ProfileProvider>
      <div className="app-layout">
        <Sidebar />
        <div style={{
          display: "flex", flexDirection: "column",
          minWidth: 0, gridRow: "1 / -1", overflow: "hidden",
          height: "100vh",
        }}>
          <Topbar />
          <main className="app-main" style={{ flex: 1, overflowY: "auto" }}>
            {children}
          </main>
        </div>
      </div>
    </ProfileProvider>
  );
}
