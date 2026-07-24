import type { ReactNode } from "react";
import Sidebar from "@/components/sidebar";
import Topbar  from "@/components/topbar";
import ChallengeCenter from "@/components/ChallengeCenter";
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
          {/* "Was this you?" prompts are answered here, not in the tracked app */}
          <ChallengeCenter />
        </div>
      </div>
    </ProfileProvider>
  );
}
