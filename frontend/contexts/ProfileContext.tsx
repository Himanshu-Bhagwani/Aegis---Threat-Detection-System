"use client";

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";
import {
  Profile, AlertRecord,
  loadProfiles, saveProfiles, addProfile,
  updateProfileMetrics, generateAlerts,
  removeProfile, hideDemoProfile,
} from "@/lib/profiles";
import {
  wsClient, WSEvent, dismissAlert as dismissAlertApi,
  getProfiles, RemoteProfile, deleteProfileRemote,
} from "@/lib/api";

const SELECTED_KEY = "apeilo_selected_profile";

// Map a backend (live) profile into the local Profile shape.
function remoteToProfile(rp: RemoteProfile): Profile {
  return {
    id:         rp.id,
    name:       rp.name || rp.id,
    email:      rp.email || "",
    created_at: rp.created_at || new Date().toISOString(),
    is_demo:    false,
    notes:      rp.notes || "Live profile",
    password_meta: rp.password_meta ?? null,
    metrics: {
      gps_spoof:     rp.metrics.gps_spoof,
      login_anomaly: rp.metrics.login_anomaly,
      password_leak: rp.metrics.password_leak,
      fraud_risk:    rp.metrics.fraud_risk,
      breach_risk:   rp.metrics.breach_risk,
      unified_score: rp.metrics.unified_score,
      risk_level:    rp.metrics.risk_level,
      last_updated:  rp.metrics.last_updated,
    },
  };
}

interface ProfileContextValue {
  profiles:       Profile[];
  selectedId:     string;
  setSelectedId:  (id: string) => void;
  selected:       Profile | undefined;
  alerts:         AlertRecord[];
  dismissedIds:   Set<string>;
  dismissAlert:   (id: string) => void;
  addNewProfile:  (name: string, email: string) => void;
  updateMetrics:  (id: string, partial: Parameters<typeof updateProfileMetrics>[2]) => void;
  deleteProfile:  (id: string) => Promise<void>;
}

const ProfileCtx = createContext<ProfileContextValue | null>(null);

export function ProfileProvider({ children }: { children: ReactNode }) {
  const [profiles,     setProfiles]     = useState<Profile[]>([]);
  const [selectedId,   setSelectedIdRaw] = useState<string>("himanshu");
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());
  const [hydrated,     setHydrated]     = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    setProfiles(loadProfiles());
    const saved = localStorage.getItem(SELECTED_KEY);
    if (saved) setSelectedIdRaw(saved);
    try {
      const dis = localStorage.getItem("apeilo_dismissed_alerts");
      if (dis) setDismissedIds(new Set(JSON.parse(dis)));
    } catch {}
    setHydrated(true);
  }, []);

  // Merge in real, event-driven profiles from the backend (the live SODA users)
  // and keep them fresh by polling. WebSocket events then live-update them too.
  const refreshRemote = useCallback(async () => {
    try {
      const { profiles: remote } = await getProfiles();
      if (!Array.isArray(remote) || remote.length === 0) return;
      const real = remote.map(remoteToProfile);
      setProfiles(prev => {
        const byId = new Map<string, Profile>();
        for (const p of prev) byId.set(p.id, p);   // demo + locally-added first
        for (const r of real) byId.set(r.id, r);    // live backend data wins
        return Array.from(byId.values());
      });
    } catch {
      /* backend unreachable — keep showing whatever we have */
    }
  }, []);

  useEffect(() => {
    refreshRemote();
    const t = setInterval(refreshRemote, 8000);
    return () => clearInterval(t);
  }, [refreshRemote]);

  // Auto-update profile metrics from real-time unified_risk WebSocket events.
  // When the backend fires a unified_risk event with a user_id that matches a
  // profile id, all module scores are refreshed from the event details.
  useEffect(() => {
    const unsub = wsClient.subscribe((evt: WSEvent) => {
      if (evt.type !== "detection_event" || evt.event_type !== "unified_risk") return;
      if (!evt.user_id || !evt.details) return;
      const d = evt.details as Record<string, number>;
      setProfiles(prev => {
        if (!prev.find(p => p.id === evt.user_id)) return prev;
        const partial: Record<string, number> = {};
        if (d.gps_risk      != null) partial.gps_spoof     = d.gps_risk;
        if (d.login_risk    != null) partial.login_anomaly  = d.login_risk;
        if (d.password_risk != null) partial.password_leak  = d.password_risk;
        if (d.fraud_risk    != null) partial.fraud_risk     = d.fraud_risk;
        if (d.breach_risk   != null) partial.breach_risk    = d.breach_risk;
        // Accept the backend's weighted unified_score directly instead of recalculating locally
        if (d.unified_score != null) partial.unified_score  = d.unified_score;
        return updateProfileMetrics(prev, evt.user_id!, partial as any);
      });
    });
    return unsub;
  }, []);

  // Persist selectedId whenever it changes
  const setSelectedId = useCallback((id: string) => {
    setSelectedIdRaw(id);
    localStorage.setItem(SELECTED_KEY, id);
  }, []);

  const addNewProfile = useCallback((name: string, email: string) => {
    setProfiles(prev => addProfile(prev, name, email));
  }, []);

  const updateMetrics = useCallback((id: string, partial: Parameters<typeof updateProfileMetrics>[2]) => {
    setProfiles(prev => updateProfileMetrics(prev, id, partial));
  }, []);

  const deleteProfile = useCallback(async (id: string) => {
    const target = profiles.find(p => p.id === id);

    // Demo profiles live only in the browser — remember the deletion so they
    // don't reappear from the seed list on reload. Live profiles have a
    // backend row: remove it first so the 8s poll can't resurrect them.
    if (target?.is_demo) {
      hideDemoProfile(id);
    } else {
      try { await deleteProfileRemote(id); } catch { /* already gone / offline */ }
    }

    const next = removeProfile(profiles, id);
    setProfiles(next);

    // Keep a valid selection after removing the selected profile.
    if (id === selectedId) {
      const fallback = next[0]?.id ?? "";
      setSelectedIdRaw(fallback);
      try { localStorage.setItem(SELECTED_KEY, fallback); } catch {}
    }
  }, [profiles, selectedId]);

  const dismissAlert = useCallback((id: string) => {
    // Optimistic local update
    setDismissedIds(prev => {
      const next = new Set(prev);
      next.add(id);
      localStorage.setItem("apeilo_dismissed_alerts", JSON.stringify(Array.from(next)));
      return next;
    });
    // Persist to backend (best-effort; local state remains source of truth for demo profiles)
    dismissAlertApi(id).catch(() => {});
  }, []);

  const rawAlerts = generateAlerts(profiles);
  const alerts    = rawAlerts.filter(a => !dismissedIds.has(a.id));
  const selected  = profiles.find(p => p.id === selectedId);

  return (
    <ProfileCtx.Provider value={{
      profiles, selectedId, setSelectedId, selected,
      alerts, dismissedIds, dismissAlert,
      addNewProfile, updateMetrics, deleteProfile,
    }}>
      {children}
    </ProfileCtx.Provider>
  );
}

export function useProfiles() {
  const ctx = useContext(ProfileCtx);
  if (!ctx) throw new Error("useProfiles must be used within ProfileProvider");
  return ctx;
}
