"use client";

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";
import {
  Profile, AlertRecord,
  loadProfiles, saveProfiles, addProfile,
  updateProfileMetrics, generateAlerts,
} from "@/lib/profiles";

const SELECTED_KEY = "apeilo_selected_profile";

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

  const dismissAlert = useCallback((id: string) => {
    setDismissedIds(prev => {
      const next = new Set(prev);
      next.add(id);
      localStorage.setItem("apeilo_dismissed_alerts", JSON.stringify(Array.from(next)));
      return next;
    });
  }, []);

  const rawAlerts = generateAlerts(profiles);
  const alerts    = rawAlerts.filter(a => !dismissedIds.has(a.id));
  const selected  = profiles.find(p => p.id === selectedId);

  return (
    <ProfileCtx.Provider value={{
      profiles, selectedId, setSelectedId, selected,
      alerts, dismissedIds, dismissAlert,
      addNewProfile, updateMetrics,
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
