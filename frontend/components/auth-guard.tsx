"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { isAuthenticated } from "@/lib/auth";
import type { ReactNode } from "react";

type AuthGuardProps = {
  children: ReactNode;
};

/**
 * AuthGuard component - Protects routes that require authentication
 * Redirects to login if user is not authenticated
 */
export function AuthGuard({ 
  children 
}: AuthGuardProps) {
  const router = useRouter();
  const [isAuth, setIsAuth] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check authentication status
    const checkAuth = () => {
      const authenticated = isAuthenticated();
      setIsAuth(authenticated);
      setIsLoading(false);

      if (!authenticated) {
        // Redirect to login if not authenticated
        router.push("/login");
      }
    };

    checkAuth();
  }, [router]);

  // Show loading state while checking
  if (isLoading) {
    return (
      <div style={{ 
        display: "flex", 
        justifyContent: "center", 
        alignItems: "center", 
        height: "100vh" 
      }}>
        <div>Checking authentication...</div>
      </div>
    );
  }

  // Only render children if authenticated
  if (!isAuth) {
    return null; // Will redirect in useEffect
  }

  return <>{children}</>;
}

