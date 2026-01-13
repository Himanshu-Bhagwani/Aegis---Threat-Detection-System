/**
 * Authentication utilities for checking login status and managing tokens
 */

const TOKEN_KEY = "aegis.idToken";

/**
 * Get the stored authentication token
 */
export function getAuthToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
  const token = getAuthToken();
  return token !== null && token.length > 0;
}

/**
 * Store authentication token
 */
export function setAuthToken(token: string): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(TOKEN_KEY, token);
}

/**
 * Remove authentication token (logout)
 */
export function removeAuthToken(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(TOKEN_KEY);
  // Clean up old typo key if it exists
  localStorage.removeItem("ageis.idToken");
}

/**
 * Get Authorization header value for API requests
 */
export function getAuthHeader(): string | null {
  const token = getAuthToken();
  return token ? `Bearer ${token}` : null;
}

