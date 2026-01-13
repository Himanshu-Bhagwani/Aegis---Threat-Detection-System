"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { isAuthenticated, removeAuthToken } from "@/lib/auth";

const primaryLinks = [
  { href: "/", label: "Overview" },
  { href: "/#features", label: "Features" },
  { href: "/#how-it-works", label: "How It Works" },
  { href: "/#why-aegis", label: "Why Aegis" },
  { href: "/dashboard", label: "Dashboard" }
];

export function Navigation() {
  const pathname = usePathname();
  const router = useRouter();
  const [authenticated, setAuthenticated] = useState(false);

  useEffect(() => {
    // Check auth status on mount and when pathname changes
    setAuthenticated(isAuthenticated());
  }, [pathname]);

  const handleLogout = () => {
    removeAuthToken();
    setAuthenticated(false);
    router.push("/login");
  };

  return (
    <header className="app-nav">
      <div className="app-nav__brand">
        <Link href="/" className="app-nav__logo">
          <span className="app-nav__logo-mark" />
          <span className="app-nav__logo-text">Aegis</span>
        </Link>
        <span className="app-nav__label">Fraud &amp; Identity Protection</span>
      </div>

      <nav className="app-nav__links" aria-label="Primary navigation">
        {primaryLinks.map((link) => {
          const basePath = link.href.split("#")[0] || "/";
          const isHashLink = link.href.includes("#");
          const isActive =
            basePath === "/"
              ? pathname === "/"
              : !isHashLink && pathname.startsWith(basePath);
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`app-nav__link${isActive ? " is-active" : ""}`}
            >
              {link.label}
            </Link>
          );
        })}
      </nav>

      <div className="app-nav__actions">
        {authenticated ? (
          <>
            <button 
              onClick={handleLogout}
              className="app-nav__action ghost"
              style={{ cursor: "pointer", background: "none", border: "none", fontFamily: "inherit" }}
            >
              Log out
            </button>
          </>
        ) : (
          <>
            <Link href="/login" className="app-nav__action ghost">
              Log in
            </Link>
            <Link href="/signup" className="app-nav__action primary">
              Get Started
            </Link>
          </>
        )}
      </div>
    </header>
  );
}
