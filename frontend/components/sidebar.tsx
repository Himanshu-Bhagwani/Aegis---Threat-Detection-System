"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const sidebarLinks = [
  { href: "/dashboard", label: "Dashboard", description: "Command center overview" },
  { href: "/login", label: "Sign In", description: "Access existing guardians" },
  { href: "/signup", label: "Create Account", description: "Provision new identity nodes" }
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="app-sidebar">
      <section className="sidebar-section">
        <h2>Quick Access</h2>
        <nav aria-label="Secondary navigation">
          <ul className="sidebar-nav">
            {sidebarLinks.map((link) => {
              const isActive = pathname.startsWith(link.href);
              return (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className={`sidebar-link${isActive ? " is-active" : ""}`}
                  >
                    <span className="sidebar-link__title">{link.label}</span>
                    <span className="sidebar-link__meta">{link.description}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
      </section>

      <section className="sidebar-section sidebar-section--status">
        <h2>Operations</h2>
        <div className="sidebar-status">
          <div>
            <span className="sidebar-status__label">Threat posture</span>
            <span className="sidebar-status__value">Elevated</span>
          </div>
          <div>
            <span className="sidebar-status__label">Signals</span>
            <span className="sidebar-status__value">12.4k / hr</span>
          </div>
          <div>
            <span className="sidebar-status__label">Response SLA</span>
            <span className="sidebar-status__value">00:12:08</span>
          </div>
        </div>
      </section>
    </aside>
  );
}
