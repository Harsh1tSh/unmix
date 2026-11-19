/** Inline SVG icons (no icon font or library: nothing to load, nothing external). */
import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement> & { size?: number };

const base = (size: number) => ({ width: size, height: size, viewBox: "0 0 16 16", fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true });

export const PlayIcon = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} {...p} fill="currentColor" stroke="none">
    <path d="M4 2.3v11.4a.6.6 0 0 0 .9.5l9.2-5.7a.6.6 0 0 0 0-1L4.9 1.8a.6.6 0 0 0-.9.5Z" />
  </svg>
);
export const PauseIcon = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} {...p} fill="currentColor" stroke="none">
    <rect x="3" y="2" width="3.5" height="12" rx="1" />
    <rect x="9.5" y="2" width="3.5" height="12" rx="1" />
  </svg>
);
export const BackIcon = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M7 4 3 8l4 4M3 8h10" />
  </svg>
);
export const ForwardIcon = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="m9 4 4 4-4 4m4-4H3" />
  </svg>
);
export const LoopIcon = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M3 7V6a2 2 0 0 1 2-2h7m0 0-2-2m2 2-2 2M13 9v1a2 2 0 0 1-2 2H4m0 0 2 2m-2-2 2-2" />
  </svg>
);
export const VolumeIcon = ({ size = 14, muted = false, ...p }: P & { muted?: boolean }) => (
  <svg {...base(size)} {...p}>
    <path d="M2 6h2.5L8 3v10L4.5 10H2z" />
    {muted ? <path d="m11 6 4 4m0-4-4 4" /> : <path d="M11 5.5a3.5 3.5 0 0 1 0 5M13 3.5a6.5 6.5 0 0 1 0 9" />}
  </svg>
);
export const HeadphonesIcon = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M2.5 11V8a5.5 5.5 0 0 1 11 0v3" />
    <rect x="1.8" y="9.5" width="3" height="4.5" rx="1" />
    <rect x="11.2" y="9.5" width="3" height="4.5" rx="1" />
  </svg>
);
export const DownloadIcon = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M8 2v8m0 0L5 7m3 3 3-3M3 13h10" />
  </svg>
);
export const TrashIcon = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M2.5 4h11M6 4V2.5h4V4m-6 0 .7 9.2a1 1 0 0 0 1 .8h4.6a1 1 0 0 0 1-.8L12 4" />
  </svg>
);
export const PlusIcon = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M8 3v10M3 8h10" />
  </svg>
);
export const UserIcon = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <circle cx="8" cy="5.5" r="2.8" />
    <path d="M2.8 14a5.2 5.2 0 0 1 10.4 0" />
  </svg>
);
export const ChevronIcon = ({ size = 12, open = false, ...p }: P & { open?: boolean }) => (
  <svg {...base(size)} {...p} style={{ transform: open ? "rotate(180deg)" : undefined, transition: "transform 200ms" }}>
    <path d="m3 6 5 5 5-5" />
  </svg>
);
export const UploadIcon = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M8 11V3m0 0L5 6m3-3 3 3M3 13h10" />
  </svg>
);
export const ArrowUpRightIcon = ({ size = 14, ...p }: P) => (
  <svg {...base(size)} {...p}>
    <path d="M5 11 11 5M6 5h5v5" />
  </svg>
);
