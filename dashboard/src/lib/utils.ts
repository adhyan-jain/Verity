import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merges conditional class names (clsx) and resolves conflicting Tailwind
 * utility classes (tailwind-merge). Standard shadcn/ui helper — every
 * component under `components/ui/` imports this as `@/lib/utils`.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
