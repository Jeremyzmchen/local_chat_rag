```markdown
# Design System Strategy: The Forensic Terminal

## 1. Overview & Creative North Star
**Creative North Star: "The Digital Forensicist"**

This design system rejects the "fluff" of modern SaaS in favor of a high-density, hyper-utilitarian aesthetic. It is built for the legal engineer—someone who treats a contract like source code. By leveraging the precision of monospace typography and a "low-chrome" interface, we signal total competence and zero-latency performance.

The system breaks the "standard template" look through **Rigid Asymmetry**. While most dashboards aim for centered balance, this system pushes content to the edges, utilizing a three-panel layout inspired by IDEs (Integrated Development Environments). We use intentional shifts in surface tonality to define boundaries, creating an interface that feels carved out of a single block of charcoal rather than assembled from components.

## 2. Colors & Surface Architecture

The palette is anchored in deep blacks and electric blues, designed to reduce eye strain during prolonged "deep work" sessions.

### Surface Hierarchy & The "No-Line" Rule
Traditional 1px borders are prohibited for sectioning. To maintain a "terminal" feel, boundaries are defined through **background color shifts**.
*   **Base Layer:** The application background is `surface` (#131313).
*   **Primary Workspaces:** Main editors or document views use `surface_container_low` (#1c1b1b).
*   **Utility Panels:** Sidebars and property inspectors use `surface_container` (#20201f).
*   **Elevated Overlays:** Command palettes and context menus use `surface_container_highest` (#353535).

### The Glass & Gradient Rule
While the system is "built by engineers," it is polished by directors. Use **Glassmorphism** for floating elements like risk-score popovers:
*   **Blur:** 12px – 20px backdrop-blur.
*   **Fill:** `surface_container_highest` at 80% opacity.
*   **Signature Texture:** Use a subtle linear gradient on the `primary` (#adc6ff) buttons, transitioning from `primary` to `primary_container` (#4d8eff) at a 45-degree angle. This provides a "glowing phosphor" effect reminiscent of high-end retro-future displays.

## 3. Typography: Monospace Authority

We utilize a strictly monospace-driven hierarchy to reinforce the "legal-as-code" metaphor.

*   **Display & Headlines:** Use `spaceGrotesk` (sm/md/lg). While the body is monospace, these headers provide an "Editorial Brutalist" feel, offering a sharp, geometric contrast to the density of the data.
*   **Body & Labels:** Use `inter` (sm/md/lg). Wait—per the brand direction, we treat these with the *logic* of a code editor. All technical data, diff views, and logs must be rendered in a monospace typeface (JetBrains Mono or Fira Code) to ensure character alignment for document comparisons.
*   **The Identity Logic:** The mix of Space Grotesk for high-level "Review Stats" and Inter/Monospace for "Contract Text" creates a hierarchy of **Summary vs. Source**.

## 4. Elevation & Depth

### The Layering Principle
We do not use drop shadows to indicate importance; we use **Tonal Stacking**. 
1.  **Level 0:** `surface_container_lowest` (Background).
2.  **Level 1:** `surface_container_low` (Main Review Area).
3.  **Level 2:** `surface_container_high` (Active Panel/Hover State).

### Ambient Shadows & Ghost Borders
When a modal must float (e.g., an AI Suggestion Box):
*   **Shadow:** `0px 8px 32px rgba(0, 0, 0, 0.4)`. 
*   **Ghost Border:** If a boundary is required for accessibility, use a 1px border with `outline_variant` (#424754) at **20% opacity**. It should be felt, not seen.

## 5. Components & High-Density UI

### High-Density Tables
*   **Rule:** No dividers. 
*   **Separation:** Use alternating row fills (Zebra striping) with `surface_container_low` and `surface_container_lowest`.
*   **Typography:** All table data uses `label-sm` (0.6875rem) to maximize information density.

### Status Chips & Risk Indicators
*   **Risk Score (0-100):** A circular gauge using `tertiary` (#ffb786) for high risk and `primary` (#adc6ff) for low risk.
*   **Status Chips:** Rectangular with `DEFAULT` (0.25rem) corner radius. Use `secondary_container` for the background and `on_secondary_container` for text. No icons unless they represent a specific Git-style action (e.g., merge, conflict).

### Git-Style Diff Views
*   **Additions:** Background `primary_container` at 20% opacity with `primary` text.
*   **Deletions:** Background `error_container` at 20% opacity with `error` text.
*   **Alignment:** Strict monospace grid to ensure clauses align perfectly across the three-panel view.

### Input Fields & Terminal Logs
*   **Inputs:** `none` border-radius. Use a 2px bottom border of `outline` (#8c909f) that transitions to `primary` (#adc6ff) on focus.
*   **Terminal Logs:** Use `surface_container_lowest` with a slight `surface_bright` inner-glow top border to simulate a recessed screen.

## 6. Do’s and Don’ts

### Do
*   **Do** use extreme information density. If there is more than 16px of whitespace between two data points, tighten it.
*   **Do** use "Phosphor Glow" (subtle outer glow) on the `primary` accent color for active AI processing states.
*   **Do** favor vertical layouts that mimic a code IDE (Left: File Tree, Center: Document, Right: Meta-data).

### Don't
*   **Don't** use rounded corners larger than `md` (0.375rem). The system should feel sharp and precise.
*   **Don't** use standard "Select" dropdowns. Use command palettes (CMD+K) style inputs to cater to power users.
*   **Don't** use 100% opaque borders to separate panels. Use background color shifts. If it looks like a "box," you’ve failed; it should look like a "region."

### Accessibility Note
Despite the high density, ensure that all `on_surface_variant` text meets a 4.5:1 contrast ratio against the `surface_container` tiers. Precision does not excuse illegibility.```