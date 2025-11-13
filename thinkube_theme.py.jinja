#!/usr/bin/env python3
"""
Thinkube Theme for Gradio
Custom theme matching Thinkube design system
"""

import gradio as gr

# Thinkube color palette (OKLCH converted to RGB)
THINKUBE_COLORS = {
    # Light mode
    "light": {
        "primary": "rgb(43, 94, 118)",          # oklch(38% 0.15 220) - Teal
        "primary_hover": "rgb(60, 130, 163)",   # Lighter teal for hover
        "accent": "rgb(191, 168, 115)",         # oklch(65% 0.10 75) - Yellow-amber
        "background": "rgb(252, 252, 250)",     # oklch(99% 0.005 75) - Warm off-white
        "card": "rgb(255, 255, 255)",           # Pure white for cards
        "border": "rgb(229, 229, 228)",         # oklch(90% 0.005 220)
        "text": "rgb(51, 51, 51)",              # oklch(20% 0 0) - Dark gray
        "text_muted": "rgb(128, 128, 128)",     # oklch(50% 0 0) - Medium gray
        "success": "rgb(89, 142, 108)",         # oklch(50% 0.12 145) - Muted green
        "warning": "rgb(191, 168, 115)",        # oklch(65% 0.12 80) - Muted amber
        "destructive": "rgb(166, 84, 77)",      # oklch(55% 0.15 25) - Toned red
    },
    # Dark mode
    "dark": {
        "primary": "rgb(112, 165, 188)",        # oklch(60% 0.13 220) - Brighter teal
        "primary_hover": "rgb(138, 185, 204)",  # Even lighter teal
        "accent": "rgb(209, 185, 118)",         # oklch(72% 0.12 70) - Bright yellow-amber
        "background": "rgb(41, 52, 61)",        # oklch(24% 0.06 220) - Dark teal-gray
        "card": "rgb(48, 61, 71)",              # oklch(26% 0.06 220) - Brighter card
        "border": "rgb(106, 130, 143)",         # oklch(52% 0.10 220) - Much brighter border
        "text": "rgb(250, 250, 250)",           # oklch(98% 0 0) - Bright text
        "text_muted": "rgb(224, 224, 224)",     # oklch(88% 0 0) - Muted text
        "success": "rgb(139, 191, 156)",        # oklch(68% 0.14 145) - Brighter green
        "warning": "rgb(224, 201, 125)",        # oklch(78% 0.14 75) - Brighter amber
        "destructive": "rgb(199, 116, 103)",    # oklch(68% 0.17 25) - Brighter red
    }
}

# Create Thinkube theme
def create_thinkube_theme():
    """Create a custom Gradio theme matching Thinkube design system"""

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[
            gr.themes.GoogleFont("Poppins"),
            "system-ui",
            "-apple-system",
            "sans-serif"
        ],
        font_mono=[
            gr.themes.GoogleFont("Noto Sans Mono"),
            "Courier New",
            "monospace"
        ],
        radius_size=gr.themes.sizes.radius_none,  # Sharp corners (Thinkube style)
        spacing_size=gr.themes.sizes.spacing_md,
        text_size=gr.themes.sizes.text_md,
    ).set(
        # Light mode colors
        button_primary_background_fill=THINKUBE_COLORS["light"]["primary"],
        button_primary_background_fill_hover=THINKUBE_COLORS["light"]["primary_hover"],
        button_primary_text_color="rgb(255, 255, 255)",
        button_secondary_background_fill=THINKUBE_COLORS["light"]["card"],
        button_secondary_background_fill_hover=THINKUBE_COLORS["light"]["border"],
        button_secondary_text_color=THINKUBE_COLORS["light"]["text"],
        background_fill_primary=THINKUBE_COLORS["light"]["background"],
        background_fill_secondary=THINKUBE_COLORS["light"]["card"],
        border_color_primary=THINKUBE_COLORS["light"]["border"],

        # Dark mode colors
        button_primary_background_fill_dark=THINKUBE_COLORS["dark"]["primary"],
        button_primary_background_fill_hover_dark=THINKUBE_COLORS["dark"]["primary_hover"],
        button_primary_text_color_dark="rgb(255, 255, 255)",
        button_secondary_background_fill_dark=THINKUBE_COLORS["dark"]["card"],
        button_secondary_background_fill_hover_dark=THINKUBE_COLORS["dark"]["border"],
        button_secondary_text_color_dark=THINKUBE_COLORS["dark"]["text"],
        background_fill_primary_dark=THINKUBE_COLORS["dark"]["background"],
        background_fill_secondary_dark=THINKUBE_COLORS["dark"]["card"],
        border_color_primary_dark=THINKUBE_COLORS["dark"]["border"],
    )

    return theme


# Custom CSS for Thinkube styling
THINKUBE_CSS = """
/* Import Thinkube fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Mono:wght@400;500;600;700&display=swap');

/* Apply Thinkube fonts globally */
body, .gradio-container {
    font-family: 'Poppins', system-ui, -apple-system, sans-serif !important;
}

code, pre, .code {
    font-family: 'Noto Sans Mono', 'Courier New', monospace !important;
}

/* Ensure zero border radius everywhere (Thinkube style) */
* {
    border-radius: 0 !important;
}

/* Thinkube background colors */
.gradio-container {
    background-color: rgb(252, 252, 250) !important;
}

.dark .gradio-container {
    background-color: rgb(41, 52, 61) !important;
}

/* Hide default Gradio loading animations */
.wrap.svelte-1cl284s .generating,
.wrap.svelte-1cl284s .pending,
.loading {
    background-image: none !important;
}

/* Custom pulsing tk_ai.svg loading animation */
.generating::before,
.pending::before {
    content: "" !important;
    display: inline-block !important;
    width: 32px !important;
    height: 32px !important;
    background-image: url('/file=/app/icons/tk_ai.svg') !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    animation: tk-pulse 1.5s ease-in-out infinite !important;
    margin: 0 auto !important;
    position: relative !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
}

@keyframes tk-pulse {
    0%, 100% {
        opacity: 0.4;
        transform: translateX(-50%) scale(0.95);
    }
    50% {
        opacity: 1;
        transform: translateX(-50%) scale(1.1);
    }
}

/* Thinkube primary color for links and accents */
a {
    color: rgb(43, 94, 118) !important;
}

.dark a {
    color: rgb(112, 165, 188) !important;
}

/* Thinkube teal for focus states */
input:focus, textarea:focus, select:focus {
    border-color: rgb(43, 94, 118) !important;
    outline-color: rgb(43, 94, 118) !important;
}

.dark input:focus, .dark textarea:focus, .dark select:focus {
    border-color: rgb(112, 165, 188) !important;
    outline-color: rgb(112, 165, 188) !important;
}

/* Custom scrollbar with Thinkube colors */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: rgb(229, 229, 228);
}

::-webkit-scrollbar-thumb {
    background: rgb(43, 94, 118);
}

::-webkit-scrollbar-thumb:hover {
    background: rgb(60, 130, 163);
}

.dark ::-webkit-scrollbar-track {
    background: rgb(41, 52, 61);
}

.dark ::-webkit-scrollbar-thumb {
    background: rgb(112, 165, 188);
}

.dark ::-webkit-scrollbar-thumb:hover {
    background: rgb(138, 185, 204);
}

/* Enhance chat interface with Thinkube styling */
.message.bot {
    background-color: rgb(255, 255, 255) !important;
    border: 1px solid rgb(229, 229, 228) !important;
}

.dark .message.bot {
    background-color: rgb(48, 61, 71) !important;
    border: 1px solid rgb(106, 130, 143) !important;
}

.message.user {
    background-color: rgb(43, 94, 118) !important;
    color: white !important;
}

.dark .message.user {
    background-color: rgb(112, 165, 188) !important;
    color: rgb(41, 52, 61) !important;
}
"""
