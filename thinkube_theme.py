#!/usr/bin/env python3
"""
Thinkube Theme for Gradio
Custom theme matching Thinkube design system
"""

import gradio as gr

# Thinkube color palette (OKLCH converted to RGB)
THINKUBE_COLORS = {
    "light": {
        "primary": "rgb(43, 94, 118)",
        "primary_hover": "rgb(60, 130, 163)",
        "accent": "rgb(191, 168, 115)",
        "background": "rgb(252, 252, 250)",
        "card": "rgb(255, 255, 255)",
        "border": "rgb(229, 229, 228)",
        "text": "rgb(51, 51, 51)",
        "text_muted": "rgb(128, 128, 128)",
        "success": "rgb(89, 142, 108)",
        "warning": "rgb(191, 168, 115)",
        "destructive": "rgb(166, 84, 77)",
    },
    "dark": {
        "primary": "rgb(112, 165, 188)",
        "primary_hover": "rgb(138, 185, 204)",
        "accent": "rgb(209, 185, 118)",
        "background": "rgb(41, 52, 61)",
        "card": "rgb(48, 61, 71)",
        "border": "rgb(106, 130, 143)",
        "text": "rgb(250, 250, 250)",
        "text_muted": "rgb(224, 224, 224)",
        "success": "rgb(139, 191, 156)",
        "warning": "rgb(224, 201, 125)",
        "destructive": "rgb(199, 116, 103)",
    }
}

def create_thinkube_theme():
    """Create a custom Gradio theme matching Thinkube design system"""
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Poppins"), "system-ui", "-apple-system", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Noto Sans Mono"), "Courier New", "monospace"],
        radius_size=gr.themes.sizes.radius_none,
        spacing_size=gr.themes.sizes.spacing_md,
        text_size=gr.themes.sizes.text_md,
    ).set(
        button_primary_background_fill=THINKUBE_COLORS["light"]["primary"],
        button_primary_background_fill_hover=THINKUBE_COLORS["light"]["primary_hover"],
        button_primary_text_color="rgb(255, 255, 255)",
        button_secondary_background_fill=THINKUBE_COLORS["light"]["card"],
        button_secondary_background_fill_hover=THINKUBE_COLORS["light"]["border"],
        button_secondary_text_color=THINKUBE_COLORS["light"]["text"],
        background_fill_primary=THINKUBE_COLORS["light"]["background"],
        background_fill_secondary=THINKUBE_COLORS["light"]["card"],
        border_color_primary=THINKUBE_COLORS["light"]["border"],
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

THINKUBE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Mono:wght@400;500;600;700&display=swap');

body, .gradio-container { font-family: 'Poppins', system-ui, -apple-system, sans-serif !important; }
code, pre, .code { font-family: 'Noto Sans Mono', 'Courier New', monospace !important; }
* { border-radius: 0 !important; }
.gradio-container { background-color: rgb(252, 252, 250) !important; }
.dark .gradio-container { background-color: rgb(41, 52, 61) !important; }
.wrap.svelte-1cl284s .generating, .wrap.svelte-1cl284s .pending, .loading { background-image: none !important; }
.generating::before, .pending::before {
    content: "" !important; display: inline-block !important; width: 32px !important; height: 32px !important;
    background-image: url('/file=/app/icons/tk_ai.svg') !important; background-size: contain !important;
    background-repeat: no-repeat !important; background-position: center !important;
    animation: tk-pulse 1.5s ease-in-out infinite !important; margin: 0 auto !important;
    position: relative !important; left: 50% !important; transform: translateX(-50%) !important;
}
@keyframes tk-pulse {
    0%, 100% { opacity: 0.4; transform: translateX(-50%) scale(0.95); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.1); }
}
a { color: rgb(43, 94, 118) !important; }
.dark a { color: rgb(112, 165, 188) !important; }
input:focus, textarea:focus, select:focus { border-color: rgb(43, 94, 118) !important; outline-color: rgb(43, 94, 118) !important; }
.dark input:focus, .dark textarea:focus, .dark select:focus { border-color: rgb(112, 165, 188) !important; outline-color: rgb(112, 165, 188) !important; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: rgb(229, 229, 228); }
::-webkit-scrollbar-thumb { background: rgb(43, 94, 118); }
::-webkit-scrollbar-thumb:hover { background: rgb(60, 130, 163); }
.dark ::-webkit-scrollbar-track { background: rgb(41, 52, 61); }
.dark ::-webkit-scrollbar-thumb { background: rgb(112, 165, 188); }
.dark ::-webkit-scrollbar-thumb:hover { background: rgb(138, 185, 204); }
.message.bot { background-color: rgb(255, 255, 255) !important; border: 1px solid rgb(229, 229, 228) !important; }
.dark .message.bot { background-color: rgb(48, 61, 71) !important; border: 1px solid rgb(106, 130, 143) !important; }
.message.user { background-color: rgb(43, 94, 118) !important; color: white !important; }
.dark .message.user { background-color: rgb(112, 165, 188) !important; color: rgb(41, 52, 61) !important; }
"""
