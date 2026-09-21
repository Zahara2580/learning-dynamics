"""Define shared figure colours, font sizes and legend styles."""

import os

SCALE = float(os.environ.get("FIGSTYLE_SCALE", "1.0"))
HEIGHT = float(os.environ.get("FIGSTYLE_HEIGHT", "1.0"))

MODEL_COLOUR = {"t5":"#0B3D91", "byt5": "#FF6EB4", "nguni-byt5": "#8E44AD"}
MODEL_ORDER = ["t5", "byt5", "nguni-byt5"]
MODEL_DISPLAY = {"t5": "T5", "byt5": "ByT5", "nguni-byt5": "Nguni-ByT5"}

POOL_COLOUR = {"first": "#BE8A4E", "mean": "#2A9D8F", "last": "#C0392B"}

BASE_SWATCH = "black"


def base_model(m: str) -> str:
    """Remove the bilingual suffix from a model name."""
    return m.removesuffix("-bilingual")


def colour(m: str) -> str:
    """Return the colour assigned to a model family."""
    return MODEL_COLOUR.get(base_model(m), "#555555")


def display_name(m: str) -> str:
    """Return the display label for a model and its training arm."""
    n = MODEL_DISPLAY.get(base_model(m), base_model(m))
    return f"{n} (bilingual)" if m.endswith("-bilingual") else n


def model_key(m: str):
    """Return the model-family ordering key for figures."""
    b = base_model(m)
    return (MODEL_ORDER.index(b) if b in MODEL_ORDER else 99, m)


def fs(name: str) -> float:
    """Return a scaled font size for a figure element."""
    return {"tick": 13, "label": 15, "title": 15, "letter": 17,
            "legend": 14, "suptitle": 16}[name] * SCALE


def size(w: float, h: float) -> tuple[float, float]:
    """Return the scaled figure width and height."""
    return (w * SCALE, h * SCALE * HEIGHT)


def bold_axes(ax) -> None:
    """Apply shared tick sizing and bold tick labels."""
    ax.tick_params(labelsize=fs("tick"), width=1.2 * SCALE)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")


def boxed_legend(fig, handles, labels, ncol=None, y=-0.055):
    """Add a styled legend beneath a figure."""
    leg = fig.legend(handles, labels, ncol=ncol or len(labels),
                     loc="lower center", fontsize=fs("legend"),
                     frameon=True, fancybox=False, edgecolor="0.35",
                     framealpha=1.0, borderpad=0.7,
                     columnspacing=2.4 * SCALE,
                     bbox_to_anchor=(0.5, y))
    leg.get_frame().set_linewidth(1.4 * SCALE)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    return leg
