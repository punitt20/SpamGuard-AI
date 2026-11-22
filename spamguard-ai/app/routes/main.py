"""HTML page routes."""
from __future__ import annotations

from flask import Blueprint, current_app, redirect, render_template, url_for

main_bp = Blueprint("main", __name__)


def _render(template: str, page: str, **context):
    return render_template(
        template,
        active_page=page,
        max_message_length=current_app.config["MAX_MESSAGE_LENGTH"],
        **context,
    )


@main_bp.get("/")
def dashboard():
    """Home — redirect to the analyzer."""
    return redirect(url_for("main.analyzer"))


@main_bp.get("/analyzer")
def analyzer():
    """SMS spam analyzer page."""
    return _render("analyzer.html", "analyzer")


@main_bp.get("/model")
def model_info_page():
    """Model information page showing metrics and capabilities."""
    return _render("model-info.html", "model")


@main_bp.get("/analytics")
def analytics_page():
    """Analytics dashboard page with charts and trends."""
    return _render("analytics.html", "analytics")