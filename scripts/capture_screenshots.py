"""One-off script to capture README screenshots. Not part of the app."""

from playwright.sync_api import sync_playwright

OUT_DIR = "docs/screenshots"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto("http://localhost:8050", wait_until="networkidle")
        page.wait_for_timeout(1500)

        page.screenshot(path=f"{OUT_DIR}/01_overview_leaderboard.png")

        page.locator("#embedding-graph").scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        page.screenshot(path=f"{OUT_DIR}/02_embedding.png")

        page.locator("#shap-graph").scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        page.screenshot(path=f"{OUT_DIR}/03_feature_importance_shap.png")

        page.locator("#live-sim-play").scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        page.screenshot(path=f"{OUT_DIR}/04_threshold_livesim.png")

        browser.close()


if __name__ == "__main__":
    main()
