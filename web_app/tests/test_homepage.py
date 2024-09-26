import re

from playwright.sync_api import Page, expect


def test_homepage_new_game(page: Page) -> None:
    # Navigate to the homepage
    page.goto("http://localhost:8000")

    expect(page).to_have_title(re.compile(r"RGI Game Portal"))
    expect(page.locator("text=Start Connect 4 Game")).to_be_visible()
    expect(page.locator("text=Start Othello Game")).to_be_visible()
