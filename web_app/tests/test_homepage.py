import pytest
from playwright.sync_api import Page, expect
import re


def test_homepage_new_game(page: Page):
    # Navigate to the homepage
    page.goto("http://localhost:8000")

    # Check if we're on the homepage
    expect(page).to_have_title("Home - RGI Game Portal")

    # Click the "Play Connect Four" button
    play_button = page.get_by_role("button", name="Play Connect Four", exact=True)
    play_button.click()

    # Wait for navigation to complete
    page.wait_for_url(re.compile(r"/connect4/\d+"))

    # Check if the URL matches the expected pattern
    current_url = page.url
    assert re.match(r"http://localhost:8000/connect4/\d+", current_url), f"Unexpected URL: {current_url}"

    # Check if the game grid is visible
    grid_container = page.locator(".grid-container")
    expect(grid_container).to_be_visible()

    # Check if we have the correct number of cells
    cells = page.locator(".grid-cell")
    expect(cells).to_have_count(42)  # 6 rows * 7 columns

    print(f"Successfully created and loaded new game at {current_url}")
