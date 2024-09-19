import pytest
from playwright.sync_api import Page, expect
import re


@pytest.fixture
def game_page(page: Page):
    page.set_default_timeout(500)  # 500ms default timeout
    expect.set_options(timeout=500)
    page.goto("http://localhost:8000")
    with page.expect_navigation(url=re.compile(r"http://localhost:8000/connect4/\d+")):
        page.click("text=Play Connect Four")
    return page


def test_create_new_game(game_page: Page):
    expect(game_page).to_have_url(re.compile(r"http://localhost:8000/connect4/\d+"))
    expect(game_page.locator(".grid-container")).to_be_visible()


def test_make_move(game_page: Page):
    game_page.click(".grid-cell[data-column='0']")
    expect(game_page.locator(".grid-cell.player1[data-column='0']")).to_be_visible()


def test_game_over(game_page: Page):
    # Simulate moves to create a win condition
    for _ in range(3):
        game_page.click(".grid-cell[data-column='0']")
        game_page.click(".grid-cell[data-column='1']")
    game_page.click(".grid-cell[data-column='0']")
    expect(game_page.locator("#modalBody")).to_contain_text("Wins!")


def test_invalid_move(game_page: Page):
    # Fill a column
    for _ in range(6):
        game_page.click(".grid-cell[data-column='0']")

    # Try to make an invalid move
    game_page.click(".grid-cell[data-column='0']")
    expect(game_page.locator("#errorToast")).to_be_visible()
    expect(game_page.locator("#toastBody")).to_contain_text("Failed")


def test_game_state_updates(game_page: Page):
    # Capture initial state by counting filled cells
    initial_filled_cells = game_page.locator(".grid-cell.player1, .grid-cell.player2").count()

    # Make a move
    empty_cell = game_page.locator(".grid-cell:not(.player1):not(.player2)").first
    empty_cell.click()

    # Wait for the move to be reflected in the UI
    expect(game_page.locator(".grid-cell.player1, .grid-cell.player2")).to_have_count(initial_filled_cells + 1)

    # Capture updated state
    updated_filled_cells = game_page.locator(".grid-cell.player1, .grid-cell.player2").count()

    # Assert that the state has changed
    assert updated_filled_cells == initial_filled_cells + 1, "Game state should update after a move"

    # Optionally, check if the correct player made the move
    new_cell = game_page.locator(".grid-cell.player1").last
    expect(new_cell).to_be_visible()


def test_ai_player_move(page: Page):
    # Start a new game against AI
    page.goto("http://localhost:8000")
    page.click("text=Play Connect Four vs AI")
    page.wait_for_url(re.compile(r"http://localhost:8000/connect4/\d+"))

    # Make a move and wait for AI response
    page.click(".grid-cell[data-column='3']")
    page.wait_for_selector(".grid-cell.player2")
    ai_move = page.locator(".grid-cell.player2")
    expect(ai_move).to_be_visible()


def test_game_reset(game_page: Page):
    expect(game_page.locator(".grid-cell.player1, .grid-cell.player2")).to_have_count(0)
    expect(game_page.locator("#status")).to_contain_text("Current Turn: Player 1")


def test_responsive_layout(game_page: Page):
    # Test desktop layout
    desktop_css = game_page.evaluate(
        'window.getComputedStyle(document.querySelector(".grid-container")).getPropertyValue("grid-template-columns")'
    )
    assert_grid_columns(desktop_css, expected_count=7, expected_size="100px")

    # Test mobile layout
    game_page.set_viewport_size({"width": 375, "height": 667})
    game_page.wait_for_timeout(1000)  # Give time for any responsive JS to run
    mobile_css = game_page.evaluate(
        'window.getComputedStyle(document.querySelector(".grid-container")).getPropertyValue("grid-template-columns")'
    )
    assert_grid_columns(mobile_css, expected_count=7, expected_size="60px")


def assert_grid_columns(css_value: str, expected_count: int, expected_size: str):
    # Split the CSS value into individual column sizes
    columns = css_value.split()

    # Check the number of columns
    assert len(columns) == expected_count, f"Expected {expected_count} columns, but got {len(columns)}"

    # Check each column size
    for column in columns:
        assert column == expected_size, f"Expected column size {expected_size}, but got {column}"

    print(f"Grid columns assertion passed: {css_value}")


def test_game_draw(game_page: Page):
    # Fill the board without a winner
    for col in range(6):
        for _ in range(3):
            game_page.click(f".grid-cell[data-column='{col}']")
            game_page.click(f".grid-cell[data-column='{col}']")
        if col in (2, 5):
            game_page.click(f".grid-cell[data-column='{6}']")
    for _ in range(4):
        game_page.click(f".grid-cell[data-column='{6}']")

    expect(game_page.locator("#modalBody")).to_contain_text("draw")


def test_last_move_highlight(game_page: Page):
    game_page.click(".grid-cell[data-column='3']")
    expect(game_page.locator(".grid-cell.last-move")).to_have_count(1)
    expect(game_page.locator(".grid-cell.last-move")).to_have_attribute("data-column", "3")


def test_unique_game_ids(page):
    page.goto("http://localhost:8000")
    with page.expect_navigation(url=re.compile(r"http://localhost:8000/connect4/\d+")):
        page.click("text=Play Connect Four")
    url_1 = page.url
    assert re.match(r"http://localhost:8000/connect4/\d+", url_1), f"Unexpected URL: {url_1}"

    page.goto("http://localhost:8000")
    with page.expect_navigation(url=re.compile(r"http://localhost:8000/connect4/\d+")):
        page.click("text=Play Connect Four")
    url_2 = page.url
    assert re.match(r"http://localhost:8000/connect4/\d+", url_2), f"Unexpected URL: {url_2}"

    assert url_1 != url_2, f"Game IDs should be unique. Got {url_1} and {url_2}"
