import pytest
from playwright.sync_api import Page, expect
import re

# max timeout of 5s for each test
pytestmark = pytest.mark.timeout(5)


@pytest.fixture(scope="function")
def game_page(page: Page):
    return custom_game_page(page)


def custom_game_page(page: Page, timeout_ms=500, player1_type: str = "human", player2_type: str = "human"):
    page.set_default_timeout(timeout_ms)
    expect.set_options(timeout=timeout_ms)

    # Go to the main page
    page.goto("http://localhost:8000")

    # Select the AI types from the form (assuming these are select elements with specific IDs)
    page.select_option("#connect4Player1", player1_type)
    page.select_option("#connect4Player2", player2_type)

    # Submit the form to start the game
    start_button = page.get_by_text("Start Connect 4 Game")
    expect(start_button).to_be_visible()

    with page.expect_navigation(url=re.compile(r"http://localhost:8000/connect4/\d+")):
        start_button.click()

    expect(page.locator(".grid-container")).to_be_visible(timeout=5000)
    return page


def test_create_new_game(game_page: Page):
    expect(game_page).to_have_url(re.compile(r"http://localhost:8000/connect4/\d+"))
    expect(game_page.locator(".grid-container")).to_be_visible()


def test_make_move(game_page: Page):
    game_page.click(".grid-cell[data-column='0']")
    expect(game_page.locator(".grid-cell.player1")).to_be_visible()


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
        print("Click!", _)
        game_page.click(".grid-cell[data-column='0']", timeout=500)

    # Try to make an invalid move
    game_page.click(".grid-cell[data-column='0']")
    expect(game_page.locator("#errorToast")).to_be_visible()
    expect(game_page.locator("#toastBody")).to_contain_text("Invalid move")
    print("done.")


def test_game_state_updates(game_page: Page):
    # Capture initial state by counting filled cells
    initial_filled_cells = game_page.locator(".grid-cell.player1, .grid-cell.player2").count()

    # Make a move
    empty_cell = game_page.locator(".grid-cell:not(.player1):not(.player2)").first
    empty_cell.click()

    # Wait for the move to be reflected in the UI
    game_page.wait_for_timeout(1000)  # Wait for 1 second

    # Capture updated state
    updated_filled_cells = game_page.locator(".grid-cell.player1, .grid-cell.player2").count()

    # Assert that the state has changed
    assert updated_filled_cells == initial_filled_cells + 1, "Game state should update after a move"


def test_human_vs_ai_player_move(page: Page):
    page = custom_game_page(page, player1_type="human", player2_type="random")

    for i in range(3):  # Simulate three moves
        # Human player (player1) makes a move
        page.click(f".grid-cell[data-column='{i}']")

        # Wait for AI (player2) to make a move in response by locating the last player2 move
        ai_move = page.locator(".grid-cell.player2").last
        page.wait_for_selector(".grid-cell.player2", timeout=5000)

        # Check that the latest AI move is visible
        expect(ai_move).to_be_visible()


def test_two_random_bots_play_to_end(page: Page):
    # Setup the game with two random AI players
    page = custom_game_page(page, player1_type="random", player2_type="random")

    # Define the maximum time for the game to complete (30 seconds)
    max_game_time_ms = 30_000
    start_time = page.evaluate("performance.now()")

    # Helper function to count pieces on the board
    def count_pieces():
        player1_pieces = page.locator(".grid-cell.player1").count()
        player2_pieces = page.locator(".grid-cell.player2").count()
        return player1_pieces + player2_pieces

    previous_count = count_pieces()

    # Keep checking every second until game ends or timeout
    while page.evaluate("performance.now()") - start_time < max_game_time_ms:
        page.wait_for_timeout(1000)  # Wait for 1 second

        # Get the current piece count
        current_count = count_pieces()

        # Check if the game has ended (look for the game-over modal)
        if page.locator("#modalBody").is_visible():
            break

        # Assert that pieces are being added
        assert current_count > previous_count, "No new pieces added in the last second."
        previous_count = current_count

    # After exiting the loop, check for a win/loss/draw message
    modal_body = page.locator("#modalBody")
    expect(modal_body).to_have_text(re.compile(r"(Player 1 Wins|Player 2 Wins|Draw)"))


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
    page = custom_game_page(page)
    url_1 = page.url
    assert re.match(r"http://localhost:8000/connect4/\d+", url_1), f"Unexpected URL: {url_1}"

    page = custom_game_page(page)
    url_2 = page.url
    assert re.match(r"http://localhost:8000/connect4/\d+", url_2), f"Unexpected URL: {url_2}"

    assert url_1 != url_2, f"Game IDs should be unique. Got {url_1} and {url_2}"
