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

    # Select the AI types from the form
    page.select_option("#othelloPlayer1", player1_type)
    page.select_option("#othelloPlayer2", player2_type)

    # Submit the form to start the game
    start_button = page.get_by_text("Start Othello Game")
    expect(start_button).to_be_visible()

    with page.expect_navigation(url=re.compile(r"http://localhost:8000/othello/\d+")):
        start_button.click()

    expect(page.locator(".grid-container")).to_be_visible(timeout=5000)
    return page


def test_create_new_game(game_page: Page):
    expect(game_page).to_have_url(re.compile(r"http://localhost:8000/othello/\d+"))
    expect(game_page.locator(".grid-container")).to_be_visible()


def test_initial_board_state(game_page: Page):
    # Check for the initial four pieces in the center
    expect(game_page.locator(".grid-cell.player1")).to_have_count(2)
    expect(game_page.locator(".grid-cell.player2")).to_have_count(2)


def test_make_move(game_page: Page):
    # Make a move to a valid position
    game_page.click(".grid-cell.legal-move")
    expect(game_page.locator(".grid-cell.player1")).to_have_count(4)  # 2 initial + 1 new + 1 flipped


def test_invalid_move(game_page: Page):
    # Try to make an invalid move
    game_page.click(".grid-cell:not(.legal-move)")
    expect(game_page.locator("#errorToast")).to_be_visible()
    expect(game_page.locator("#toastBody")).to_contain_text("Invalid move")


def test_game_state_updates(game_page: Page):
    initial_filled_cells = game_page.locator(".grid-cell.player1, .grid-cell.player2").count()
    game_page.click(".grid-cell.legal-move")

    expect(game_page.locator(".grid-cell.player1, .grid-cell.player2")).to_have_count(
        initial_filled_cells + 1, timeout=5000
    )


def test_human_vs_ai_player_move(page: Page):
    page = custom_game_page(page, player1_type="human", player2_type="random")

    for i in range(3):  # Simulate three moves
        # Count filled cells before the move
        initial_filled_cells = page.locator(".grid-cell.player1, .grid-cell.player2").count()

        # Human player (player1) makes a move
        page.click(".grid-cell.legal-move")

        # Wait for AI (player2) to make a move and for the cell count to increase
        expect(page.locator(".grid-cell.player1, .grid-cell.player2")).to_have_count(
            initial_filled_cells + 2, timeout=5000
        )

        # Get the new count of filled cells
        new_filled_cells = page.locator(".grid-cell.player1, .grid-cell.player2").count()

        # Optional: Print current board state for debugging
        print(f"Move {i + 1} completed. Filled cells: {initial_filled_cells} -> {new_filled_cells}")

        # Assert that the count has increased
        assert (
            new_filled_cells > initial_filled_cells
        ), f"Cell count did not increase on move {i + 1}. Before: {initial_filled_cells}, After: {new_filled_cells}"


@pytest.mark.timeout(60)
def test_two_random_bots_play_to_end(page: Page):
    page = custom_game_page(page, player1_type="random", player2_type="random")

    max_game_time_ms = 30_000
    start_time = page.evaluate("performance.now()")

    filled_cells_start = 0
    while page.evaluate("performance.now()") - start_time < max_game_time_ms:
        page.wait_for_timeout(1000)  # Wait for 1 second

        if page.locator("#modalBody").is_visible():
            break

        # Check that the game state is changing
        filled_cells_end = page.locator(".grid-cell.player1, .grid-cell.player2").count()
        assert filled_cells_end > filled_cells_start, "Game state should be changing"
        filled_cells_start = filled_cells_end

    modal_body = page.locator("#modalBody")
    expect(modal_body).to_have_text(re.compile(r"(Player 1 Wins|Player 2 Wins|The game is a draw)"))


def test_legal_moves_highlight(game_page: Page):
    legal_moves = game_page.locator(".grid-cell.legal-move")
    expect(legal_moves).to_have_count(greater_than=0)


def test_game_over_no_legal_moves(page: Page):
    # This test simulates a game ending when there are no more legal moves
    # We'll use two random AIs to play until the game ends
    page = custom_game_page(page, player1_type="random", player2_type="random")

    max_game_time_ms = 60_000  # Allow up to 60 seconds for the game to complete
    start_time = page.evaluate("performance.now()")

    while page.evaluate("performance.now()") - start_time < max_game_time_ms:
        page.wait_for_timeout(1000)  # Wait for 1 second

        if page.locator("#modalBody").is_visible():
            break

    expect(page.locator("#modalBody")).to_be_visible()
    expect(page.locator("#modalBody")).to_have_text(re.compile(r"(Player 1 Wins|Player 2 Wins|The game is a draw)"))


def test_unique_game_ids(page):
    page = custom_game_page(page)
    url_1 = page.url
    assert re.match(r"http://localhost:8000/othello/\d+", url_1), f"Unexpected URL: {url_1}"

    page = custom_game_page(page)
    url_2 = page.url
    assert re.match(r"http://localhost:8000/othello/\d+", url_2), f"Unexpected URL: {url_2}"

    assert url_1 != url_2, f"Game IDs should be unique. Got {url_1} and {url_2}"
