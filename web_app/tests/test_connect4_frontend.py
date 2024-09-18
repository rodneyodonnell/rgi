import pytest
from playwright.sync_api import Page, expect, TimeoutError as PlaywrightTimeoutError
from fastapi.testclient import TestClient
from web_app.main import app

DEFAULT_TIMEOUT = 500  # 500ms default timeout


@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


@pytest.fixture(scope="function")
def game_id(api_client):
    response = api_client.post("/games/new", json={"game_type": "connect4", "ai_player": False})
    assert response.status_code == 200, f"Failed to create game: {response.text}"
    return response.json()["game_id"]


@pytest.fixture(scope="module")
def browser_context(browser):
    context = browser.new_context(viewport={"width": 1280, "height": 720}, ignore_https_errors=True)
    context.set_default_timeout(DEFAULT_TIMEOUT)
    return context


@pytest.fixture(scope="function")
def page(browser_context):
    page = browser_context.new_page()
    yield page
    page.close()


def test_game_board_rendering(page: Page, game_id: str, api_client: TestClient):
    # Verify game exists
    response = api_client.get(f"/games/{game_id}/state")
    assert response.status_code == 200, f"Game {game_id} not found: {response.text}"

    # Navigate to the game page
    page.goto(f"http://localhost:8000/connect4/{game_id}")

    try:
        # Check if the game board is rendered correctly
        grid_container = page.locator(".grid-container")
        expect(grid_container).to_be_visible(timeout=DEFAULT_TIMEOUT)

        cells = page.locator(".grid-cell")
        expect(cells).to_have_count(42, timeout=DEFAULT_TIMEOUT)  # 6 rows * 7 columns
    except PlaywrightTimeoutError:
        print(f"Page content: {page.content()}")
        raise


def test_player_move(page: Page, game_id: str, api_client: TestClient):
    # Verify game exists
    response = api_client.get(f"/games/{game_id}/state")
    assert response.status_code == 200, f"Game {game_id} not found: {response.text}"

    # Navigate to the game page
    page.goto(f"http://localhost:8000/connect4/{game_id}")

    try:
        # Make a move
        first_column = page.locator('.grid-cell[data-column="0"]').first
        first_column.click(timeout=DEFAULT_TIMEOUT)

        # Check if a player1 disc is present in the first column
        player1_disc = page.locator('.grid-cell.player1[data-column="0"]')
        expect(player1_disc).to_be_visible(timeout=DEFAULT_TIMEOUT)
    except PlaywrightTimeoutError:
        print(f"Page content: {page.content()}")
        raise


def test_game_over_message(page: Page, api_client: TestClient, game_id: str):
    # Verify game exists
    response = api_client.get(f"/games/{game_id}/state")
    assert response.status_code == 200, f"Game {game_id} not found: {response.text}"

    # Simulate a game-over scenario
    for column in [1, 2, 1, 2, 1, 2, 1]:
        response = api_client.post(f"/games/{game_id}/move", json={"column": column})
        assert response.status_code == 200, f"Failed to make move: {response.text}"

    # Navigate to the game page
    page.goto(f"http://localhost:8000/connect4/{game_id}")

    try:
        # Check if the game-over message is displayed
        game_over_message = page.locator("#modalBody")
        expect(game_over_message).to_be_visible(timeout=DEFAULT_TIMEOUT)
        expect(game_over_message).to_contain_text("Wins!", timeout=DEFAULT_TIMEOUT)
    except PlaywrightTimeoutError:
        print(f"Page content: {page.content()}")
        raise


# Add this to ensure the browser doesn't close immediately after tests
def test_keep_browser_open(page: Page):
    page.pause()
