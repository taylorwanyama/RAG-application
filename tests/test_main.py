import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, patch, MagicMock

from app.main import(
    app,
    verify_api_key,
    check_rate_limit,
    get_rag_dependencies
)    
fake_dependencies = MagicMock()

@pytest.mark.anyio
async def test_ask_returns_200():

    mock_answer = "Employees receive 21 days of annual leave each year."

    app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
    app.dependency_overrides[check_rate_limit] = lambda: None
    app.dependency_overrides[get_rag_dependencies] = (
        lambda: fake_dependencies
    )

    try:
        with patch(
            "app.main.answer_question",
            new=AsyncMock(return_value=mock_answer)
        ) as mock_rag:

            transport = ASGITransport(app=app)

            async with AsyncClient(
                transport=transport,
                base_url="http://test"
            ) as client:

                response = await client.post(
                    "/ask",
                    json={
                        "question":
                        "Which tenders are related to medical equipment?"
                    }
                )

        assert response.status_code == 200

        assert response.json() == {
            "Answer": mock_answer
        }

        mock_rag.assert_awaited_once_with(
            "Which tenders are related to medical equipment?",
            fake_dependencies
        )

    finally:
        app.dependency_overrides.clear()

@pytest.mark.anyio
async def test_ask_handles_rag_failure():

    app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
    app.dependency_overrides[check_rate_limit] = lambda: None
    app.dependency_overrides[get_rag_dependencies] = (
        lambda: fake_dependencies
    )

    try:
        with patch(
            "app.main.answer_question",
            new=AsyncMock(
                side_effect=Exception("RAG failed")
            )
        ):

            transport = ASGITransport(app=app)

            async with AsyncClient(
                transport=transport,
                base_url="http://test"
            ) as client:

                response = await client.post(
                    "/ask",
                    json={
                        "question": "Which tenders are related to medical equipment?"
                    }
                )

        assert response.status_code == 500

        assert response.json() == {
            "detail":
            "An internal error occurred while processing the request."
        }

    finally:
        app.dependency_overrides.clear()        