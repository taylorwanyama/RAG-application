import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, patch

from app.main import app


@pytest.mark.anyio
async def test_ask_returns_200():

    mock_answer = "Employees receive 21 days of annual leave each year."

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
                    "question": "How many days of annual leave do employees get?"
                }
            )

    assert response.status_code == 200

    assert response.json() == {
        "Answer": mock_answer
    }

    mock_rag.assert_awaited_once_with(
        "How many days of annual leave do employees get?"
    )
    print("MOCK WAS AWAITED:", mock_rag.await_count)