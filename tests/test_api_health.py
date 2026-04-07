def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_messages_endpoint_accepts_post(api_client):
    payload = {"text": "test message", "user_id": "test-user-id", "source": "real"}
    response = api_client.post("/messages", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert "message_id" in data


def test_flags_endpoint_accepts_post(api_client):
    payload = {"message_id": "test-msg-id", "flagged_by": "test-user", "reason": "spam"}
    response = api_client.post("/flags", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert "flag_id" in data
