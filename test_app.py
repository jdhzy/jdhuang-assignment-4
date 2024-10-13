import pytest
from app import app

@pytest.fixture
def client():
    # Set up the test client for the Flask app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage(client):
    # Test that the homepage is reachable and contains the correct content
    response = client.get("/")
    assert response.status_code == 200
    assert b"LSA Search Engine" in response.data

def test_search(client):
    # Test that the search functionality works
    response = client.post("/", data={"query": "test"})
    assert response.status_code == 200
    assert b"Top 5 Results" in response.data