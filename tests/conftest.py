"""
Pytest configuration for LLM Hub tests.

Configures:
- pytest-asyncio for async test support
- Common fixtures
"""

import pytest

# Configure pytest-asyncio mode
pytest_plugins = ('pytest_asyncio',)


def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )


@pytest.fixture
def mock_session():
    """Provide a mock database session for tests"""
    class MockSession:
        def __init__(self):
            self.added = []
            self.committed = False

        def add(self, obj):
            self.added.append(obj)

        def commit(self):
            self.committed = True

        def refresh(self, obj):
            if not hasattr(obj, 'id'):
                obj.id = 1

        def close(self):
            pass

    return MockSession()
