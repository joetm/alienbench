def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: live API call test (requires API keys; run with: pytest -m live)",
    )
