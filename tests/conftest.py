collect_ignore = [
    "test_api.py",
    "test_llm_client.py",
    "test_my_calculator.py",
    "test_planandsolve_agent.py",
    "test_reflection_agent.py",
    "test_simple_agent.py",
]


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "online" not in item.keywords and "integration" not in item.keywords:
            item.add_marker("unit")
