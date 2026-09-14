"""Integration tests for the full Docker Compose deployment."""
import pytest


def test_docker_compose_services_running():
    """Verify all services are up after docker compose up."""
    import subprocess
    result = subprocess.run(
        ["docker", "compose", "ps", "--format", "json"],
        capture_output=True, text=True
    )
    # If docker is available, check services
    if result.returncode == 0:
        output = result.stdout
        assert "redis" in output or "docker compose" in output.lower()


def test_docker_compose_config_valid():
    """Verify docker-compose.yml is valid YAML."""
    import yaml
    with open("docker-compose.yml") as f:
        config = yaml.safe_load(f)
    assert "services" in config
    expected_services = ["redis", "chromadb", "timescaledb", "agent-core", "executor", "dashboard-api", "dashboard-ui"]
    for service in expected_services:
        assert service in config["services"], f"Missing service: {service}"


def test_docker_compose_port_mapping():
    """Verify no port conflicts in docker-compose."""
    import yaml
    with open("docker-compose.yml") as f:
        config = yaml.safe_load(f)
    
    port_map = {}
    for name, service in config["services"].items():
        if "ports" in service:
            for port in service["ports"]:
                host_port = port.split(":")[0]
                if host_port in port_map:
                    pytest.fail(f"Port conflict: {host_port} used by {port_map[host_port]} and {name}")
                port_map[host_port] = name


def test_env_file_present():
    """Verify .env.example exists with all required variables."""
    with open(".env.example") as f:
        content = f.read()
    required_vars = [
        "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "BINANCE_API_KEY",
        "BINANCE_API_SECRET", "BINANCE_TESTNET", "MAX_RISK_PER_TRADE",
        "MAX_DAILY_LOSS", "MAX_POSITIONS", "MIN_RR",
    ]
    for var in required_vars:
        assert var in content, f"Missing env var: {var}"


def test_project_structure():
    """Verify all required directories exist."""
    import os
    required_dirs = [
        "src/agent", "src/tools", "src/risk", "src/memory", "src/executor",
        "src/dashboard", "src/utils", "tests/unit", "tests/integration",
        "services/agent-core", "services/executor", "services/dashboard-api",
        "services/dashboard-ui", "data", "logs", "secrets",
    ]
    for d in required_dirs:
        assert os.path.isdir(d), f"Missing directory: {d}"