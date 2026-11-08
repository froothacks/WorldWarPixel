import re
from pathlib import Path

from worldwarpixel.models import model_registry


def test_app_secret_allowlist_covers_model_credentials_and_valid_names() -> None:
    deploy_script = (Path(__file__).parents[1] / "deploy" / "cloudrun.sh").read_text()
    match = re.search(r"(?ms)^APP_SECRETS=\(\n(.*?)^\)", deploy_script)
    assert match is not None

    app_secrets = {
        name for line in match.group(1).splitlines() for name in line.split("#", 1)[0].split()
    }
    model_secrets = {
        secret
        for model in model_registry()
        for endpoint in model.endpoints
        for secret in endpoint.secrets
    }

    assert model_secrets <= app_secrets
    assert all(re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) for name in app_secrets)
