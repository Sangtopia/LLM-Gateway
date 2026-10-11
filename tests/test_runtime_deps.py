import subprocess
import sys

# Packages in requirements-dev.txt that the server image (requirements.txt) doesn't have.
OPTIONAL = ["numpy", "fastembed", "fakeredis", "anthropic", "pytest"]


def test_importing_the_app_needs_no_optional_packages():
    # A fresh interpreter, so modules imported by other tests don't count.
    code = "import sys, gateway.main; print(','.join(m for m in %r if m in sys.modules))" % OPTIONAL
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "", f"gateway.main imports optional packages: {result.stdout.strip()}"
