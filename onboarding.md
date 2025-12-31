# Onboarding: Setting Up a Fresh Mac

This guide walks you through setting up LearnFinance-2025 on a brand-new macOS machine. We use **Devbox** (powered by Nix) to manage most of the development toolchain.

## Quick Start (One Command)

If you want the fastest path, run the bootstrap script:

```bash
./scripts/bootstrap_macos.sh
```

This will:
1. Install Xcode Command Line Tools (triggers interactive Apple prompt)
2. Install Nix via the Determinate Nix Installer
3. Install Devbox
4. Run `devbox install` to fetch all repo dependencies

Then follow the "Day-to-Day Commands" section below.

---

## Step-by-Step Setup (Manual)

If you prefer to understand each step or need to debug, follow these instructions.

### 1. Install Xcode Command Line Tools

macOS needs compiler toolchains for many development tasks.

```bash
xcode-select --install
```

A dialog will appear — click **Install** and wait for it to complete.

Verify installation:

```bash
xcode-select -p
# Should output something like: /Library/Developer/CommandLineTools
```

### 2. Install Nix (Determinate Nix Installer)

Nix is the package manager that powers Devbox. We use the Determinate Nix Installer (recommended for macOS).

```bash
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```

Follow the prompts (may ask for sudo password).

After installation, **restart your terminal** or source the profile:

```bash
# If using zsh (default on modern macOS):
source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
```

Verify:

```bash
nix --version
# Should output: nix (Nix) 2.x.x
```

### 3. Install Devbox

Devbox provides reproducible dev environments using Nix, without needing to learn Nix syntax.

```bash
curl -fsSL https://get.jetify.com/devbox | bash
```

Restart your terminal or add to PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Verify:

```bash
devbox version
```

### 4. Install Repo Dependencies via Devbox

Navigate to the repo root and install all packages defined in `devbox.json`:

```bash
cd /path/to/LearnFinance-2025
devbox install
```

This installs:
- Python 3.11
- `uv` (Python package manager)
- `git`
- `colima` (Docker runtime for macOS)
- `docker` CLI
- `docker-compose`
- `jq`, `curl` (utilities)

---

## Day-to-Day Commands

All commands below assume you're in the repo root directory.

### Enter Devbox Shell (Recommended)

The easiest way to use the tools is to enter a Devbox shell:

```bash
devbox shell
```

This activates an environment where `python`, `uv`, `docker`, `colima`, etc. are all available.

### Using Devbox Run Scripts

You can also run commands without entering the shell using `devbox run`:

```bash
# See all available scripts
devbox run --list
```

### Start Colima (Docker Runtime)

Before using Docker, start Colima:

```bash
devbox run colima:start
```

To stop later:

```bash
devbox run colima:stop
```

### Start n8n (Orchestrator)

n8n runs in Docker via `docker-compose.yml`:

```bash
devbox run n8n:up
```

Open http://localhost:5678 in your browser.

To stop:

```bash
devbox run n8n:down
```

To view logs:

```bash
devbox run n8n:logs
```

### Set Up Brain API

Install Python dependencies for the Brain API:

```bash
devbox run brain:setup
```

This runs `uv sync --all-extras` in the `brain_api/` directory.

### Run Brain API

Start the FastAPI server:

```bash
devbox run brain:run
```

The API will be available at:
- http://localhost:8000 (API root)
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

The server binds to `0.0.0.0` so it's accessible from Docker containers (e.g., n8n).

### Run Tests

```bash
devbox run brain:test
```

### Lint and Format

```bash
devbox run brain:lint    # Check for issues
devbox run brain:format  # Auto-format code
```

---

## Networking: n8n → Brain API

When n8n runs in Docker and Brain API runs on your host machine:

- n8n must call `http://host.docker.internal:8000` (not `localhost`)
- Brain API must bind to `0.0.0.0` (which `devbox run brain:run` does)

This is already configured in the n8n workflows.

---

## Verification Checklist

After setup, verify everything works:

1. **Devbox environment**:
   ```bash
   devbox shell
   python --version  # Should show 3.11.x
   uv --version
   docker --version
   ```

2. **Colima + Docker**:
   ```bash
   devbox run colima:start
   docker ps  # Should work without errors
   ```

3. **n8n**:
   ```bash
   devbox run n8n:up
   # Open http://localhost:5678 — should see n8n UI
   ```

4. **Brain API**:
   ```bash
   devbox run brain:setup
   devbox run brain:test   # All tests should pass
   devbox run brain:run    # Start server
   # Open http://localhost:8000/docs — should see Swagger UI
   ```

---

## Troubleshooting

### "command not found: devbox" after install

Restart your terminal, or manually add to PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### "Cannot connect to Docker daemon"

Make sure Colima is running:

```bash
devbox run colima:start
```

### Nix not found after install

Restart your terminal or source the daemon script:

```bash
source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
```

### Brain API not accessible from n8n

1. Make sure Brain API is running with `--host 0.0.0.0`
2. In n8n, use `http://host.docker.internal:8000` as the URL

### Xcode CLT install hangs

The install dialog may be hidden behind other windows. Check for a popup or run:

```bash
softwareupdate --list
```

---

## What's Installed Where

| Tool | Managed By | Location |
|------|------------|----------|
| Xcode CLT | Apple | `/Library/Developer/CommandLineTools` |
| Nix | Determinate Installer | `/nix/store/...` |
| Devbox | Jetify | `~/.local/bin/devbox` |
| Python, uv, docker, etc. | Devbox (Nix) | `/nix/store/...` (symlinked via Devbox) |
| Brain API deps | uv | `brain_api/.venv/` |
| n8n | Docker | Container image |

---

## Uninstalling

### Remove Devbox packages (keep Devbox)

```bash
devbox rm --all
```

### Uninstall Devbox

```bash
rm -rf ~/.local/bin/devbox ~/.cache/devbox
```

### Uninstall Nix

```bash
/nix/nix-installer uninstall
```

---

## Summary of Commands

| Task | Command |
|------|---------|
| Bootstrap (one command) | `./scripts/bootstrap_macos.sh` |
| Enter Devbox shell | `devbox shell` |
| Start Colima | `devbox run colima:start` |
| Start n8n | `devbox run n8n:up` |
| Stop n8n | `devbox run n8n:down` |
| Setup Brain API | `devbox run brain:setup` |
| Run Brain API | `devbox run brain:run` |
| Test Brain API | `devbox run brain:test` |
| Lint code | `devbox run brain:lint` |
| Format code | `devbox run brain:format` |

