import os
import re
import base64
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path
import subprocess
import json
import shutil
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.responses import JSONResponse, FileResponse
from fastapi.responses import HTMLResponse
# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

STUDENT_SECRET = os.getenv("STUDENT_SECRET")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_BASE = os.getenv("AIPIPE_BASE", "https://aipipe.org/openai/v1")
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "gpt-4o-mini")
EVALUATION_SECRET = os.getenv("EVALUATION_SECRET", "")


# ---------------------------
# Utilities
# ---------------------------
def log(message: str):
    """Print with timestamp prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def run_shell(cmd: str, cwd: Optional[Path] = None) -> str:
    """Run a shell command and return stdout. Raise on error."""
    try:
        log(f"Running command: {cmd}")
        result = subprocess.run(
            cmd, cwd=cwd, shell=True, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        log(f"ERROR: Command failed: {cmd}\n{e.stderr}")
        raise


# ---------------------------
# Pydantic Models
# ---------------------------
class Attachment(BaseModel):
    name: str
    url: str


class TaskRequest(BaseModel):
    email: str
    secret: str
    task: str
    round: int = Field(alias="round")
    nonce: str
    brief: str
    checks: List[str]
    evaluation_url: str
    attachments: Optional[List[Attachment]] = []

    class Config:
        populate_by_name = True


class StatusResponse(BaseModel):
    status: str
    repo_url: str | None = None
    commit_sha: str | None = None
    pages_url: str | None = None


class ErrorResponse(BaseModel):
    error: str


# ---------------------------
# App Lifespan
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log("🚀 FastAPI service starting up")
    if not STUDENT_SECRET:
        log("⚠️ STUDENT_SECRET not found in environment variables")
    if not AIPIPE_TOKEN:
        log("⚠️ AIPIPE_TOKEN not set — LLM generation disabled (fallback only).")
    yield
    log("🛑 FastAPI service shutting down")


app = FastAPI(lifespan=lifespan)


# ---------------------------
# Attachment Saver
# ---------------------------
def save_attachments(attachments: List[Attachment], workdir: Path) -> None:
    attachments_dir = workdir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    log(f"Saving {len(attachments)} attachments to {attachments_dir}")

    for att in attachments:
        try:
            if att.url.startswith("data:"):
                header, encoded = att.url.split(",", 1)
                data = base64.b64decode(encoded)
                filepath = attachments_dir / att.name
                filepath.write_bytes(data)
                log(f"✅ Saved data URI: {att.name} ({len(data)} bytes)")
            elif att.url.startswith("http"):
                res = requests.get(att.url, timeout=10)
                if res.status_code == 200:
                    filepath = attachments_dir / att.name
                    filepath.write_bytes(res.content)
                    log(f"✅ Downloaded: {att.name} ({len(res.content)} bytes)")
                else:
                    log(f"⚠️ Skipped {att.name}: HTTP {res.status_code}")
        except Exception as e:
            log(f"❌ Error saving {att.name}: {e}")


# ---------------------------
# Template Detection
# ---------------------------
def detect_template(brief: str) -> str:
    text = brief.lower()
    if "captcha" in text:
        return "captcha-solver"
    elif "markdown" in text:
        return "markdown-to-html"
    elif "github" in text and ("user" in text or "created" in text):
        return "github-user-created"
    else:
        return "sum-of-sales"

@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h1>Service is running!</h1><p>Use /api/request for tasks.</p>"

# ---------------------------
# Static App Generator (core)
# ---------------------------
def generate_static_app(template_id: str, payload, workdir: Path) -> Path:
    """Generate static app files either from LLM output or fallback HTML."""
    if hasattr(payload, "dict"):
        payload = payload.dict()

    site_dir = workdir / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    log(f"[generate_static_app] Generating app for template: {template_id}")

    llm_files = payload.get("llm_files")

    if llm_files and isinstance(llm_files, dict):
        log("[generate_static_app] ✅ Using LLM-generated files")
        for filename, content in llm_files.items():
            filepath = site_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding="utf-8")
            log(f"📝 Wrote {filename} ({len(content)} bytes)")
    else:
        log("[generate_static_app] ⚠️ No valid LLM output found, using fallback template")

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{payload.get('title', 'Generated App')}</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>{payload.get('header', 'Welcome')}</h1>
    <p>{payload.get('description', 'This is a fallback static app.')}</p>
    <script src="script.js"></script>
</body>
</html>
"""
        css_content = """body { font-family: Arial; padding: 20px; background: #fafafa; }"""
        js_content = f"""console.log("Fallback app for template: {template_id}");"""

        (site_dir / "index.html").write_text(html_content, encoding="utf-8")
        (site_dir / "style.css").write_text(css_content, encoding="utf-8")
        (site_dir / "script.js").write_text(js_content, encoding="utf-8")
        log(f"[generate_static_app] Fallback app saved to {site_dir}")

    return site_dir / "index.html"


# ---------------------------
# LLM Integration
# ---------------------------
def call_aipipe_chat(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> Dict[str, Any]:
    if not AIPIPE_TOKEN:
        raise RuntimeError("AIPIPE_TOKEN not configured")

    url = f"{AIPIPE_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"}

    payload = {
        "model": AIPIPE_MODEL,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    log(f"Calling AI Pipe at {url} with model {AIPIPE_MODEL}")
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def llm_generate_files_from_brief(payload: TaskRequest) -> Optional[Dict[str, str]]:
    system_prompt = (
        "You are a safe web app generator. Return only a JSON object mapping filenames to text content. "
        "Use keys: index.html, script.js, README.md, LICENSE. Return valid JSON only."
    )

    user_prompt = (
        f"Brief: {payload.brief}\nTask: {payload.task}\nRound: {payload.round}\n"
        f"Attachments: {[a.name for a in (payload.attachments or [])]}"
    )

    try:
        resp = call_aipipe_chat(system_prompt, user_prompt, max_tokens=1600)
        choices = resp.get("choices") or []
        if not choices:
            log("LLM returned no choices.")
            return None

        msg = choices[0].get("message", {}).get("content") or ""
        text = re.sub(r"^```(?:json)?\n|\n```$", "", msg.strip())
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        log(f"LLM generation failed: {e}")
    return None


def write_generated_files(file_map: Dict[str, str], workdir: Path) -> None:
    site_dir = workdir / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, content in file_map.items():
        target = site_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        log(f"📝 Wrote {rel_path} ({len(content)} bytes)")


def generate_with_llm(payload: TaskRequest, workdir: Path) -> bool:
    if not AIPIPE_TOKEN:
        log("No AIPIPE_TOKEN configured — skipping LLM generation.")
        return False

    log("🌩️ Attempting LLM-assisted generation")
    file_map = llm_generate_files_from_brief(payload)
    if not file_map:
        return False

    try:
        write_generated_files(file_map, workdir)
        log("✅ LLM-assisted generation completed.")
        return True
    except Exception as e:
        log(f"Error writing LLM files: {e}")
        return False


# ---------------------------
# GitHub Deployment
# ---------------------------
def sanitize_repo_name(name: str) -> str:
    name = re.sub(r'[^a-z0-9-]', '-', name.lower())
    name = re.sub(r'-+', '-', name)
    return name.strip('-')


def scan_for_secrets(site_dir: Path):
    """Run Trufflehog and Gitleaks. Abort if secrets are found."""
    log(f"🔒 Running secret scans in {site_dir}")

    try:
        trufflehog_output = run_shell("trufflehog filesystem --json", cwd=site_dir)
        if trufflehog_output.strip():
            findings = json.loads(trufflehog_output)
            if findings:
                log(f"❌ Trufflehog found secrets: {len(findings)} items")
                raise Exception("Secrets detected by Trufflehog")
    except FileNotFoundError:
        log("⚠️ Trufflehog not installed, skipping")
    except subprocess.CalledProcessError as e:
        log(f"⚠️ Trufflehog error, skipping: {e}")

    gitleaks_report = site_dir / "gitleaks.json"
    gitleaks_report.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_shell(f"gitleaks detect --no-banner --source . --report-path {gitleaks_report}", cwd=site_dir)
        if gitleaks_report.exists() and gitleaks_report.stat().st_size > 2:
            log(f"❌ Gitleaks found secrets, see {gitleaks_report}")
            raise Exception("Secrets detected by Gitleaks")
    except FileNotFoundError:
        log("⚠️ Gitleaks not installed, skipping")
    except subprocess.CalledProcessError as e:
        log(f"⚠️ Gitleaks error, skipping: {e}")

    log("✅ Secret scan passed")


def deploy_to_github(workdir: Path, payload: TaskRequest):
    site_dir = workdir / "site"
    if not site_dir.exists():
        raise Exception("❌ Site directory not found — cannot deploy.")

    log(f"📦 Moving site contents from {site_dir} to {workdir}")
    for item in site_dir.iterdir():
        target = workdir / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))
    shutil.rmtree(site_dir)

    try:
        scan_for_secrets(workdir)
    except Exception as e:
        log(f"⚠️ Secret scan failed/skipped: {e}")

    repo_base = f"llm-task-{payload.task}"
    repo = sanitize_repo_name(repo_base)

    try:
        run_shell(f"gh repo view {GITHUB_USERNAME}/{repo}")
        short_nonce = payload.nonce[:6]
        repo = f"{repo}-{short_nonce}"
        log(f"ℹ️ Repo already exists, using new name: {repo}")
    except Exception:
        log(f"✅ Repo name available: {repo}")

    license_file = workdir / "LICENSE"
    if not license_file.exists():
        license_file.write_text("MIT License\nCopyright 2025")
        log("🪪 LICENSE file created.")

    if not (workdir / ".git").exists():
        run_shell("git init", cwd=workdir)
        run_shell("git branch -M main", cwd=workdir)

    run_shell(f'git config user.name "{GITHUB_USERNAME}"', cwd=workdir)
    run_shell('git config user.email "actions@github.com"', cwd=workdir)

    run_shell("git add .", cwd=workdir)
    status = run_shell("git status --porcelain", cwd=workdir)
    commit_sha = None  # default if no commit is made
    if status.strip():
        run_shell('git commit -m "Deploy app"', cwd=workdir)
        commit_sha = run_shell("git rev-parse HEAD", cwd=workdir)
        log(f"✅ Changes committed. Commit SHA: {commit_sha}")
        log("✅ Changes committed.")
    else:
        log("⚠️ No changes to commit (skipping commit step).")

    commit_sha = run_shell("git rev-parse HEAD", cwd=workdir)

    try:
        run_shell(f"gh repo create {GITHUB_USERNAME}/{repo} --public --source . --remote origin --push", cwd=workdir)
        log("✅ Repository created and pushed successfully.")
    except Exception as e:
        log(f"⚠️ Repo creation failed: {e}")
        run_shell(f"git remote add origin https://github.com/{GITHUB_USERNAME}/{repo}.git", cwd=workdir)
        run_shell("git push -u origin main", cwd=workdir)
        log("✅ Repo pushed using fallback remote.")

    # -----------------------------
    # Generic GitHub Pages deployment
    # -----------------------------
    pages_payload = {
        "source": {
            "branch": "main",
            "path": "/"
        }
    }

    pages_file = workdir / "pages.json"
    pages_file.write_text(json.dumps(pages_payload))

    try:
        run_shell(
    f'gh api -X POST repos/{GITHUB_USERNAME}/{repo}/pages '
    f'-H "Accept: application/vnd.github+json" --input {pages_file.absolute()}',
    cwd=workdir
)
        log("🌐 GitHub Pages enabled successfully (modern API)")
    except Exception as e:
        log(f"⚠️ GitHub Pages enable failed: {e}")


    repo_url = f"https://github.com/{GITHUB_USERNAME}/{repo}"
    pages_url = f"https://{GITHUB_USERNAME}.github.io/{repo}/"
    log(f"✅ Deployment complete:\nRepo: {repo_url}\nPages: {pages_url}")

    return repo_url, commit_sha.strip(), pages_url


# ---------------------------
# In-memory store for deployment results
# ---------------------------
deployment_results: Dict[str, Dict[str, str]] = {}  # nonce -> repo_url, commit_sha, pages_url

# ---------------------------
# API Endpoints
# ---------------------------
@app.post("/api/request", response_model=StatusResponse)
async def handle_request(payload: TaskRequest, background_tasks: BackgroundTasks):
    if payload.secret != STUDENT_SECRET:
        raise HTTPException(status_code=400, detail="Invalid student secret")

    log(f"✅ Task received: {payload.task} (Round {payload.round})")
    workdir = Path(f"workdir_{payload.nonce}")
    workdir.mkdir(exist_ok=True)

    if payload.attachments:
        save_attachments(payload.attachments, workdir)

    llm_ok = generate_with_llm(payload, workdir)
    if not llm_ok:
        template_id = detect_template(payload.brief)
        generate_static_app(template_id, payload, workdir)

    # Background deployment
    def deploy_task():
        try:
            repo_url, commit_sha, pages_url = deploy_to_github(workdir, payload)
            deployment_results[payload.nonce] = {
                "repo_url": repo_url,
                "commit_sha": commit_sha,
                "pages_url": pages_url,
            }
            log(f"Deployment finished. Pages URL: {pages_url}")

            # Notify evaluation server if provided
            if payload.evaluation_url:
                try:
                    requests.post(
                        payload.evaluation_url,
                        headers={"Content-Type": "application/json"},
                        json={
                            "email": payload.email,
                            "task": payload.task,
                            "round": payload.round,
                            "nonce": payload.nonce,
                            "repo_url": repo_url,
                            "commit_sha": commit_sha,
                            "pages_url": pages_url,
                        },
                        timeout=600
                    )
                    log("✅ Evaluation server notified successfully")
                except Exception as e:
                    log(f"❌ Failed to notify evaluation server: {e}")

        except Exception as e:
            log(f"Deployment failed: {e}")
            deployment_results[payload.nonce] = {"error": str(e)}

    repo_url, commit_sha, pages_url = deploy_to_github(workdir, payload)

    # Notify evaluation server immediately
    if payload.evaluation_url:
        requests.post(
            payload.evaluation_url,
            headers={"Content-Type": "application/json"},
            json={
                "email": payload.email,
                "task": payload.task,
                "round": payload.round,
                "nonce": payload.nonce,
                "repo_url": repo_url,
                "commit_sha": commit_sha,
                "pages_url": pages_url,
            },
            timeout=600
        )
    return {
        "status": "success",
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "pages_url": pages_url
    }


@app.get("/api/status/{nonce}", response_model=StatusResponse)
async def get_deploy_status(nonce: str):
    if nonce in deployment_results:
        result = deployment_results[nonce]
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"status": "success", **result}
    return {"status": "pending", "repo_url": None, "commit_sha": None, "pages_url": None}



@app.get("/api/validate-secret")
async def validate_secret(secret: str):
    if secret == STUDENT_SECRET:
        return {"valid": True}
    raise HTTPException(status_code=403, detail="Invalid secret")


@app.get("/site/{nonce}/{filename}")
async def serve_site_file(nonce: str, filename: str):
    file_path = Path(f"workdir_{nonce}/site") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.get("/attachments/{nonce}/{filename}")
async def serve_attachment_file(nonce: str, filename: str):
    file_path = Path(f"workdir_{nonce}/attachments") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

import threading

  # Add this at the top of your file if not already imported

@app.post("/api/request-round2", response_model=StatusResponse)
async def handle_round2(payload: TaskRequest):
    if payload.secret != STUDENT_SECRET:
        raise HTTPException(status_code=400, detail="Invalid student secret")

    log(f"✅ Round 2 task received: {payload.task}")
    workdir = Path(f"workdir_{payload.nonce}")
    workdir.mkdir(exist_ok=True)

    # Step 1: Save attachments
    if payload.attachments:
        save_attachments(payload.attachments, workdir)

    # Step 2: Generate/update files
    llm_ok = generate_with_llm(payload, workdir)
    if not llm_ok:
        template_id = detect_template(payload.brief)
        generate_static_app(template_id, payload, workdir)

    # Step 3: Update README.md
    readme_file = workdir / "README.md"
    readme_content = f"# Task {payload.task} - Round 2\n\n"
    readme_content += f"Updated features based on brief:\n{payload.brief}\n"
    readme_file.write_text(readme_content, encoding="utf-8")
    log("📝 README.md updated for Round 2")

    # Step 4: Background deployment
    def _deploy():
        try:
            repo_name = f"llm-task-{payload.task}"
            remote_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{repo_name}.git"

            # --- Git setup ---
            if not (workdir / ".git").exists():
                run_shell("git init", cwd=workdir)
                run_shell("git branch -M main", cwd=workdir)

            run_shell(f'git config user.name "{GITHUB_USERNAME}"', cwd=workdir)
            run_shell('git config user.email "actions@github.com"', cwd=workdir)

            remotes = run_shell("git remote", cwd=workdir).splitlines()
            if "origin" not in remotes:
                run_shell(f"git remote add origin {remote_url}", cwd=workdir)
            else:
                run_shell(f"git remote set-url origin {remote_url}", cwd=workdir)

            # --- Fetch & merge existing repo ---
            try:
                run_shell("git fetch origin main", cwd=workdir)
                run_shell("git merge origin/main --allow-unrelated-histories -m 'Round 2 merge'", cwd=workdir)
                log("✅ Merge with remote successful")
            except Exception as merge_exc:
                log(f"⚠️ Merge failed: {merge_exc}. Proceeding with local changes only.")

            # --- Commit changes ---
            status = run_shell("git status --porcelain", cwd=workdir)
            if status.strip():
                run_shell('git add .', cwd=workdir)
                run_shell('git commit -m "Round 2 updates"', cwd=workdir)
                log("✅ Round 2 changes committed")
            else:
                log("⚠️ No changes detected to commit for Round 2")

            # --- Push to GitHub ---
            run_shell("git push origin main", cwd=workdir)
            log("✅ Changes pushed to GitHub (Round 2)")

            # --- GitHub Pages redeploy ---
            pages_payload = {"source": {"branch": "main", "path": "/"}}
            pages_file = workdir / "pages.json"
            pages_file.write_text(json.dumps(pages_payload))
            run_shell(
                f'gh api -X PUT repos/{GITHUB_USERNAME}/{repo_name}/pages '
                f'-H "Accept: application/vnd.github+json" --input {pages_file.absolute()}',
                cwd=workdir
            )
            log("🌐 GitHub Pages redeployed successfully")

            # --- Notify evaluation server ---
            try:
                resp = requests.post(
                    payload.evaluation_url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "email": payload.email,
                        "task": payload.task,
                        "round": 2,
                        "nonce": payload.nonce,
                        "repo_url": f"https://github.com/{GITHUB_USERNAME}/llm-task-{payload.task}",
                        "commit_sha": run_shell("git rev-parse HEAD", cwd=workdir),
                        "pages_url": f"https://{GITHUB_USERNAME}.github.io/llm-task-{payload.task}/"
                    },
                    timeout=600
                )
                if resp.status_code == 200:
                    log("✅ Evaluation server notified successfully for Round 2")
                else:
                    log(f"⚠️ Evaluation server responded: {resp.status_code}")
            except Exception as e:
                log(f"❌ Failed to notify evaluation server: {e}")

        except Exception as e:
            log(f"❌ Round 2 deployment failed: {e}")

    # Run deployment in background thread
    threading.Thread(target=_deploy, daemon=True).start()
    return {"status": "success"}



@app.post("/api/evaluate")
async def evaluate(payload: dict):
    evaluation_url = payload.get("evaluation_url")
    if not evaluation_url:
        raise HTTPException(status_code=400, detail="evaluation_url missing")

    try:
        data = {
            "email": payload.get("email"),
            "task": payload.get("task"),
            "round": payload.get("round", 2),
            "nonce": payload.get("nonce"),
            "repo_url": payload.get("repo_url"),
            "commit_sha": payload.get("commit_sha"),
            "pages_url": payload.get("pages_url")
        }
        resp = requests.post(
            evaluation_url,
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=600
        )
        if resp.status_code == 200:
            log("✅ Evaluation server notified successfully")
        else:
            log(f"⚠️ Evaluation server responded: {resp.status_code}")
    except Exception as e:
        log(f"❌ Failed to notify evaluation server: {e}")

    return {"status": "ok"}
