import os
import json
import re
import subprocess
import time
import uuid
import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Tuple
from openai import OpenAI
from tqdm import tqdm

# =================== User Configuration ===================
ATTEMPT_SPEC = "1-5"  # e.g., "1", "2-4", "1,3,5"
BASE_INPUT_ROOT = Path("/path/to/code_generation_outputs")      # Change to your input dir
BASE_OUTPUT_ROOT = Path("/path/to/validation_outputs")          # Change to your output dir
API_KEY = "YOUR_OPENAI_API_KEY"
TARGET_MODELS = {
    "baseline": "gpt-4o",
    "framework": "ft:gpt-4o-2024-08-06:your-finetune-id",
    "expert":    "ft:gpt-4o-2024-08-06:your-finetune-id"
}
MAX_ATTEMPTS = 5  # Max validation/refinement cycles per file

client = OpenAI(api_key=API_KEY)

# =================== Utility: Docstring ===================
def extract_module_docstring(code: str) -> str:
    for m in re.finditer(r'("""|\'\'\')([\s\S]*?)\1', code):
        ds = m.group(0)
        if all(k in ds for k in ("Input Prompt", "Intention", "Functionality")):
            return ds
    return '"""\nInput Prompt:\nIntention:\nFunctionality:\n"""'

def strip_module_docstring(code: str) -> str:
    for m in re.finditer(r'("""|\'\'\')([\s\S]*?)\1', code):
        ds = m.group(0)
        if all(k in ds for k in ("Input Prompt", "Intention", "Functionality")):
            return (code[:m.start()] + code[m.end():]).lstrip()
    return code.lstrip()

def ensure_module_docstring(body: str, doc: str) -> str:
    return doc.strip() + "\n\n" + body.lstrip()

def split_docstring_and_code(code: str) -> Tuple[str, str]:
    return extract_module_docstring(code).strip(), strip_module_docstring(code).lstrip()

# =================== Utility: Code & Tags ===================
def extract_tag_section(text: str, tag: str) -> str | None:
    m = re.search(fr"<{tag}>([\s\S]*?)</{tag}>", text, re.IGNORECASE)
    return m.group(1).strip() if m else None

def strip_code_fence(txt: str) -> str:
    lines = txt.strip().splitlines()
    if lines and lines[0].startswith("```"): lines = lines[1:]
    if lines and lines[-1].startswith("```"): lines = lines[:-1]
    return "\n".join(lines).strip()

def extract_pure_code(content: str) -> str:
    tagged = extract_tag_section(content, "Code")
    if tagged:
        return tagged
    start, end = content.find("```python"), content.rfind("```")
    if start != -1 and end != -1 and end > start:
        return content[start+10:end].strip()
    return strip_code_fence(content)

# =================== SAST Runners ===================
def run_bandit(path: Path) -> Dict[str, Any]:
    out = subprocess.run(["bandit", "-f", "json", "-q", str(path)],
                         capture_output=True, text=True, check=False).stdout
    return json.loads(out or '{"results": []}')

def run_pylint(path: Path) -> List[Dict[str, Any]]:
    out = subprocess.run(
        ["pylint", "-f", "json", "--disable=all", "--enable=E0001", str(path)],
        capture_output=True, text=True, check=False
    ).stdout
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return []

# =================== OpenAI Wrapper ===================
def oai(model: str, prompt: str, retries: int = 4) -> str:
    wait = 4.0
    for _ in range(retries):
        try:
            return client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content.strip()
        except Exception:
            if _ == retries - 1:
                raise
            time.sleep(wait)
            wait *= 2

# =================== Functional Validation ===================
def functional_check(full_code: str, model: str) -> Tuple[str, str, str]:
    doc, body = split_docstring_and_code(full_code)
    prompt = f"""
<Instruction>
Follow these rules **exactly** when you answer.
1. Determine whether the implementation in <Code> mostly satisfies the requirements described in <Docstring>,
   including cases where it uses a different library or algorithm that achieves the same observable behaviour
   while improving security or performance. Compare results, not the exact API calls.
2. If the implementation meets the intention (identical behaviour) — even if it swaps to a more secure or faster library — output one word only: Correct
3. If any change is needed, output in *precisely* this structure—nothing more, nothing less:
Incorrect
<Code>
# (full corrected code body here — do NOT include the docstring)
</Code>
<Reason>
# (brief explanation of what you fixed and why)
</Reason>
Formatting constraints
• Start the corrected code block with a line that contains only <Code>
  and end it with a line that contains only </Code>.
• Start the reason block with <Reason> and end it with </Reason>.
• Do NOT add Markdown fences (```), line numbers, extra headers, or footers.
• Never duplicate, delete, or modify the original docstring; keep comments intact.
</Instruction>
<Docstring>
{doc}
</Docstring>
<Code>
{body}
</Code>
""".strip()
    resp = oai(model, prompt)
    if resp.strip() == "Correct":
        return "Correct", prompt, resp
    fixed_body = strip_module_docstring(extract_pure_code(resp))
    return fixed_body, prompt, resp

# =================== Main File Validation ===================
def validate_file(args: Tuple[Path, str, str, str, Path]):
    fpath, fname, mname, mid, out_base = args
    m_out = out_base / mname
    m_out.mkdir(parents=True, exist_ok=True)
    final_dir = m_out / "Final_code"
    final_dir.mkdir(parents=True, exist_ok=True)

    original = fpath.read_text(encoding="utf-8")
    latest_doc = extract_module_docstring(original)
    code_body = strip_module_docstring(original)
    revisions = []

    for att in range(1, MAX_ATTEMPTS + 1):
        att_dir = m_out / f"attempt_{att}"
        att_dir.mkdir(exist_ok=True)
        code_full = ensure_module_docstring(code_body, latest_doc)
        tmp = att_dir / f"tmp_{fname}"
        tmp.write_text(code_full, encoding="utf-8")

        pylint_iss = [{"line": i["line"], "message": i["message"]} for i in run_pylint(tmp)]
        bandit_iss = run_bandit(tmp).get("results", [])
        sast_fixed = False

        if pylint_iss or bandit_iss:
            issues_summary = (
                "Pylint Issues:\n" + ("\n".join(i["message"] for i in pylint_iss) or "None") +
                "\n\nBandit Issues:\n" + ("\n".join(i["issue_text"] for i in bandit_iss) or "None")
            )
            sast_prompt = f"""
The static-analysis tools below reported problems in the implementation.
<Code>
{code_body}
</Code>
<PylintIssues>
{'\n'.join(i['message'] for i in pylint_iss) if pylint_iss else 'None'}
</PylintIssues>
<BanditIssues>
{'\n'.join(i['issue_text'] for i in bandit_iss) if bandit_iss else 'None'}
</BanditIssues>
Instructions
• Fix **every** issue listed above.
• Keep all existing docstrings and comments exactly as they are.
• Reply with *only* the corrected implementation, enclosed in a single code block:
<Code>
… full fixed code body (docstring excluded) …
</Code>
Formatting rules
• The first line of your answer must be “<Code>”.
• The last line must be “</Code>”.
• Do **not** output anything else—no explanations, no Markdown fences, no extra text.
""".strip()
            sast_resp = oai(mid, sast_prompt)
            if sast_resp.strip() != "Correct":
                code_body = strip_module_docstring(extract_pure_code(sast_resp))
            sast_fixed = True
        else:
            sast_prompt, sast_resp = "No SAST issues found.", ""

        (att_dir / f"{fname[:-3]}_attempt_{att}.py").write_text(
            ensure_module_docstring(code_body, latest_doc), encoding="utf-8"
        )
        tmp.unlink(missing_ok=True)

        if sast_fixed and (pylint_iss or bandit_iss):
            revisions.append({
                "attempt": att, "pylint_issues": pylint_iss, "bandit_issues": bandit_iss,
                "sast_prompt": sast_prompt, "sast_response": sast_resp,
                "functional_prompt": "", "functional_response": "", "status": "SAST-Fixed"
            })
            continue

        func_res, func_prompt, func_resp = functional_check(
            ensure_module_docstring(code_body, latest_doc), mid
        )
        if func_res == "Correct":
            status = "Correct"
        else:
            code_body = strip_module_docstring(func_res)
            status = "Incorrect"

        revisions.append({
            "attempt": att, "pylint_issues": pylint_iss, "bandit_issues": bandit_iss,
            "sast_prompt": sast_prompt, "sast_response": sast_resp,
            "functional_prompt": func_prompt, "functional_response": func_resp,
            "status": status
        })

        if status == "Correct":
            (final_dir / fname).write_text(ensure_module_docstring(code_body, latest_doc), encoding="utf-8")
            break

    (m_out / f"{fname[:-3]}_validation_results.json").write_text(
        json.dumps({"filename": fname, "revisions": revisions}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if revisions[-1]["status"] != "Correct":
        tmp_unres = m_out / f"unresolved_tmp_{os.getpid()}_{uuid.uuid4()}.json"
        tmp_unres.write_text(json.dumps({"filename": fname, "final_status": revisions[-1]["status"]},
                                        ensure_ascii=False), encoding="utf-8")
        (final_dir / fname).write_text(ensure_module_docstring(code_body, latest_doc), encoding="utf-8")

    return {"filename": fname, "revisions": revisions}

# =================== Attempt Spec Parser ===================
def parse_attempts(spec: str) -> List[int]:
    parts, out = spec.split(","), set()
    for p in parts:
        p = p.strip()
        if "-" in p:
            a, b = map(int, p.split("-"))
            out.update(range(a, b + 1))
        else:
            out.add(int(p))
    return sorted(out)

# =================== Main Loop ===================
if __name__ == "__main__":
    for ext_att in parse_attempts(ATTEMPT_SPEC):
        in_root = BASE_INPUT_ROOT / f"Attempt {ext_att}"
        out_root = BASE_OUTPUT_ROOT / f"Attempt {ext_att}"
        if not in_root.exists():
            print(f"[WARN] Input Attempt folder not found: {in_root}")
            continue

        print(f"\n==================== Attempt {ext_att} ====================")
        for mname, mid in TARGET_MODELS.items():
            in_dir = in_root / mname / "code"
            if not in_dir.exists():
                print(f"[WARN] Input folder not found: {in_dir}")
                continue

            print(f"\nValidating model: {mname}")
            tasks = [
                (p, p.name, mname, mid, out_root)
                for p in in_dir.iterdir()
                if p.suffix == ".py" and "attempt" not in p.name
            ]

            max_workers = min(cpu_count(), len(tasks))
            with Pool(processes=max_workers) as pool:
                results = list(tqdm(pool.imap(validate_file, tasks), total=len(tasks)))

            (out_root / mname / "all_result.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            pattern = str((out_root / mname) / "unresolved_tmp_*.json")
            unresolved = []
            for fp in glob.glob(pattern):
                with open(fp, encoding="utf-8") as f:
                    try:
                        unresolved.append(json.load(f))
                    except Exception:
                        pass
                os.remove(fp)
            if unresolved:
                with open(out_root / mname / "unresolved.json", "w", encoding="utf-8") as f:
                    json.dump(unresolved, f, indent=2, ensure_ascii=False)

            print(f"Validation done: {mname} — {len(results)} files")

