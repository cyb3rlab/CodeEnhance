import os
import json
import pathlib
import concurrent.futures
import re
from openai import OpenAI
from tqdm import tqdm

# ------------------ Configuration ------------------
BASE_DIR = str(pathlib.Path.home() / "Downloads")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")   # Change as needed

MODEL_INFO = {
    "baseline": {
        "model": "gpt-4o",  # Change to your model name or OpenAI fine-tuned model ID
        "dataset": DATASET_PATH,
    },
    # "another_model": { ... }  # Add more models as needed
}

API_KEY = "YOUR_OPENAI_API_KEY"
client = OpenAI(api_key=API_KEY)

SAMPLE_LIMIT = None
MAX_WORKERS = 10
MAX_RETRIES = 5

REQUIRED_SECTIONS = ["Input Prompt", "Intention", "Functionality"]

# ------------------ Dataset Loader ------------------
def read_dataset(path, limit=None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for i, item in enumerate(data):
        if limit and i >= limit:
            break
        samples.append({
            "id": item.get("id", f"sample_{i:04d}"),
            "prompt": item.get("prompt", ""),
        })
    return samples

# ------------------ Prompt Construction ------------------
def create_prompt(prompt: str) -> str:
    return (
        "Please write Python code for the following task. "
        "At the very top, add a triple-quoted docstring with these three sections, each starting on its own line:\n"
        "• **Input Prompt**: Restate the prompt clearly.\n"
        "• **Intention**: State the purpose of the code.\n"
        "• **Functionality**: Describe briefly how the code solves the task.\n\n"
        "Write only valid Python code and no extra explanations. Return the complete script only.\n\n"
        f"Prompt: {prompt}"
    )

_fence_pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_code_from_response(response: str) -> str:
    text = response.strip()
    code_blocks = _fence_pattern.findall(text)
    if code_blocks:
        return code_blocks[-1].strip()
    inline_match = re.search(r"((?:def |class ).*)", text, re.DOTALL)
    if inline_match:
        return inline_match.group(1).strip()
    return text

def clean_python_code(code: str) -> str:
    code = re.sub(r"^\s*```(?:python)?", "", code, flags=re.IGNORECASE).strip()
    code = re.sub(r"```$", "", code).strip()
    code = "\n".join(line.rstrip() for line in code.splitlines())
    try:
        import black
        code = black.format_str(code, mode=black.FileMode())
    except Exception:
        pass
    return code.rstrip() + "\n"

def has_required_docstring_sections(code: str) -> bool:
    return all(section in code for section in REQUIRED_SECTIONS)

def process_prompt(model_name, model_id, prompt, prompt_id, output_dir):
    model_folder = os.path.join(output_dir, model_name)
    code_dir = os.path.join(model_folder, "code")
    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)

    code_file = os.path.join(code_dir, f"{prompt_id}.py")
    dialogue_file = os.path.join(model_folder, f"{prompt_id}_dialogue.json")

    if os.path.exists(code_file) and os.path.exists(dialogue_file):
        try:
            with open(dialogue_file, encoding="utf-8") as f:
                dialogue_history = json.load(f)
        except Exception:
            dialogue_history = None
        return {"prompt_id": prompt_id, "status": "skipped", "dialogue": dialogue_history}

    messages = [
        {"role": "system", "content": "You are a code generation assistant."},
        {"role": "user", "content": create_prompt(prompt)},
    ]

    cleaned_code = ""
    pass_check = False
    retry_count = 0

    while retry_count <= MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=1024,
                temperature=0,
            )
            assistant_reply = response.choices[0].message.content.strip()
        except Exception as e:
            return {"prompt_id": prompt_id, "status": f"error: {e}", "dialogue": None}

        raw_code = extract_code_from_response(assistant_reply)
        cleaned_code = clean_python_code(raw_code)

        if has_required_docstring_sections(cleaned_code):
            pass_check = True
            messages.append({"role": "assistant", "content": assistant_reply})
            break

        retry_count += 1
        messages.append({"role": "assistant", "content": assistant_reply})

        if retry_count > MAX_RETRIES:
            break

        correction_prompt = (
            "The Python script you just provided does NOT start with the required "
            "top-level docstring.\n\n"
            "Please regenerate the ENTIRE script again and:\n"
            "- Put a triple-quoted docstring ( \"\"\" ... \"\"\" ) at the VERY TOP of the file.\n"
            "- Inside that docstring include the following three headings EXACTLY as written,\n"
            "  each on its own line, followed by their description:\n"
            "  **Input Prompt**\n"
            "  **Intention**\n"
            "  **Functionality**\n"
            "- After the docstring, write the rest of the valid Python code.\n\n"
            "Return ONLY one fenced code block in the form:\n"
            "```python\n"
            "(complete script)\n"
            "```\n"
            "Do NOT add explanations or text outside the code block."
        )
        messages.append({"role": "user", "content": correction_prompt})

    status = "success" if pass_check else "fail_docstring"

    with open(code_file, "w", encoding="utf-8") as f:
        f.write(cleaned_code)

    dialogue_history = [
        {"role": m["role"], "content": [{"text": m["content"], "type": "text"}]}
        for m in messages
    ]
    with open(dialogue_file, "w", encoding="utf-8") as f:
        json.dump(dialogue_history, f, indent=2, ensure_ascii=False)

    return {"prompt_id": prompt_id, "status": status, "dialogue": dialogue_history}

def run_model_eval(model_name, model_info, output_dir, summary_path, sample_limit=None, max_workers=10):
    dataset = read_dataset(model_info["dataset"], limit=sample_limit)
    results, all_dialogues, missing_ids = [], [], []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(
                process_prompt,
                model_name,
                model_info["model"],
                d["prompt"],
                d["id"],
                output_dir,
            ): d["id"]
            for d in dataset
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(future_to_id)):
            result = future.result()
            results.append({"prompt_id": result["prompt_id"], "status": result["status"]})
            if result["dialogue"] is not None:
                all_dialogues.append({"prompt_id": result["prompt_id"], "dialogue": result["dialogue"]})
            if result["status"] == "fail_docstring":
                missing_ids.append(result["prompt_id"])

    model_folder = os.path.join(output_dir, model_name)
    result_path = os.path.join(model_folder, "results.json")
    dialogue_path = os.path.join(model_folder, "all_dialogues.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(dialogue_path, "w", encoding="utf-8") as f:
        json.dump(all_dialogues, f, indent=2, ensure_ascii=False)

    missing_path = os.path.join(model_folder, "missing_docstrings.txt")
    with open(missing_path, "w", encoding="utf-8") as f:
        for pid in missing_ids:
            f.write(pid + "\n")

    if missing_ids:
        with open(summary_path, "a", encoding="utf-8") as f:
            for pid in missing_ids:
                f.write(f"{model_name}\t{pid}\n")

    print(f"[{model_name}] Results saved to {result_path}")
    print(f"[{model_name}] Dialogues saved to {dialogue_path}")
    if missing_ids:
        print(f"[{model_name}] {len(missing_ids)} missing docstrings → {missing_path}")

if __name__ == "__main__":
    summary_file = os.path.join(BASE_DIR, "missing_docstrings_summary.txt")
    if os.path.exists(summary_file):
        os.remove(summary_file)

    for model_name, model_info in MODEL_INFO.items():
        print(f"\n▶ Running model: {model_name} ({model_info['model']})")
        run_model_eval(
            model_name = model_name,
            model_info = model_info,
            output_dir = BASE_DIR,
            summary_path = summary_file,
            sample_limit = SAMPLE_LIMIT,
            max_workers = MAX_WORKERS,
        )

    print("\n=== All processing complete ===")
    if os.path.exists(summary_file):
        print(f"Missing docstrings for all models: {summary_file}")
    else:
        print("No missing docstring samples.")


