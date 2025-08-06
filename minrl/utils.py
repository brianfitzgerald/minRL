import re


def clean_observation(obs: str) -> str:
    obs = obs.strip()
    lines = obs.split("\n")
    processed_lines = []
    for line in lines:
        # Rule 1: If a line starts with '^', discard this line and all subsequent lines
        if line.lstrip().startswith("^") or line.lstrip().startswith(">"):
            break

        lstripped_line = line.lstrip()

        # Filter for any line starting with '>' or containing "Purpose:"
        if lstripped_line.startswith(">") or "Purpose:" in lstripped_line:
            continue

        cleaned_line = re.sub(r"Score: \d+ Moves: \d+", "", line).strip()
        processed_lines.append(cleaned_line)

    return "\n".join(processed_lines).strip()
