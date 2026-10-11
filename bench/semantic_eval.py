"""Measure the semantic cache's hit and false-hit rates on real help desk tickets.

    python -m bench.semantic_eval ../Ai-Help-Desk/evals/cases.json

For each ticket, Claude Haiku writes a paraphrase: the same person asking for the
same thing in different words. Every original ticket is "cached"; each paraphrase
is then looked up the way the gateway would (closest cached ticket by cosine
similarity of bge-small embeddings). At each threshold:

  correct hit  the closest match is the paraphrase's own original (reusing its answer is right)
  false hit    the closest match is a different ticket (the user gets someone else's answer)
  miss         nothing clears the threshold (the request goes to the model, which is always safe)

It also re-sends each ticket unchanged from a different sender. For a help desk that is a
different person's account, so any hit is a false hit.

Paraphrases are saved to bench/data/paraphrases.json and reused on later runs.
"""

import json
import sys
from pathlib import Path

import anthropic
import numpy as np

from gateway.config import settings
from gateway.semanticcache import Embedder

DATA = Path(__file__).parent / "data" / "paraphrases.json"
THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]

PROMPT = """Rewrite this IT help desk ticket the way the same person might have written it on a different day:
same request, same details and names, different wording and sentence structure. Keep the subject line.
Reply with exactly two parts: a line starting "Subject: " and then the body. No commentary.

Subject: {subject}

{body}"""


def ticket_text(sender: str, subject: str, body: str) -> str:
    return f"From: {sender}\nSubject: {subject}\n\n{body}"


def paraphrases(cases: list[dict]) -> dict[str, dict]:
    cached = json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else {}
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    for case in cases:
        if case["id"] in cached:
            continue
        message = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=2000,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": PROMPT.format(subject=case["subject"], body=case["body"])}],
        )
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        subject, _, body = text.partition("\n")
        cached[case["id"]] = {"subject": subject.removeprefix("Subject:").strip(), "body": body.strip()}
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")
    return cached


def main(cases_path: str) -> None:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    para = paraphrases(cases)
    embedder = Embedder()
    embed = lambda texts: np.stack([embedder._embed_sync(t) for t in texts])  # noqa: E731

    originals = embed([ticket_text(c["sender"], c["subject"], c["body"]) for c in cases])
    queries = embed([ticket_text(c["sender"], para[c["id"]]["subject"], para[c["id"]]["body"]) for c in cases])
    other_sender = [c["sender"].replace(c["sender"].split("@")[0], "someone.else") for c in cases]
    swapped = embed([ticket_text(s, c["subject"], c["body"]) for s, c in zip(other_sender, cases)])

    sims = queries @ originals.T  # [paraphrase i, original j]
    best = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    own_sim = np.diag(sims)
    swap_sim = np.einsum("ij,ij->i", swapped, originals)
    n = len(cases)

    print(f"{n} tickets. Paraphrase similarity to own original: median {np.median(own_sim):.3f}, "
          f"min {own_sim.min():.3f}. Same ticket from another sender: median {np.median(swap_sim):.3f}, "
          f"min {swap_sim.min():.3f}.\n")
    print("| Threshold | Correct hits | False hits | Misses | Sender-swap false hits |")
    print("| --- | --- | --- | --- | --- |")
    for t in THRESHOLDS:
        hit = best_sim >= t
        correct = int(np.sum(hit & (best == np.arange(n))))
        false = int(np.sum(hit & (best != np.arange(n))))
        swap = int(np.sum(swap_sim >= t))
        print(f"| {t:.2f} | {correct}/{n} ({correct / n:.0%}) | {false}/{n} ({false / n:.0%}) | "
              f"{n - correct - false}/{n} | {swap}/{n} ({swap / n:.0%}) |")

    print("\nFalse hits at 0.90 (paraphrase -> wrong ticket, similarity):")
    for i in np.where((best_sim >= 0.90) & (best != np.arange(n)))[0]:
        print(f"  {cases[i]['id']} -> {cases[best[i]]['id']} ({best_sim[i]:.3f})")


if __name__ == "__main__":
    main(sys.argv[1])
