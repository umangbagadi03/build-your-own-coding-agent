# Contributing

Thanks for helping. Readers learn from every line here, so a small fix to a confusing sentence is as welcome as a new feature.

## Good ways to help

| Contribution | How to start |
|---|---|
| Fix a typo, a wrong claim or a confusing explanation | Open a pull request |
| Report a bug in a notebook | Open a **Lesson problem** issue |
| Share results from a model or provider | Open a **Model or provider report** issue |
| Improve an exercise | Open a pull request |
| Add a new notebook, or a translation | Open an issue first, so nobody's work gets wasted |
| Change the course structure or `llm_setup.py` behaviour | Open an issue first |

## Setup

```bash
git clone https://github.com/<your-username>/build-your-own-coding-agent.git
cd build-your-own-coding-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

## Making a change

1. Create a branch and edit the notebook in Jupyter, in VS Code or on Colab.
2. Run everything from top to bottom in offline mode. This needs no key:

   ```bash
   python scripts/run_notebooks.py
   ```

3. Save fresh outputs, so the notebook displays correctly on GitHub:

   ```bash
   python scripts/run_notebooks.py --write
   ```

4. Commit only the notebooks you meant to change. Step 3 updates every notebook, and timings differ on each run, so restore the others:

   ```bash
   git restore notebooks/<a-notebook-you-did-not-change>.ipynb
   ```

5. Open a pull request. Keep it to one topic, because notebook diffs are hard to review. In your description, say what readers will see differently.

## Rules for lessons

These keep the course coherent as it grows.

- **Build it by hand.** No agent frameworks or LLM SDKs. If a concept can be written in plain Python and `requests`, write it that way.
- **Every notebook must run with no key.** Offline mode is what makes the course free for everyone and testable in CI.
- **One new idea at a time.** A notebook adds one feature to the agent and states what it will teach at the top.
- **Each notebook stands alone.** It can only use `llm_setup.py` and code it defines itself, because Colab opens one notebook at a time. Repeat short helpers rather than importing them from another notebook.
- **Show evidence.** Prefer numbers you measured over adjectives. If you state a result from a real run, name the model and say it was one run.
- **Be honest about limits.** Say when something is simulated, untested or unsafe.

### Offline mode and checkpoints

Offline mode isn't a model. It only knows the prompts in `_script()` in [`notebooks/llm_setup.py`](notebooks/llm_setup.py), so **a new prompt or tool usually needs a new scripted reply there**.

Add checkpoint cells so a broken lesson fails CI:

```python
# Real models vary, so only check their exact wording in offline mode.
if llm.offline:
    assert "Asha" in text_of(reply)

# Your own code should behave the same everywhere, so always check it.
assert run_tool_call(cases["valid call"]) == "42"
```

A notebook has to finish on a real model too, even if the model gives a poor answer. If you can, run it once with `LLM_PROVIDER=ollama` or another real provider, and mention the result in your pull request.

## Safety rules

Learners will run this code on their own machines without reading it closely, so these aren't optional.

- **Keep the limits in code.** A system prompt is never a security control. Restrict what a tool can do in the tool itself.
- **Give agents only a temporary workspace.** Don't let a lesson act on a learner's real files or home folder without a clear warning and a confirmation step.
- **Call out risk.** If a lesson runs code or commands a model chose, say so plainly, right above that cell.
- **Keep secrets out.** Tools must not pass environment variables to code the model writes, and no cell should print a key.

## Don't commit

- API keys or `.env` files. If you commit a key by accident, revoke it right away, since deleting it from the repo doesn't make it safe.
- Local paths or personal information in outputs. `scripts/run_notebooks.py` checks for common patterns but can't catch everything, so look through your diff.
- Large outputs. Trim long output in the cell rather than saving it.

## AI-assisted contributions

Using AI tools is fine. You're still responsible for the result, so please read it, run it and be able to explain it. Unverified claims and code that was never run are the most likely reasons for a pull request to be declined.

## Reporting a security problem

If a lesson's safety measures can be bypassed, such as a path escape in the file tools, please open an issue describing what is affected. If describing it would put learners at risk, ask for a private way to share the details instead of posting them.

## Being respectful

Many people here are learning this for the first time. Be kind, assume good intent, and critique the work rather than the person. Harassment isn't welcome, and maintainers may remove comments or block accounts to keep this a good place to learn.

## License

By contributing, you agree that your contribution is licensed under the [MIT License](LICENSE).
