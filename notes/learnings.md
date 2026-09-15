Different ways to run Python files:
- python3 -m src.evaluator this runs the file within a module
- python3 src/evaluator.py this runs it with src as the root, instead of root as the actual root.

Python Modules & Packages

When a project uses package-style imports such as:

    from src.classifier import classify_email
    from src.models import ClassificationOutput

run the module from the PROJECT ROOT:

    python3 -m src.evaluator

Not:

    python3 src/evaluator.py

Why:
- `python3 src/evaluator.py` executes the file as a standalone script and puts `src/` on the import path.
- `python3 -m src.evaluator` executes `evaluator.py` as the `src.evaluator` module and runs it with the project root on the import path.
- This allows Python to correctly resolve imports such as `src.classifier` and `src.models`.

Key distinction:
    python3 src/evaluator.py
        → "run this file"

    python3 -m src.evaluator
        → "run this module as part of the package"

Production Python projects generally favor package/module execution (`python -m ...`) and proper package structure over manually modifying `sys.path`.

# OpenAI Response Output Example
```python
ParsedChatCompletion[TypeVar](id='chatcmpl-EF3OBFuueE0FRHchk8reM4gOYVjAB', choices=[ParsedChoice[TypeVar](finish_reason='stop', index=0, logprobs=None, message=ParsedChatCompletionMessage[TypeVar](content='{"relevance_score":4}', refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None, parsed=ScoredSummaryRelevance(relevance_score=4)))], created=1787256655, model='gpt-4o-mini-2024-07-18', object='chat.completion', metadata=None, moderation=None, service_tier='default', system_fingerprint='fp_c881474fd1', usage=CompletionUsage(completion_tokens=7, prompt_tokens=209, total_tokens=216, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0, text_tokens=None), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cache_write_tokens=None, cached_tokens=0, image_tokens=None, text_tokens=None)))
```