"""Doctest configuration.

Every ``Examples`` block in a public docstring, and every ``python``/``pycon``
block in ``README.md`` and ``docs/**``, is executed by
`Sybil <https://sybil.readthedocs.io/>`_. Doctests are load-bearing tests here,
not decoration -- ``-p no:doctest`` in ``pyproject.toml`` hands the job to Sybil
rather than to stdlib doctest.
"""

from collections.abc import Callable, Iterable, Sequence
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

from sybil import Document, Region, Sybil
from sybil.evaluators.doctest import DocTestEvaluator
from sybil.evaluators.python import PythonEvaluator
from sybil.parsers import myst
from sybil.parsers.abstract.codeblock import PythonDocTestOrCodeBlockParser
from sybil.parsers.abstract.doctest import DocTestStringParser

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE

# Hypothesis' 200ms per-example deadline is meaningless for tests that call into
# jax: an example that trips a fresh trace/compile, or just lands on a busy CPU
# during a full-suite run, blows past it. The deadline failure then does not
# reproduce on replay, so it surfaces as an unactionable `FlakyFailure`. Turn it
# off globally; per-test `@settings` still override this.
try:
    import hypothesis
except ImportError:
    pass
else:
    hypothesis.settings.register_profile("spexial", deadline=None)
    hypothesis.settings.load_profile("spexial")


class PlainDocTestParser:
    """Parser for plain >>> doctests in Python docstrings."""

    def __init__(self, doctest_optionflags: int = 0) -> None:
        """Initialize parser state."""
        self.doctest_parser = DocTestStringParser(DocTestEvaluator(doctest_optionflags))

    def __call__(self, document: Document) -> Iterable[Region]:
        """Parse plain doctest prompts from Python docstring text."""
        yield from self.doctest_parser(document.text, document.path)


class PyconCodeBlockParser(PythonDocTestOrCodeBlockParser):
    """Parser for MyST pycon code blocks with doctest evaluation."""

    def __init__(
        self,
        future_imports: Sequence[str] = (),
        doctest_optionflags: int = 0,
    ) -> None:
        """Initialize parser state."""
        self.doctest_parser = DocTestStringParser(DocTestEvaluator(doctest_optionflags))
        self.codeblock_parser = myst.CodeBlockParser(
            language="pycon",
            evaluator=PythonEvaluator(future_imports),
        )


markdown_parsers: Sequence[Callable[[Document], Iterable[Region]]] = [
    PyconCodeBlockParser(doctest_optionflags=optionflags),
    myst.DocTestDirectiveParser(optionflags=optionflags),
    myst.PythonCodeBlockParser(doctest_optionflags=optionflags),
    myst.SkipParser(),
]

docs = Sybil(parsers=markdown_parsers, patterns=["*.md"])
python = Sybil(
    parsers=[
        myst.SkipParser(),
        myst.PythonCodeBlockParser(doctest_optionflags=optionflags),
        PlainDocTestParser(doctest_optionflags=optionflags),
    ],
    patterns=["*.py"],
)

pytest_collect_file = (docs + python).pytest()
