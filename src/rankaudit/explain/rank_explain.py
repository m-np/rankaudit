"""
RankExplain — GenAI layer that translates audit reports into plain English.

Calls any OpenAI-compatible chat API (including Claude via the Anthropic SDK)
to produce natural-language explanations of audit findings.
"""

from __future__ import annotations

from typing import Any

from ..core.report import AuditReport


_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert in information retrieval and ranking systems. "
    "You receive a structured audit report and produce clear, concise, "
    "non-technical explanations suitable for product managers, legal teams, "
    "or domain experts who are not machine learning engineers."
)


class RankExplain:
    """
    Natural-language explanation layer for AuditReport objects.

    Parameters
    ----------
    client:
        An LLM client.  Must have a `chat.completions.create` method
        (OpenAI SDK style) OR a `messages.create` method (Anthropic SDK style).
    model:
        Model identifier to call.  Defaults to "claude-sonnet-4-6".
    system_prompt:
        Override the default system prompt.
    max_tokens:
        Maximum tokens in the LLM response.
    """

    def __init__(
        self,
        client: Any,
        model: str = "claude-sonnet-4-6",
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def explain_report(self, report: AuditReport) -> str:
        """Return a plain-English summary of the entire audit report."""
        prompt = self._build_report_prompt(report)
        return self._call_llm(prompt)

    def explain_doc(self, report: AuditReport, doc_id: str, query_id: str | None = None) -> str:
        """Return a plain-English explanation for a single document's ranking."""
        structured = report.explain(doc_id=doc_id, query_id=query_id)
        prompt = (
            f"Here is a structured ranking explanation for document '{doc_id}':\n\n"
            f"{structured}\n\n"
            "Please translate this into a clear, non-technical explanation of "
            "why this document ranked where it did and what factors helped or hurt it."
        )
        return self._call_llm(prompt)

    def explain_counterfactuals(self, report: AuditReport, query_id: str | None = None) -> str:
        """Return a plain-English explanation of the counterfactual findings."""
        cfs = report.counterfactuals
        if query_id:
            cfs = [c for c in cfs if c.query_id == query_id]
        if not cfs:
            return "No counterfactual results found."

        lines = []
        for cf in cfs:
            changes = "; ".join(
                f"{feat}: {old:.3f} → {new:.3f}"
                for feat, (old, new) in cf.flipping_changes.items()
            )
            lines.append(
                f"Query '{cf.query_id}': doc '{cf.doc_b_id}' (rank {cf.original_rank_b}) "
                f"would overtake doc '{cf.doc_a_id}' (rank {cf.original_rank_a}) "
                f"if these features changed: {changes}"
            )

        prompt = (
            "Here are counterfactual ranking flip examples:\n\n"
            + "\n".join(lines)
            + "\n\nPlease explain in plain English what these mean for the ranking system "
            "and what practical actions could be taken."
        )
        return self._call_llm(prompt)

    def explain_bias(self, report: AuditReport) -> str:
        """Return a plain-English explanation of bias findings."""
        bias_items = report.bias
        if not bias_items:
            return "No bias analysis results found."

        lines = []
        for b in bias_items:
            lines.append(f"\nQuery '{b.query_id}':")
            if b.position_bias is not None:
                lines.append(f"  Position-relevance correlation: {b.position_bias:.3f}")
            if b.exposure_bias:
                for g, exp in b.exposure_bias.items():
                    lines.append(f"  Group '{g}' average exposure: {exp:.4f}")
            if b.demographic_parity:
                for g, frac in b.demographic_parity.items():
                    lines.append(f"  Group '{g}' share of top-k: {frac:.2%}")
            if b.notes:
                for note in b.notes:
                    lines.append(f"  [!] {note}")

        prompt = (
            "Here is a bias analysis of a ranking system:\n"
            + "\n".join(lines)
            + "\n\nPlease explain in plain English what these numbers mean, "
            "which findings are concerning, and what remediation steps are recommended."
        )
        return self._call_llm(prompt)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_report_prompt(self, report: AuditReport) -> str:
        return (
            "Here is a ranking audit report summary:\n\n"
            f"{report.summary()}\n\n"
            "Please write a clear, non-technical explanation of these findings, "
            "highlighting the most important quality and fairness issues, "
            "and suggesting actionable next steps for the engineering team."
        )

    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM client — supports both OpenAI and Anthropic SDK styles."""
        # Anthropic SDK: client.messages.create(...)
        if hasattr(self.client, "messages") and hasattr(self.client.messages, "create"):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        # OpenAI SDK: client.chat.completions.create(...)
        if hasattr(self.client, "chat"):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content

        raise TypeError(
            "Unsupported client type. Provide an Anthropic or OpenAI-compatible client."
        )