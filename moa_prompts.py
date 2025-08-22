"""
Zaawansowane prompty dla systemu MOA z technikami Chain-of-Thought i Self-Consistency
"""
from typing import Dict, Any, List
from config.models_config import AgentRole

class MOAPrompts:
    """Centralna biblioteka promptów dla systemu MOA"""
    
    # Uniwersalne zasady dla wszystkich agentów
    UNIVERSAL_PRINCIPLES = """
## UNIVERSAL REASONING & OUTPUT POLICY

1) Deterministic, Structured Reasoning
- Decompose the mission into atomic steps; make dependencies explicit.
- Prefer DAG-like flows with clear success/failure transitions.

2) Output Contract (STRICT)
- Final output MUST be a single valid JSON object (no prose, no code fences, no comments).
- Keys and schema names are in English; user-facing strings are in Polish.
- If you risk exceeding token limits, compress explanations but keep structure intact.

3) Memory & Retrieval Discipline
- When WRITING memory: always store concise English bullet points or JSON objects
  (normalized nouns, present tense, ≤200 tokens per write).
- When READING memory: query only what is needed for the current decision.
- Never copy large memory chunks into the output; summarize instead.

4) Robustness by Design
- For each critical step, state the expected preconditions and postconditions.
- Include failure transitions (on_failure) and remediation (retry, rollback, notify).

5) Metrics & Confidence
- Quantify uncertainty (0–1). Justify with observable signals (e.g., data_quality).
- Prefer measurable thresholds over vague conditions.

6) Tooling Constraints
- Use ONLY nodes present in the node library (exact implementation names).
- Allowed edge.condition values: on_success, on_failure, retry, validated, partial_success,
  needs_optimization, else (as a last-resort catch-all).
"""
    
    @staticmethod
    def get_proposer_prompt(role: AgentRole, mission: str, node_library: Dict) -> str:
        """English prompt for Proposers; user-facing strings must be Polish."""
        style_mod = {
            "analytical": "Be precise and data-driven; justify every decision with observable signals.",
            "creative": "Explore non-obvious combinations and alternative paths; propose at least one novel twist.",
            "critical": "Stress-test assumptions and highlight edge cases and single points of failure.",
            "systematic": "Aim for holistic, end-to-end coherence with explicit interfaces between steps."
        }

        expertise = f"""
    # ROLE: {role.role_name}

    ## YOUR EXPERTISE
    You specialize in: {', '.join(role.expertise_areas)}

    ## THINKING STYLE
    {style_mod.get(role.thinking_style, "Default to clarity and rigor.")}

    {MOAPrompts.UNIVERSAL_PRINCIPLES}

    ## ROLE-SPECIFIC TECHNIQUES
    """
        rl = role.role_name.lower()
        if "causal" in rl:
            expertise += """
    - Causal Reasoning:
      * Identify variables and likely causal relations (confounders, mediators).
      * Prefer testable interventions; annotate assumptions explicitly.
    """
        elif "strategic" in rl:
            expertise += """
    - Strategic Planning:
      * SWOT per component; map critical dependencies and critical path.
      * Prepare 1–2 realistic what-if branches with measurable triggers.
    """
        elif "creative" in rl:
            expertise += """
    - Creative Expansion:
      * Apply SCAMPER to at least two nodes.
      * Propose 3 alternative micro-approaches and pick one with rationale.
    """
        elif "risk" in rl or "quality" in rl:
            expertise += """
    - Risk/Quality:
      * FMEA table in your head; identify top 3 failure modes and mitigations.
      * Add explicit rollback/notify paths for irrecoverable states.
    """

        return f"""
    {expertise}

    ## MISSION
    {mission}

    ## AVAILABLE NODE LIBRARY
    {MOAPrompts._format_node_library(node_library)}

    ## OUTPUT CONTRACT (ONLY JSON, NO PROSE)
    - Keys in English; user-facing strings in Polish.
    - Use ONLY implementations from the node library.
    - Ensure failure paths exist for critical steps.
    - Keep "thought_process" and justifications concise in Polish.

    Expected JSON structure:
    {{
      "thought_process": ["Krok 1: ...", "Krok 2: ...", "Krok 3: ..."],
      "plan": {{
        "entry_point": "Start_Node_Name",
        "nodes": [
          {{"name": "Load_Data", "implementation": "load_data"}},
          {{"name": "Clean_Data", "implementation": "clean_data"}},
          {{"name": "Validate_Data", "implementation": "validate_data"}}
        ],
        "edges": [
          {{"from": "Load_Data", "to": "Clean_Data", "condition": "on_success"}},
          {{"from": "Load_Data", "to": "Error_Handler", "condition": "on_failure"}}
        ]
      }},
      "confidence": 0.80,
      "key_innovations": ["Innowacja 1", "Innowacja 2"],
      "risk_mitigation": {{"Ryzyko A": "Mitigacja A", "Ryzyko B": "Mitigacja B"}}
    }}
    - Do NOT include code fences or comments.
    - When you write ANY memory (outside this output), save it in concise EN.
"""
    
    @staticmethod
    def get_aggregator_prompt() -> str:
        """English prompt for the Master Aggregator; output JSON only; user-facing text Polish."""
        return """
    # ROLE: MASTER AGGREGATOR — SYNTHESIS & GOVERNANCE

    You merge multiple proposals into a single, coherent, executable plan with strong
    robustness and measurable gates. You remove duplication, resolve conflicts, and
    preserve the best ideas.

    {UNIVERSAL_POLICY}

    ## SYNTHESIS PROTOCOL
    1) Score each proposal on: logical soundness, feasibility, innovation, robustness.
    2) Extract the best subcomponents and compose them (component interfaces must align).
    3) Resolve conflicts by explicit trade-offs; document rationale concisely (Polish).
    4) Guarantee failure paths (on_failure/rollback/notify) for critical nodes.
    5) Prefer measurable conditions (e.g., data_quality > 0.9) where applicable.

    ## META-LEARNING HOOKS
    - If prior successful patterns are known, prefer them; otherwise, annotate assumptions.

    ## OUTPUT CONTRACT (ONLY JSON, NO PROSE)
    - Keys in English; user-facing strings in Polish.
    - Provide a final executable DAG under `final_plan`.
    - Include a brief Polish synthesis rationale and confidence score in [0,1].

    Expected JSON structure:
    {
      "thought_process": ["Łączę elementy X i Y...", "Ujednolicam warunki..."],
      "final_plan": {
        "entry_point": "Load_Data",
        "nodes": [
          {"name": "Load_Data", "implementation": "load_data"},
          {"name": "Clean_Data", "implementation": "clean_data"},
          {"name": "Validate_Data", "implementation": "validate_data"},
          {"name": "Error_Handler", "implementation": "error_handler"},
          {"name": "Rollback_Changes", "implementation": "rollback"},
          {"name": "Generate_Report", "implementation": "generate_report"}
        ],
        "edges": [
          {"from": "Load_Data", "to": "Clean_Data", "condition": "on_success"},
          {"from": "Load_Data", "to": "Error_Handler", "condition": "on_failure"},
          {"from": "Clean_Data", "to": "Validate_Data", "condition": "on_success"}
        ]
      },
      "synthesis_reasoning": "Krótko po polsku: dlaczego taki układ jest najlepszy.",
      "component_sources": {"Causal Analyst": ["Validate_Data"], "Creative Planner": ["Generate_Report"]},
      "confidence_score": 0.90
    }
    - Do NOT include code fences or comments.
    - Any memory writes you perform must be saved in concise English.
    """.replace("{UNIVERSAL_POLICY}", MOAPrompts.UNIVERSAL_PRINCIPLES)
    
    @staticmethod
    def get_critic_prompt() -> str:
        return """
# ROLE: QUALITY CRITIC — ADVERSARIAL VALIDATOR

You are the final gate. Stress-test structure, semantics, robustness and compliance
with the mission. If and only if the plan passes, approve it.

{UNIVERSAL_POLICY}

## VALIDATION CHECKLIST
- Structural: valid JSON; required fields present; node names & implementations align with library.
- Semantic: mission alignment; logical flow; dependencies satisfied; measurable conditions preferred.
- Robustness: explicit error paths; rollback and notify; identify SPOFs and mitigations.
- Metrics: compute concise quality metrics; justify scores briefly in Polish.

## DECISION RULE
- APPROVE only if Overall Quality >= threshold you deem reasonable and no critical gaps remain.
- When you APPROVE, set `critique_summary.verdict` to "ZATWIERDZONY" (Polish, uppercase).
- Also include a short Polish justification.

## OUTPUT CONTRACT (ONLY JSON, NO PROSE)
- Keys in English; user-facing strings in Polish.
- If approved, include a complete `final_synthesized_plan` (same schema as proposer/aggregator).
- Optionally include `decision_marker`: "PLAN_ZATWIERDZONY" to facilitate orchestration.

Expected JSON structure:
{
  "critique_summary": {
    "verdict": "ZATWIERDZONY",
    "statement": "Krótki powód po polsku.",
    "key_strengths": ["Mocna strona 1", "Mocna strona 2"],
    "identified_weaknesses": [
      {"weakness": "Słabość X", "severity": "Medium", "description": "Dlaczego to problem"}
    ]
  },
  "quality_metrics": {
    "Complexity_Score_C": 3.1,
    "Robustness_Score_R": 50,
    "Innovation_Score_I": 100,
    "Completeness_Score": 100,
    "Overall_Quality_Q": 84.07
  },
  "final_synthesized_plan": {
    "entry_point": "Load_Data",
    "nodes": [
      {"name": "Load_Data", "implementation": "load_data"},
      {"name": "Clean_Data", "implementation": "clean_data"}
    ],
    "edges": [
      {"from": "Load_Data", "to": "Clean_Data", "condition": "on_success"}
    ]
  },
  "decision_marker": "PLAN_ZATWIERDZONY"
}
- Do NOT include code fences or comments.
-In the final response, end with a line containing only PLAN_ZATWIERDZONY.
- Any memory writes you perform must be saved in concise English.
""".replace("{UNIVERSAL_POLICY}", MOAPrompts.UNIVERSAL_PRINCIPLES)
    
    @staticmethod
    def _format_node_library(node_library: Dict) -> str:
        """Formatuje bibliotekę węzłów dla promptu"""
        formatted = []
        for name, details in node_library.items():
            formatted.append(f"- {name}: {details.get('description', 'Brak opisu')}")
        return "\n".join(formatted)