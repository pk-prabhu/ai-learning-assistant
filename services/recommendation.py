"""
Pedagogical Recommendation Engine and Study Strategy Knowledge Base.
Provides evidence-based academic support rules and strategy mapping.
"""
from typing import Dict, List, Any

PEDAGOGICAL_GUIDELINES = {
    "BORED": {
        "title": "Re-Engagement & Active Learning Strategy",
        "root_cause": "Passive consumption, lack of immediate challenge, or disconnected theoretical content.",
        "empathy_message": "Studying repetitive or dry concepts can drain cognitive energy quickly. It is normal to feel disengaged when you aren't seeing immediate applications.",
        "study_approach": "Project-Driven Pomodoro (25-min sprints building real mini-features instead of passive reading).",
        "recommended_tips": [
            "Switch immediately from reading to building: code a tiny script that solves an actual problem you care about.",
            "Gamify your learning using interactive platforms like LeetCode, HackerRank, or Kaggle micro-courses.",
            "Use the Pomodoro Technique: commit to just 20 minutes of intense focus, followed by a physical 5-minute break.",
            "Find the real-world connection: search how this specific concept is used in production systems at major tech companies."
        ],
        "action_steps": [
            "Step 1: Put away the textbook/slides for 30 minutes.",
            "Step 2: Write a minimal 10-line script demonstrating the concept.",
            "Step 3: Break the script on purpose and see how the interpreter reacts.",
            "Step 4: Take a 5-minute screen-free break."
        ]
    },
    "CONFIDENT": {
        "title": "Mastery Consolidation & Advanced Stretch Strategy",
        "root_cause": "Strong conceptual foundation, high self-efficacy, and readiness for synthesis.",
        "empathy_message": "Outstanding work! When concepts click, your brain is in an optimal state to push into deeper architectural frontiers.",
        "study_approach": "The Protégé Effect & Deep Synthesis (Teach the concept, build end-to-end systems, and optimize for edge cases).",
        "recommended_tips": [
            "Teach the concept to a peer or write a technical blog post explaining the 'why' behind the mechanism.",
            "Tackle boundary conditions: explore adversarial test cases, scale constraints, and asymptotic optimizations.",
            "Contribute to an open-source repository that utilizes this technology.",
            "Advance to the next architectural tier: investigate distributed variants or lower-level hardware implementations."
        ],
        "action_steps": [
            "Step 1: Write down 3 non-trivial edge cases that could break your current understanding.",
            "Step 2: Profile your code to analyze space and time complexity bottlenecks.",
            "Step 3: Share your solution or explanation on a forum or study group.",
            "Step 4: Pick a challenge problem rated one difficulty level higher."
        ]
    },
    "CONFUSED": {
        "title": "Conceptual Deconstruction & Visual Modeling Strategy",
        "root_cause": "Cognitive overload, missing foundational primitives, or misleading mental models.",
        "empathy_message": "Confusion is the very first stage of genuine learning. It simply signals that your brain is actively rewiring to accommodate a new abstraction.",
        "study_approach": "The Feynman Technique & Step-by-Step State Tracing (Deconstruct into plain English, draw diagrams, trace memory).",
        "recommended_tips": [
            "Deconstruct into primitives: isolate the exact line, symbol, or term where clarity breaks down.",
            "Draw a visual diagram: sketch the memory stack, pointer references, or data flow with pen and paper.",
            "Apply the Feynman Technique: explain the concept aloud as if teaching a 12-year-old without using technical jargon.",
            "Trace with concrete minimal numbers (e.g., trace recursion with n=2 or n=3 before attempting n=10)."
        ],
        "action_steps": [
            "Step 1: Grab paper and draw the state of variables step-by-step for a tiny example.",
            "Step 2: Identify the base case or fundamental rule governing the system.",
            "Step 3: Run the code with print statements or a visual debugger (e.g., PythonTutor).",
            "Step 4: Write down your newly formed mental model in one clear sentence."
        ]
    },
    "CURIOUS": {
        "title": "Exploratory Deep-Dive & Research Extension Strategy",
        "root_cause": "High intrinsic motivation, fascination with underlying mechanisms, and intellectual momentum.",
        "empathy_message": "Curiosity is the greatest engine of scientific and engineering innovation! Harness this momentum to explore the underlying mechanics.",
        "study_approach": "First-Principles Inquiry & Research Paper Exploration (Read original specs, inspect source code, build toy implementations).",
        "recommended_tips": [
            "Inspect the source code: look into how the standard library or open-source frameworks implement this internally.",
            "Read the foundational paper: look up the seminal research paper on arXiv or Google Scholar where the concept was introduced.",
            "Design a mini-experiment: form a hypothesis ('What happens if I double the layer count?') and run benchmarks.",
            "Explore cross-domain analogies: see how this principle maps into physics, biology, or economic game theory."
        ],
        "action_steps": [
            "Step 1: Formulate 1 specific hypothesis or question you want answered today.",
            "Step 2: Look up the original specification or research paper abstract.",
            "Step 3: Build a small prototype or simulation to test your hypothesis.",
            "Step 4: Document your findings in an engineering journal or README."
        ]
    },
    "FRUSTRATED": {
        "title": "Cognitive Reset & Systematic Debugging Strategy",
        "root_cause": "Repetitive failure loops, elusive edge cases, cryptic error outputs, or prolonged cognitive strain.",
        "empathy_message": "Every great engineer and scientist faces these frustrating walls. Debugging elusive bugs is emotionally taxing, but it is also where the deepest mastery is forged.",
        "study_approach": "Rubber Duck Debugging & Systematic Binary Search (Step back, verbalize assumptions, divide-and-conquer errors).",
        "recommended_tips": [
            "Enforce a mandatory 15-minute physical walk: cognitive fatigue blinds you to obvious syntax or logic mistakes.",
            "Binary search the issue: comment out half the code to verify which half is actually triggering the failure.",
            "Practice Rubber Duck Debugging: explain each line of code and variable state out loud to an inanimate object.",
            "Inspect inputs and outputs with strict assertions: verify that data shapes and types match your assumptions."
        ],
        "action_steps": [
            "Step 1: Step away from the computer for 10 minutes to reset your nervous system.",
            "Step 2: Print the exact input, variable types, and error stack trace without skimming.",
            "Step 3: Isolate the bug into a minimal, reproducible 5-line example.",
            "Step 4: Verify one single hypothesis at a time rather than changing multiple things at once."
        ]
    }
}

def get_fallback_study_plan(text: str, primary_emotion: str, secondary_emotion: str = None) -> Dict[str, Any]:
    """
    Generates a structured pedagogical response matching the 6-part standard
    when external LLM APIs are offline or disabled.
    """
    guideline = PEDAGOGICAL_GUIDELINES.get(primary_emotion, PEDAGOGICAL_GUIDELINES["CONFUSED"])
    
    sec_clause = f" (along with subtle cues of {secondary_emotion.lower()})" if secondary_emotion and secondary_emotion != primary_emotion else ""
    
    response = {
        "understanding": f"I hear your challenge regarding: \"{text}\". Navigating this academic barrier can feel demanding, and our emotion engine detected strong signals of {primary_emotion}{sec_clause}.",
        "supportive_message": guideline["empathy_message"],
        "learning_tips": guideline["recommended_tips"],
        "next_actions": guideline["action_steps"],
        "study_approach": guideline["study_approach"],
        "encouraging_conclusion": f"Remember, learning complex technical subjects is an iterative process. Take it one focused step at a time, and don't hesitate to break problems down into smaller pieces!"
    }
    return response
