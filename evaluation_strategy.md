# SASLM Evaluation Strategy

This document outlines the evaluation methodology for the Sri Aurobindo Small Language Model (SASLM). Unlike standard LLM benchmarks which focus on broad knowledge or simple reasoning, our evaluation must determine if the model has internalized the specific *ontology, style, and coherence* of Sri Aurobindo's Integral Yoga.

Reference Paper: *TinyStories: How Small Can Language Models Be?* (Eldan & Li, 2023).

## 1. The "Yogic Turing Test" (Qualitative Grading)
**Objective**: Assess whether the model captures the "Voice" and "Truth" of the corpus, or merely imitates surface statistics.

**Method**: 
Generate completions for 50 unseen philosophical prompts (e.g., *"The distinction between the psychic being and the spiritual self is..."*). Use a superior model (GPT-4) as a judge to grade the outputs on three dimensions:

*   **Ontological Accuracy (1-10)**: Does the model respect the specific hierarchy of existence (Involuion/Evolution, Supermind vs Overmind)? 
    *   *Fail State*: Conflating the 'Vital' with the 'Psychic'.
*   **Stylistic Fidelity (1-10)**: Does the text exhibit Aurobindo's characteristic sentence structure (recursive, rhythmic, precise) and vocabulary (*ineffable, nescience, supramental*)?
*   **Coherence (1-10)**: Does the generated argument follow a logical progression, or does it devolve into "word salad"?

## 2. "Part-Story" / Argument Completion (Generalization)
**Objective**: Test for generalization versus memorization.

**Method**:
1.  Take a paragraph from the held-out test set (e.g., from *The Life Divine*).
2.  Cut it off at the halfway point.
3.  Have SASLM generate the completion.
4.  **Metric**: Compare the generated ending with the ground truth using semantic similarity (embedding distance), *not* exact n-gram matching (ROUGE).
    *   *Goal*: The model should produce a *plausible* alternative explanation that is philosophically consistent, even if the words differ.

## 3. Concept Association Probes (Interpretability)
**Objective**: Verify that the model has encoded structural relationships between concepts, rather than just frequent co-occurrences.

**Method**:
Create "Fill-in-the-Mask" style prompts that require understanding the specific ontology.

*   *Prompt*: "The three lower planes of existence are Matter, Life, and [MASK]."
    *   *Expected*: `Mind`
    *   *Failure*: `Love`, `Death`, `God` (generically related but ontologically incorrect).
*   *Prompt*: "The power that mediates between the Supermind and the Mind is the [MASK]."
    *   *Expected*: `Overmind`

## 4. Depth Ablation Studies
**Objective**: Determine the architectural requirements for "Wisdom".

**Method**:
Train two versions of SASLM on the same data for the same number of tokens:
1.  **Deep Model** (Current): 8 Layers, 8 Heads.
2.  **Shallow Model** (Control): 2 Layers, Same total parameters (wider).

*Hypothesis*: The Shallow model may achieve good perplexity and surface grammar but will likely fail the **Ontological Accuracy** test, proving that "depth" is required to model the complex hierarchical relationships of the philosophy.

## 5. Neuron Monitoring (Future Work)
**Objective**: Identify if specific neurons/circuits are responsible for specific concepts.
*   *Experiment*: Monitor activation patterns when processing high-frequency technical terms like *Purusha* or *Prakriti*. High activation consistency suggests the model has "dedicated" a feature to this concept.
