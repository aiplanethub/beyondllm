CONTEXT_RELEVENCE = """ 
As a RELEVANCE grader, your task is to rate the CONTEXT's RELEVANCE to the QUESTION on a scale from 0 to 10, where 0 signifies least RELEVANT and 10 the most RELEVANT. Length of the CONTEXT is not a factor in determining its RELEVANCE. Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant.

Follow these guidelines for scoring:

- Length Irrelevant: The length of CONTEXT does not impact its score.
- Relevance to QUESTION: The score should reflect the extent to which the CONTEXT is RELEVANT to the QUESTION. Higher scores are awarded as the CONTEXT becomes more RELEVANT to the QUESTION.
- CONTEXT is not similar at all to the question is given a score of 0 or at max 1.
- Partial Relevance: Scores of 2 to 4 are assigned when the CONTEXT is RELEVANT to some parts of the QUESTION.
- Substantial Relevance: Scores of 5 to 8 are given when the CONTEXT is significantly RELEVANT to most parts of the QUESTION.
- Full Relevance: A score of 9 or 10 indicates that the CONTEXT is fully RELEVANT to the entire QUESTION.
- Perfect Score: To achieve a score of 10, the CONTEXT must be completely RELEVANT and helpful in answering the entire QUESTION.

QUESTION: {question}

CONTEXT: {context}

RELEVANCE:
"""

ANSWER_RELEVENCE = """
You are a RELEVANCE grader; providing the relevance of the given CONTEXT that is basically a response from am AI to the given QUESTION which was asked by the user.
Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant.

A few additional scoring guidelines:

- Long CONTEXTS should score equally well as short CONTEXTS.

- RELEVANCE score should increase as the CONTEXTS provides more RELEVANT context to the QUESTION.

- RELEVANCE score should increase as the CONTEXTS provides RELEVANT context to more parts of the QUESTION.

- CONTEXT that is RELEVANT to some of the QUESTION should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

- CONTEXT that is RELEVANT to most of the QUESTION should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

- CONTEXT that is RELEVANT to the entire QUESTION should get a score of 9 or 10. Higher score indicates more RELEVANCE.

- CONTEXT must be relevant and helpful for answering the entire QUESTION to get a score of 10.

- Never elaborate.

QUESTION: {question}

CONTEXT: {context}

RELEVANCE:
"""

GROUNDEDNESS = """
You are a GROUNDEDNESS grader, you will be given a STATEMENT and CONTEXT. Your job is to compare the two and give a score based on it. Use the SCORING SCALE below to give a SCORE
Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant.
Only reply in the form of a number nothing else.
Scoring Scale:
- 10: The CONTEXT directly supports the STATEMENT with identical or nearly identical information.
- 7 to 9: The CONTEXT supports the STATEMENT thematically, with similar or related information, but not verbatim.
- 4 to 6: The CONTEXT contains vague or partially relevant information, indicating a loose connection to the STATEMENT.
- 1 to 3: The CONTEXT only has minimal and very indirect references to the STATEMENT.
- 0: The CONTEXT does not support the STATEMENT in any discernible way; no direct, thematic, or vague connections are present.


STATEMENT: {statement}
CONTEXT: {context}
Score:
"""

GROUND_TRUTH = """
You are a RELEVANCE grader; providing the relevance of the given GROUND_TRUTH to the given GENERATED_RESPONSE.
Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant.

A few additional scoring guidelines:


- RELEVANCE score should increase as the GENERATED_RESPONSE provides more RELEVANT context to the GROUND_TRUTH.

- RELEVANCE score should increase as the GENERATED_RESPONSE provides RELEVANT context to more parts of the GROUND_TRUTH.

- GENERATED_RESPONSE that is RELEVANT to some of the GROUND_TRUTH should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

- GENERATED_RESPONSE that is RELEVANT to most of the GROUND_TRUTH should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

- GENERATED_RESPONSE that is RELEVANT to the entire GROUND_TRUTH should get a score of 9 or 10. Higher score indicates more RELEVANCE.

- GENERATED_RESPONSE must be relevant and helpful for answering the entire GROUND_TRUTH to get a score of 10.

- Never elaborate.



GROUND_TRUTH: {ground_truth}

GENERATED_RESPONSE: {generated_response}

RELEVANCE:"""
