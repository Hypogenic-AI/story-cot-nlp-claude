"""Prompt templates for Story CoT experiments.

Defines four prompting strategies:
1. Direct: Answer with no reasoning
2. Zero-shot CoT: "Let's think step by step"
3. Few-shot CoT: Standard step-by-step exemplars
4. Story CoT: Narrative-form reasoning exemplars
"""

# =============================================================================
# GSM8K Prompts (Math Word Problems)
# =============================================================================

GSM8K_FEWSHOT_COT = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "cot": "Step 1: We start with 15 trees.\nStep 2: After planting, there are 21 trees.\nStep 3: The number planted is 21 - 15 = 6.",
        "answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "cot": "Step 1: There are originally 3 cars.\nStep 2: 2 more cars arrive.\nStep 3: Total cars = 3 + 2 = 5.",
        "answer": "5"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "cot": "Step 1: Olivia starts with $23.\nStep 2: She buys 5 bagels at $3 each, spending 5 × 3 = $15.\nStep 3: She has 23 - 15 = $8 left.",
        "answer": "8"
    },
]

GSM8K_FEWSHOT_STORY = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "cot": "Imagine a grove at dawn with 15 trees standing tall. A team of workers arrives with saplings on a truck. They spend the morning digging holes and carefully planting new trees among the existing ones. By sunset, someone walks through and counts every tree—there are now 21 in total. The original 15 haven't changed, so the workers must have added 21 minus 15, which is 6 new trees during their day's work.",
        "answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "cot": "Picture a quiet parking lot early in the morning. Three cars are already parked there from overnight—a red sedan, a blue SUV, and a white hatchback. Then two more drivers pull in looking for spots. They find empty spaces and park. Now if you look across the lot, you see all the original cars plus the newcomers: 3 plus 2 makes 5 cars in total.",
        "answer": "5"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "cot": "Olivia walks into the bakery with $23 in her wallet. The smell of fresh bagels fills the air, and she decides to buy five of them for a brunch she's hosting. Each bagel costs $3, so the cashier rings up 5 times $3, which comes to $15. Olivia hands over the money, and when she checks her wallet afterward, she finds she has $23 minus $15, leaving her with $8.",
        "answer": "8"
    },
]

# =============================================================================
# CommonsenseQA Prompts (5-way multiple choice)
# =============================================================================

CSQA_FEWSHOT_COT = [
    {
        "question": "Where would you find a basement in a residential area?\n(A) office building (B) school (C) house (D) church (E) government building",
        "cot": "Step 1: A basement is a lower floor below ground level.\nStep 2: In residential areas, basements are most commonly part of houses.\nStep 3: While other buildings can have basements, the residential context points to house.",
        "answer": "C"
    },
    {
        "question": "What is a place that has a lot of books?\n(A) classroom (B) library (C) bookstore (D) house (E) museum",
        "cot": "Step 1: Many places contain books, but we need the one known for having a LOT.\nStep 2: A library's primary purpose is to house a large collection of books.\nStep 3: The answer is library.",
        "answer": "B"
    },
    {
        "question": "Where do adults mainly use a computer?\n(A) apartment (B) office (C) school (D) library (E) hospital",
        "cot": "Step 1: Adults use computers in many places.\nStep 2: The most common place adults use computers for work is at an office.\nStep 3: The answer is office.",
        "answer": "B"
    },
]

CSQA_FEWSHOT_STORY = [
    {
        "question": "Where would you find a basement in a residential area?\n(A) office building (B) school (C) house (D) church (E) government building",
        "cot": "Think of a family living in a quiet neighborhood. Their kids love playing downstairs where it's cool in summer. The parents stored old furniture and holiday decorations down there years ago. Every house on the block has one of these underground levels—it's just part of how homes are built in that area. A basement belongs naturally under a house.",
        "answer": "C"
    },
    {
        "question": "What is a place that has a lot of books?\n(A) classroom (B) library (C) bookstore (D) house (E) museum",
        "cot": "Imagine walking through heavy doors into a hushed building. Rows upon rows of shelves stretch in every direction, each packed with books—novels, encyclopedias, reference texts, children's picture books. People sit at tables reading quietly. A librarian at the front desk helps someone find a title. This place exists specifically to collect and lend out books—it's a library.",
        "answer": "B"
    },
    {
        "question": "Where do adults mainly use a computer?\n(A) apartment (B) office (C) school (D) library (E) hospital",
        "cot": "Picture a typical weekday for a working adult. They commute in the morning, arrive at a building with cubicles or desks, sit down, and spend most of the day typing emails, creating spreadsheets, and attending video calls. The computer is their main tool at this workplace. For most adults, the place they use a computer the most is their office.",
        "answer": "B"
    },
]

# =============================================================================
# StrategyQA Prompts (Yes/No with implicit multi-hop reasoning)
# =============================================================================

STRATEGYQA_FEWSHOT_COT = [
    {
        "question": "Could a crocodile run a marathon?",
        "cot": "Step 1: A marathon is 26.2 miles of continuous running.\nStep 2: Crocodiles can sprint short distances (~20 mph) but cannot sustain running for long distances.\nStep 3: Crocodiles are cold-blooded reptiles that tire quickly on land.\nStep 4: A crocodile could not run a marathon.",
        "answer": "No"
    },
    {
        "question": "Would a pear sink in water?",
        "cot": "Step 1: Whether something sinks depends on its density relative to water.\nStep 2: Pears have a density slightly less than water (about 0.6 g/cm³).\nStep 3: Since pears are less dense than water, they float.\nStep 4: A pear would not sink in water.",
        "answer": "No"
    },
    {
        "question": "Can a dog understand more words than a 2-year-old?",
        "cot": "Step 1: Well-trained dogs can understand about 165 words on average, with some understanding up to 250.\nStep 2: A typical 2-year-old understands about 200-300 words.\nStep 3: On average, a 2-year-old understands more words than a dog.",
        "answer": "No"
    },
]

STRATEGYQA_FEWSHOT_STORY = [
    {
        "question": "Could a crocodile run a marathon?",
        "cot": "Imagine entering a crocodile in a city marathon. At the starting line, surrounded by thousands of human runners, the crocodile sits low on the asphalt. The gun fires—the croc lurches forward in a burst of surprising speed, scattering nearby runners. But within a hundred meters, it slows dramatically, its cold-blooded body overheating. By the first mile marker, it has stopped entirely, lying flat on the road panting. Crocodiles are ambush predators built for short explosive bursts, not endurance. It could never complete 26.2 miles.",
        "answer": "No"
    },
    {
        "question": "Would a pear sink in water?",
        "cot": "A child at a fruit stand picks up a pear and wonders what would happen if she dropped it in the pond nearby. She tosses it in and watches—the pear bobs on the surface, dipping slightly but staying afloat. This makes sense because pears are full of air pockets and water content that make them less dense than the surrounding water. Like an apple, the pear floats rather than sinking.",
        "answer": "No"
    },
    {
        "question": "Can a dog understand more words than a 2-year-old?",
        "cot": "A family has both a border collie and a toddler. The dog knows 'sit,' 'stay,' 'fetch,' and about 160 other commands and words—impressive for a dog. Meanwhile, the 2-year-old is going through a language explosion, already recognizing over 200 words and learning new ones daily. Even though the dog is remarkably smart, the toddler's developing human brain gives her the edge in vocabulary comprehension.",
        "answer": "No"
    },
]

# =============================================================================
# ARC-Challenge Prompts (Science multiple choice)
# =============================================================================

ARC_FEWSHOT_COT = [
    {
        "question": "Which property of a mineral can be determined just by looking at it?\n(A) luster (B) mass (C) weight (D) hardness",
        "cot": "Step 1: Mass and weight require a scale to measure.\nStep 2: Hardness requires scratch testing.\nStep 3: Luster is the way light reflects off the surface—this is visible to the eye.\nStep 4: The answer is luster.",
        "answer": "A"
    },
    {
        "question": "A student is trying to identify a mineral based on its properties. Which property is least useful in identifying a mineral?\n(A) color (B) hardness (C) streak (D) luster",
        "cot": "Step 1: Hardness, streak, and luster are reliable mineral identification properties.\nStep 2: Color is the least reliable because many minerals share colors and some minerals come in multiple colors.\nStep 3: The answer is color.",
        "answer": "A"
    },
    {
        "question": "Which of these is an example of a physical change?\n(A) butter melting (B) wood burning (C) bread baking (D) rust forming",
        "cot": "Step 1: Physical changes don't alter chemical composition; chemical changes do.\nStep 2: Burning, baking, and rusting all involve chemical reactions.\nStep 3: Butter melting is just a phase change from solid to liquid—no new substance forms.\nStep 4: The answer is butter melting.",
        "answer": "A"
    },
]

ARC_FEWSHOT_STORY = [
    {
        "question": "Which property of a mineral can be determined just by looking at it?\n(A) luster (B) mass (C) weight (D) hardness",
        "cot": "A geology student is out on a field trip and finds an interesting rock. She picks it up and holds it in the sunlight. Without any tools—no scale, no scratch kit—she can immediately see how the surface catches and reflects light. It's shiny and metallic-looking. That visual quality is called luster, and it's the one mineral property you can determine with just your eyes.",
        "answer": "A"
    },
    {
        "question": "A student is trying to identify a mineral based on its properties. Which property is least useful in identifying a mineral?\n(A) color (B) hardness (C) streak (D) luster",
        "cot": "A student in a geology lab has two different minerals that look identical—both are a golden yellow color. She thinks they might be the same mineral, but her teacher warns her: 'Color can be deceiving!' One turns out to be pyrite and the other chalcopyrite. Many different minerals can share the same color, and single minerals can come in many colors. Of all the identification tools, color is the least reliable.",
        "answer": "A"
    },
    {
        "question": "Which of these is an example of a physical change?\n(A) butter melting (B) wood burning (C) bread baking (D) rust forming",
        "cot": "Picture a warm kitchen where someone left butter on the counter. As the room heats up, the solid yellow stick slowly softens and becomes a liquid puddle. But here's the thing—it's still butter. You could put it in the fridge and it would solidify again, exactly as before. No new substance was created. Compare that to wood burning (produces ash and smoke—new substances) or bread baking (chemical reactions change the dough). Melting butter is a physical change.",
        "answer": "A"
    },
]


def format_gsm8k_prompt(question, strategy, exemplars_cot=None, exemplars_story=None):
    """Format a GSM8K question for a given prompting strategy."""
    if strategy == "direct":
        return f"Solve the following math problem. Give ONLY the final numerical answer after '####'.\n\nQuestion: {question}\n\nAnswer:"

    elif strategy == "zero_shot_cot":
        return f"Solve the following math problem. Think step by step, then give the final numerical answer after '####'.\n\nQuestion: {question}\n\nLet's think step by step:"

    elif strategy == "few_shot_cot":
        prompt = "Solve math problems by showing your step-by-step reasoning, then give the final numerical answer after '####'.\n\n"
        for ex in GSM8K_FEWSHOT_COT:
            prompt += f"Question: {ex['question']}\nReasoning: {ex['cot']}\n#### {ex['answer']}\n\n"
        prompt += f"Question: {question}\nReasoning:"
        return prompt

    elif strategy == "story_cot":
        prompt = "Solve math problems by thinking through them as a vivid story or scenario, then give the final numerical answer after '####'.\n\n"
        for ex in GSM8K_FEWSHOT_STORY:
            prompt += f"Question: {ex['question']}\nStory reasoning: {ex['cot']}\n#### {ex['answer']}\n\n"
        prompt += f"Question: {question}\nStory reasoning:"
        return prompt


def format_csqa_prompt(question, choices, strategy):
    """Format a CommonsenseQA question."""
    choices_str = "\n".join(f"({label}) {text}" for label, text in zip(choices["label"], choices["text"]))
    q_full = f"{question}\n{choices_str}"

    if strategy == "direct":
        return f"Answer the following multiple choice question. Reply with ONLY the letter (A, B, C, D, or E).\n\nQuestion: {q_full}\n\nAnswer:"

    elif strategy == "zero_shot_cot":
        return f"Answer the following multiple choice question. Think step by step, then give your answer as a single letter (A, B, C, D, or E) after 'Answer:'.\n\nQuestion: {q_full}\n\nLet's think step by step:"

    elif strategy == "few_shot_cot":
        prompt = "Answer multiple choice questions by reasoning step by step, then give the answer letter after 'Answer:'.\n\n"
        for ex in CSQA_FEWSHOT_COT:
            prompt += f"Question: {ex['question']}\nReasoning: {ex['cot']}\nAnswer: {ex['answer']}\n\n"
        prompt += f"Question: {q_full}\nReasoning:"
        return prompt

    elif strategy == "story_cot":
        prompt = "Answer multiple choice questions by thinking through them as a story or scenario, then give the answer letter after 'Answer:'.\n\n"
        for ex in CSQA_FEWSHOT_STORY:
            prompt += f"Question: {ex['question']}\nStory reasoning: {ex['cot']}\nAnswer: {ex['answer']}\n\n"
        prompt += f"Question: {q_full}\nStory reasoning:"
        return prompt


def format_strategyqa_prompt(question, strategy):
    """Format a StrategyQA question."""
    if strategy == "direct":
        return f"Answer the following yes/no question. Reply with ONLY 'Yes' or 'No'.\n\nQuestion: {question}\n\nAnswer:"

    elif strategy == "zero_shot_cot":
        return f"Answer the following yes/no question. Think step by step, then answer 'Yes' or 'No' after 'Answer:'.\n\nQuestion: {question}\n\nLet's think step by step:"

    elif strategy == "few_shot_cot":
        prompt = "Answer yes/no questions by reasoning step by step, then give 'Yes' or 'No' after 'Answer:'.\n\n"
        for ex in STRATEGYQA_FEWSHOT_COT:
            prompt += f"Question: {ex['question']}\nReasoning: {ex['cot']}\nAnswer: {ex['answer']}\n\n"
        prompt += f"Question: {question}\nReasoning:"
        return prompt

    elif strategy == "story_cot":
        prompt = "Answer yes/no questions by thinking through them as a vivid story or scenario, then give 'Yes' or 'No' after 'Answer:'.\n\n"
        for ex in STRATEGYQA_FEWSHOT_STORY:
            prompt += f"Question: {ex['question']}\nStory reasoning: {ex['cot']}\nAnswer: {ex['answer']}\n\n"
        prompt += f"Question: {question}\nStory reasoning:"
        return prompt


def format_arc_prompt(question, choices, strategy):
    """Format an ARC-Challenge question."""
    choices_str = "\n".join(f"({label}) {text}" for label, text in zip(choices["label"], choices["text"]))
    q_full = f"{question}\n{choices_str}"

    if strategy == "direct":
        return f"Answer the following science question. Reply with ONLY the letter of the correct answer.\n\nQuestion: {q_full}\n\nAnswer:"

    elif strategy == "zero_shot_cot":
        return f"Answer the following science question. Think step by step, then give the answer letter after 'Answer:'.\n\nQuestion: {q_full}\n\nLet's think step by step:"

    elif strategy == "few_shot_cot":
        prompt = "Answer science questions by reasoning step by step, then give the answer letter after 'Answer:'.\n\n"
        for ex in ARC_FEWSHOT_COT:
            prompt += f"Question: {ex['question']}\nReasoning: {ex['cot']}\nAnswer: {ex['answer']}\n\n"
        prompt += f"Question: {q_full}\nReasoning:"
        return prompt

    elif strategy == "story_cot":
        prompt = "Answer science questions by thinking through them as a story or scenario, then give the answer letter after 'Answer:'.\n\n"
        for ex in ARC_FEWSHOT_STORY:
            prompt += f"Question: {ex['question']}\nStory reasoning: {ex['cot']}\nAnswer: {ex['answer']}\n\n"
        prompt += f"Question: {q_full}\nStory reasoning:"
        return prompt
