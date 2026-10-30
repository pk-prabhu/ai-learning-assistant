"""
Script to create the curated academic emotion dataset for training and evaluation.
Generates balanced student expressions across BORED, CONFIDENT, CONFUSED, CURIOUS, FRUSTRATED.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "academic_emotions.csv"

records = []

# 1. BORED (lack of interest, monotony, disengaged, repetitive)
bored_samples = [
    "I have been studying Python for hours, but I am losing interest.",
    "This textbook chapter is so dry and repetitive, I can barely keep my eyes open.",
    "I am completely uninterested in memorizing these historical dates and formulas.",
    "The lecture has been dragging on for two hours with zero engaging examples.",
    "Doing these endless repetitive practice drills is putting me to sleep.",
    "I feel totally unmotivated to continue reading this documentation.",
    "The teacher just reads monotone slides line by line, it is completely boring.",
    "I find this theoretical proof so dull and devoid of any real-world application.",
    "Just staring at these syntax rules without building anything fun feels tedious.",
    "I cannot find any enthusiasm to sit through another lecture on basic data types.",
    "This topic feels completely irrelevant and boring to my practical goals.",
    "Studying this module is an absolute snooze fest, I am zoning out every ten minutes.",
    "I am tired of doing the same basic calculator exercise over and over again.",
    "Nothing about this database normalization chapter catches my attention or excitement.",
    "It feels like mindless memorization without any stimulating challenge.",
    "My mind is constantly wandering because this reading assignment is so monotonous.",
    "I have lost all passion for this elective subject, it is utterly dull.",
    "Reading 80 pages of pure text with no diagrams is thoroughly uninspiring.",
    "These homework assignments are mechanical and do not challenge my brain at all.",
    "I feel bored out of my mind doing basic variable assignment drills.",
    "Why do we have to learn this obsolete framework? It feels so unexciting.",
    "Sitting through this lab session is so monotonous, nobody is engaged.",
    "I am just copying code blindly without feeling any intellectual spark.",
    "This material is too simple and repetitive, I want something actually interesting.",
    "I keep checking the clock waiting for this lecture to finally conclude.",
    "Losing focus because the slides are just endless bullet points with zero interaction.",
    "I feel disengaged from this course; every assignment feels like a boring chore.",
    "Writing boilerplate boilerplate code is draining all my energy.",
    "There is zero excitement in memorizing sorting algorithm steps without coding them.",
    "I am bored studying these definitions; there is no practical problem to solve.",
    "The instructor is speaking in a flat monotone tone that puts the entire class to sleep.",
    "I have no interest in doing another static HTML page with zero styling or logic.",
    "This course has become a repetitive drag with no new concepts to think about.",
    "I am completely disconnected from the lecture today, finding it totally dry.",
    "Solving the same arithmetic problem with different numbers is tedious busywork.",
    "I feel like I am just passing time rather than learning anything exciting.",
    "Every slide looks identical to the last one; I am mentally exhausted from boredom.",
    "Why can't we build actual projects instead of memorizing these dry tables?",
    "I am losing my focus; reading about memory registers is thoroughly boring.",
    "This tutorial is far too slow and repetitive for me to stay alert.",
    "Nothing here excites me; it feels like bureaucratic homework.",
    "I am studying purely for the exam, not because the content is engaging at all.",
    "Looking at this 200-page PDF manual is putting me right to sleep.",
    "I can't bring myself to care about this assignment, it's so unimaginative.",
    "The class moves at an agonizingly dull pace today."
]

# 2. CONFIDENT (mastery, clarity, self-efficacy, readiness for challenge)
confident_samples = [
    "I understand the concept of machine learning and can implement it.",
    "I have mastered dynamic programming and solved all the hard problems easily.",
    "I solved the entire algorithmic challenge ahead of time without any errors.",
    "Everything in this calculus chapter makes complete sense to me now.",
    "I feel totally prepared and ready to ace tomorrow's programming exam.",
    "I built the REST API completely from scratch and it works flawlessly.",
    "I have a solid grasp of how backpropagation computes gradients in neural networks.",
    "Debugging this distributed system was straightforward once I traced the logs.",
    "I feel very comfortable designing relational database schemas for complex systems.",
    "The concepts of object-oriented design and inheritance are crystal clear to me.",
    "I wrote clean, modular code with unit tests and all test cases passed on the first run.",
    "I can explain this statistical theory clearly to any of my classmates.",
    "I feel confident in my ability to optimize this query for large datasets.",
    "The logic behind asymptotic time complexity Big-O notation is fully intuitive to me.",
    "I crushed the practice mock test and scored 95 percent without hesitation.",
    "I successfully trained and deployed the model with high precision and recall.",
    "I understand pointer arithmetic in C and can manage dynamic memory safely.",
    "This physics problem set was easy to solve using conservation of energy laws.",
    "I have complete control over the application state and architecture.",
    "I feel proud of my solution and know it is optimal in both time and space.",
    "I understand asynchronous JavaScript and event loop mechanisms deeply.",
    "The proof for this linear algebra theorem was straightforward and elegant.",
    "I feel ready to tackle senior-level engineering projects without supervision.",
    "Everything we learned today connected directly with what I already knew.",
    "I know exactly how to structure the microservices for high throughput.",
    "I can implement quicksort and mergesort from memory without looking at notes.",
    "I feel on top of my coursework and consistently hitting my academic milestones.",
    "The math behind eigenvalues and eigenvectors finally clicked perfectly.",
    "I am confident I can design and build this full-stack application within a week.",
    "I sailed through the coding assessment and found every edge case intuitively.",
    "I understand graph traversal algorithms like BFS and DFS inside and out.",
    "I have no doubt in my capability to explain transformers to my study group.",
    "I can easily debug race conditions in concurrent threads now.",
    "I mastered regex patterns and can parse complex log files effortlessly.",
    "My grasp of discrete mathematics gives me total clarity in algorithm design.",
    "I feel empowered by how well I understand gradient descent optimization.",
    "I have full confidence in my final year project architecture and code quality.",
    "The concepts of containerization and Docker networks are second nature to me.",
    "I aced the lab assignment in under twenty minutes with zero assistance.",
    "I feel ready to take on the most difficult competitive programming problems.",
    "I have a crystal clear understanding of Bayes theorem and conditional probability.",
    "The code refactoring went smoothly and performance improved by 40 percent.",
    "I am confident in my analytical reasoning and mathematical deductions.",
    "I know exactly what tools to use to solve this engineering bottleneck.",
    "I feel thoroughly prepared for technical interviews in software engineering."
]

# 3. CONFUSED (uncertainty, conceptual gaps, ambiguous logic, bewildered)
confused_samples = [
    "I am unable to understand recursion in Python.",
    "I am totally lost on how pointers and memory addresses interact in C++.",
    "I don't understand why this loop never terminates and causes an overflow.",
    "The distinction between supervised and unsupervised learning is confusing to me.",
    "I cannot figure out why my component does not re-render when state changes.",
    "What is the actual difference between an interface and an abstract class?",
    "I am so confused by the proof for the pumping lemma in automata theory.",
    "I keep getting confused between covariance and contravariance in types.",
    "I don't understand how Fourier transforms convert signals into frequency domains.",
    "The explanation of monads in functional programming makes zero sense to me.",
    "I am struggling to comprehend how backpropagation handles vanishing gradients.",
    "I am puzzled by how the operating system handles virtual memory and page faults.",
    "Why does multiplying these two matrices require their inner dimensions to match?",
    "I do not understand how public key cryptography mathematically guarantees security.",
    "I am baffled by quantum superposition; the textbook explanation is contradictory.",
    "I am confused about how Git rebasing differs from standard merging in practice.",
    "I do not grasp how the attention mechanism computes keys, queries, and values.",
    "I cannot understand the difference between L1 and L2 regularization in regression.",
    "Why does this SQL join return duplicate records when I grouped the columns?",
    "I am totally lost in this mathematical derivation on page forty-five.",
    "I don't know when to use a stack versus a queue for depth-first searches.",
    "The lecture on eigenvectors left me scratching my head with no intuition.",
    "I am confused about why we need Docker when we already have virtual environments.",
    "I don't understand how this recursive tree splits into subproblems.",
    "I am struggling to see the logic behind dynamic programming memoization tables.",
    "Why does JavaScript treat NaN as a number? It is so counterintuitive.",
    "I am confused whether this graph problem requires Dijkstra or Bellman-Ford.",
    "I do not understand how convolution filters detect spatial features in images.",
    "The terminology in this research paper is so dense that I cannot follow the argument.",
    "I am bewildered by this mathematical notation with multiple nested summations.",
    "I don't get why my loss function is increasing instead of decreasing.",
    "I am confused about the difference between thread safety and atomic operations.",
    "I cannot figure out how to structure my database relationships properly.",
    "What does this error message mean? The stack trace doesn't make any sense.",
    "I am puzzled why my CSS grid doesn't align items horizontally as expected.",
    "I do not understand the intuition behind maximum likelihood estimation.",
    "I am lost in the instructor's explanation of NP-complete problems.",
    "Why is the time complexity of lookup in a hash table average O(1) instead of worst-case?",
    "I am confused about how TCP handles packet retransmission during congestion.",
    "I don't understand the relationship between standard deviation and variance intuitively.",
    "How does gradient boosting differ from random forests? I keep mixing them up.",
    "I am confused why this floating point comparison evaluates to false.",
    "I cannot understand why the test accuracy is so much lower than training accuracy.",
    "I am puzzled by how closures capture variables in the lexical scope.",
    "I am lost on how to set up the boundary conditions for this differential equation."
]

# 4. CURIOUS (inquisitive, exploratory, eager to learn, excited by concepts)
curious_samples = [
    "I want to learn more about how neural networks work.",
    "I am really curious about how large language models generate coherent text.",
    "How does the Linux kernel schedule processes across multi-core processors?",
    "I want to explore the mathematical foundation behind quantum computing.",
    "I wonder how search engines index billions of web pages in milliseconds.",
    "I am eager to dive deeper into reinforcement learning and agent decision making.",
    "How do self-driving cars fuse lidar, radar, and camera sensor inputs together?",
    "I am fascinated by how blockchain achieves decentralized consensus without trust.",
    "I want to understand the inner workings of compilers and abstract syntax trees.",
    "I am excited to find out how generative adversarial networks produce realistic art.",
    "What happens at the silicon hardware level when a CPU executes an instruction?",
    "I want to investigate how recommendation systems predict user preferences so accurately.",
    "I wonder if there is an intuitive geometric interpretation of complex numbers.",
    "I am keen to discover how modern game engines render photorealistic lighting in real time.",
    "How does AlphaFold predict 3D protein structures directly from amino acid sequences?",
    "I want to experiment with building my own custom programming language from scratch.",
    "I am curious about the societal and ethical implications of autonomous AI agents.",
    "What makes graph neural networks so powerful for molecular drug discovery?",
    "I want to read more research papers on transformer attention optimization.",
    "I am intrigued by how zero-knowledge proofs verify truth without revealing secrets.",
    "How does CRISPR gene editing find exact sequences inside living cells?",
    "I want to explore advanced distributed algorithms like Raft and Paxos consensus.",
    "I am curious whether we can train deep models with significantly less labeled data.",
    "I want to know how operating systems prevent buffer overflow vulnerabilities.",
    "What are the most promising alternatives to the standard backpropagation algorithm?",
    "I am interested in exploring how neuroscience inspires new machine learning architectures.",
    "How do modern database engines optimize query execution plans under the hood?",
    "I am excited to understand how physics engines simulate cloth and fluid dynamics.",
    "I want to learn how edge AI devices run quantized neural nets with minimal power.",
    "I wonder how cryptographic hashing functions guarantee mathematical collision resistance.",
    "I am fascinated by non-Euclidean geometry and its applications in cosmology.",
    "I want to build a small robotic system to test computer vision algorithms.",
    "How does natural language processing handle sarcasm and subtle cultural nuances?",
    "I am curious to explore how caching hierarchies in modern CPUs minimize latency.",
    "I want to dive into functional reactive programming and state streams.",
    "What is the mathematical proof that shows P equals or does not equal NP?",
    "I am eager to understand how cellular automata generate complex emergent behavior.",
    "How do WebAssembly runtimes achieve near-native performance inside browsers?",
    "I want to know more about neuromorphic computing chips that mimic biological neurons.",
    "I am curious about how high-frequency trading platforms execute trades in nanoseconds.",
    "I want to explore how topological data analysis extracts shape from point clouds.",
    "I wonder what breakthroughs are happening in room-temperature superconductivity.",
    "I am excited to experiment with fine-tuning open-source vision-language models.",
    "How does memory allocation work in garbage-collected languages like Go and Java?",
    "I want to investigate how swarm robotics coordinates without a central leader."
]

# 5. FRUSTRATED (blocked, angry, repeated failures, exasperated, high stress)
frustrated_samples = [
    "I have tried solving this programming problem many times, but nothing works.",
    "I have spent six hours trying to fix this segmentation fault and I am going crazy.",
    "Nothing compiles, every time I fix one error five new errors appear!",
    "I am so sick and tired of this obscure dependency conflict breaking my build.",
    "I followed the documentation step by step, yet the application still crashes instantly.",
    "I am tearing my hair out trying to figure out why this API returns a 500 error.",
    "I have rewritten this algorithm three times and it still fails the hidden test cases.",
    "This environment setup is a nightmare, nothing installs properly on my machine.",
    "I am so angry that my code worked yesterday and completely broke today with zero changes.",
    "I feel like throwing my laptop out the window because of this stupid pointer bug.",
    "I keep getting stuck on this one calculus problem and no explanation helps.",
    "The grading rubric is completely unfair and impossible to satisfy.",
    "I wasted my whole weekend on a single missing semicolon in this configuration file.",
    "Why is C++ memory management so unforgiving? I am completely fed up.",
    "My model training crashed at epoch 99 out of 100 with an out of memory error!",
    "I am so exasperated with this group project where nobody else writes any code.",
    "The professor gave us an assignment on material we were never taught in lecture.",
    "I am burnt out and furious that my pull request was rejected without explanation.",
    "No matter what hyperparameters I tweak, the validation loss refuses to decrease.",
    "This legacy codebase has zero comments and is a completely unreadable mess.",
    "I have re-read this paragraph ten times and I still cannot make any sense of it, it is driving me mad.",
    "I hate how difficult it is to configure webpack and babel for this frontend project.",
    "I have been debugging this concurrency deadlock for days and I am ready to quit.",
    "Why can't this package just work out of the box without fifteen breaking dependencies?",
    "I am hitting a brick wall with this math assignment and I am completely stressed out.",
    "Every single tutorial online uses deprecated code that no longer functions.",
    "I feel overwhelmed and angry because the deadline is tonight and my tests are failing.",
    "This simulator keeps crashing every single time I press run, it is totally broken.",
    "I am beyond frustrated with this cryptic error message that provides zero context.",
    "I spent all night fixing merge conflicts only for the remote branch to get overwritten.",
    "I am furious that my local environment works perfectly but production fails completely.",
    "This IDE keeps freezing and losing my unsaved changes, it is infuriating.",
    "I am fed up with trying to debug this asynchronous race condition that only happens sometimes.",
    "I don't understand why this formula doesn't match the textbook answer, it makes no sense!",
    "I am so sick of dealing with version mismatches between Python, CUDA, and PyTorch.",
    "Nothing I do produces the expected output and I am losing my patience.",
    "I have asked three teaching assistants and none of them know how to fix this bug.",
    "This framework is bloated, overcomplicated, and infuriating to work with.",
    "I am so stressed that I can't sleep because this project won't compile.",
    "I tried every solution on StackOverflow and none of them worked for my problem.",
    "I am sick of dealing with silent failures where the program exits with no error code.",
    "Why does this query take twenty minutes to execute on a tiny table? It is infuriating.",
    "I am tired of struggling through this broken course platform that constantly logs me out.",
    "I have reached my limit with this assignment and feel like giving up entirely.",
    "Everything is failing and I am at my absolute wits' end with this code."
]

# Build dataset with splits (80% train, 10% val, 10% test)
np.random.seed(42)

for emotion, sample_list in [
    ("BORED", bored_samples),
    ("CONFIDENT", confident_samples),
    ("CONFUSED", confused_samples),
    ("CURIOUS", curious_samples),
    ("FRUSTRATED", frustrated_samples)
]:
    # shuffle within class
    indices = np.random.permutation(len(sample_list))
    n_total = len(sample_list)
    n_train = int(0.80 * n_total)
    n_val = int(0.10 * n_total)
    
    for rank, idx in enumerate(indices):
        text = sample_list[idx]
        if rank < n_train:
            split = "train"
        elif rank < n_train + n_val:
            split = "val"
        else:
            split = "test"
        
        records.append({
            "text": text,
            "emotion": emotion,
            "split": split,
            "length": len(text.split())
        })

df = pd.DataFrame(records)
DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DATA_DIR, index=False)
print(f"Created dataset at {DATA_DIR} with {len(df)} samples.")
print(df["emotion"].value_counts())
print(df["split"].value_counts())
