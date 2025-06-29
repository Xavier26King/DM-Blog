# How Google Finds What You're Looking For: PageRank & Linear Algebra
*A comprehensive mathematical exploration of the algorithm that built the modern web*

PageRank represents one of the most elegant applications of linear algebra to real-world problems, transforming web search from primitive keyword matching into the sophisticated authority-based ranking we rely on today. This algorithm, developed by Larry Page and Sergey Brin in 1996, interprets the web as a massive directed graph and applies fundamental theorems from matrix theory to determine page importance.

## The Mathematical Foundation: From Web Graph to Linear System

### Constructing the Web as a Matrix

The web's hyperlink structure forms a natural directed graph where pages are vertices and links are edges. This graph translates directly into mathematical objects through the **adjacency matrix** A, where A[i,j] = 1 if page j links to page i, and 0 otherwise.

For a simple 4-page network where:
- Page A links to B and C  
- Page B links to C
- Page C links to A and D
- Page D links to A

The adjacency matrix becomes:

```
A = [[0, 0, 1, 1],    # Page A receives from C, D
     [1, 0, 0, 0],    # Page B receives from A
     [1, 1, 0, 0],    # Page C receives from A, B  
     [0, 0, 1, 0]]    # Page D receives from C
```

### The Hyperlink Matrix: Normalizing Transition Probabilities

Raw link counts don't capture relative importance—a page linking to 100 others shouldn't give each the same weight as a page linking to just one. The **hyperlink matrix** H addresses this by column-normalizing A:

$$H_{ij} = \frac{A_{ij}}{\sum_{k=1}^n A_{kj}}$$

This creates a **column stochastic matrix** where each column sums to 1, representing transition probabilities in a Markov chain. In our example:

```
H = [[0.0, 0.0, 0.5, 1.0],    # Transition probabilities
     [0.5, 0.0, 0.0, 0.0],    # from each page
     [0.5, 1.0, 0.0, 0.0],    # (columns sum to 1)
     [0.0, 0.0, 0.5, 0.0]]
```

### The Google Matrix: Ensuring Convergence

The hyperlink matrix H often fails mathematical requirements for guaranteed convergence. Two critical problems emerge:

1. **Dangling nodes** (pages with no outlinks) create zero columns
2. **Disconnected components** prevent irreducibility

The **Google matrix** G solves both issues through the damping factor modification:

$$G = dH + (1-d)E$$

where d = 0.85 (the damping factor) and E is an n×n matrix with all entries equal to 1/n.

```python
import numpy as np

# Using our 4-page example
d = 0.85
n = 4
E = np.ones((n, n)) / n
G = d * H + (1 - d) * E

# Result:
G = [[0.0375, 0.0375, 0.4625, 0.8875],
     [0.4625, 0.0375, 0.0375, 0.0375], 
     [0.4625, 0.8875, 0.0375, 0.0375],
     [0.0375, 0.0375, 0.4625, 0.0375]]
```

## The Eigenvector Connection: PageRank as Linear Algebra

### Formulating the Eigenvector Problem

PageRank seeks a vector r where each component r[i] represents page i's importance[3][6]. The fundamental insight: **PageRank is the principal eigenvector of the Google matrix**.

The mathematical relationship:

$$\mathbf{r} = G\mathbf{r}$$

This is the standard eigenvalue problem with λ = 1. The Perron-Frobenius theorem guarantees that G has a unique positive eigenvector corresponding to eigenvalue 1, and this eigenvector represents the stationary distribution of the associated Markov chain.

### The Power Method: Iterative Solution

Computing eigenvectors directly for billion-node graphs is computationally prohibitive. Instead, PageRank employs the **power method**:

$$\mathbf{r}^{(k+1)} = G\mathbf{r}^{(k)}$$

Starting with uniform distribution r⁽⁰⁾ = [1/n, 1/n, ..., 1/n]:

```python
# Power iteration implementation
r = np.ones(n) / n  # Initial uniform distribution

for k in range(iterations):
    r_new = G @ r
    if np.linalg.norm(r_new - r) < tolerance:
        break
    r = r_new
```

For our 4-page example, convergence occurs within ~10 iterations:

| Iteration | Page A | Page B | Page C | Page D |
|-----------|--------|--------|--------|--------|
| 0         | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| 1         | 0.3562 | 0.1437 | 0.3562 | 0.1437 |
| 5         | 0.3256 | 0.1744 | 0.3256 | 0.1744 |
| 10        | 0.3246 | 0.1754 | 0.3246 | 0.1754 |

Pages A and C achieve higher rankings due to receiving more authoritative links.

## Mathematical Guarantees: Convergence Theory

### Perron-Frobenius Theorem Application

The Google matrix G satisfies three crucial properties that guarantee convergence:

1. **Stochastic**: Each column sums to 1
2. **Irreducible**: Every page can reach every other page  
3. **Aperiodic**: No fixed-period cycles exist

Under these conditions, the Perron-Frobenius theorem guarantees:
- Unique eigenvalue λ = 1 with multiplicity 1
- All other eigenvalues satisfy |λᵢ| < 1
- Unique positive eigenvector (the PageRank vector)

### Convergence Rate Analysis

The power method's convergence rate depends on the **second-largest eigenvalue** λ₂:

$$\text{Convergence rate} = \left|\frac{\lambda_2}{\lambda_1}\right| = |\lambda_2|$$

For the Google matrix with damping factor d = 0.85, theoretical analysis shows |λ₂| ≤ 0.85, ensuring rapid convergence. In practice, 50-100 iterations achieve sufficient precision for web-scale computation.

## The Damping Factor: Balancing Authority and Democracy

### Mathematical Interpretation

The damping factor d represents the probability that a random surfer follows links rather than jumping to a random page. This creates a balance between:

- **Authority flow** (d → 1): Rankings dominated by link structure
- **Democratic distribution** (d → 0): All pages receive equal importance

### Impact on Final Rankings

Different damping factors produce measurably different PageRank distributions:

| Damping Factor | Page A | Page B | Page C | Page D |
|----------------|--------|--------|--------|--------|
| 0.50          | 0.3000 | 0.2000 | 0.3000 | 0.2000 |
| 0.70          | 0.3148 | 0.1852 | 0.3148 | 0.1852 |
| 0.85          | 0.3246 | 0.1754 | 0.3246 | 0.1754 |
| 0.95          | 0.3305 | 0.1695 | 0.3305 | 0.1695 |

Higher damping factors increasingly concentrate authority in well-connected pages.

## A Deeper Mathematical Intuition: PageRank as "Probability Flow"

Most textbook derivations treat PageRank strictly as an eigen-problem, yet the **random-surfer** metaphor offers an intuitive *probability-flow* perspective.  Imagine one unit of probability fluid continuously circulating through the web graph.  At every time step  
1. **Link transition** (weight **d**): each page pours its current probability mass equally through its out-links.  
2. **Teleport jump** (weight **1 – d**): simultaneously, a thin mist of probability condenses uniformly on *all* pages.

Because the Google matrix \(G = dH + (1-d)E\) simply re-allocates probability without creating or destroying it, the Markov process is **conservative**: the total mass stays 1.  The steady-state vector \(\mathbf r\) is therefore nothing more than the long-run *distribution* of that fluid[15][16].  Large entries of \(\mathbf r\) correspond to *sinks* that receive persistent inflow from many well-connected sources.

Two quick corollaries become transparent in this picture:
* **Dangling stability** – a page with no out-links contributes *all* of its probability to the teleport pool, preventing masses from disappearing.
* **Second-eigenvalue speed** – the gap \(1-|\lambda_2|\) measures how fast transient eddies die out; a smaller gap means slower mixing.

---

## Toy Web: Six Interlocking Pages

We construct a miniature web of six pages that span three topical clusters.

| Page | Title | Theme | Out-links |
|------|-------------------|---------------|---------------|
| A | Python Tutorial Hub | programming | B, C, D |
| B | Basic Python Guide | programming | A, C |
| C | Advanced Python | programming | A, D, E |
| D | Data Science Blog | data-science | A, E |
| E | ML Algorithms | machine-learning | D |
| F | Random Blog | misc | A |

The directed graph translates into the adjacency matrix \(A\) below (rows = receivers, columns = senders):

```
       A  B  C  D  E  F
A  [ 0  1  1  1  0  1 ]
B  [ 1  0  0  0  0  0 ]
C  [ 1  1  0  0  0  0 ]
D  [ 1  0  1  0  1  0 ]
E  [ 0  0  1  1  0  0 ]
F  [ 0  0  0  0  0  0 ]
```

With damping factor \(d = 0.85\) the power method converges in **20 iterations**:

| Iter | A | B | C | D | E | F |
|------|-------|-------|-------|-------|-------|-------|
| 0 | 0.1667 | 0.1667 | 0.1667 | 0.1667 | 0.1667 | 0.1667 |
| 5 | 0.2574 | 0.0946 | 0.1373 | 0.2922 | 0.1934 | 0.0250 |
| 10 | 0.2519 | 0.0970 | 0.1378 | 0.2986 | 0.1898 | 0.0250 |
| 20 | **0.2527** | **0.0966** | **0.1377** | **0.2976** | **0.1904** | **0.0250** |

**Data Science Blog** (D) rises to the top despite having fewer direct in-links than A because highly ranked nodes repeatedly funnel authority into it — a tangible demonstration of *quality over quantity*.

---

## From PageRank to Actual Search Results

A real search engine blends **content relevance** with **link authority**.  Using the final PageRank vector \(r\) and a simple keyword-match score \(k\), we rank pages by
\[
S = 0.6\,k + 0.4\,r.
\]

### Query: *"python tutorial"*

| Rank | Page | k | r | Combined S |
|------|------|-----|-----|-----------|
| 1 | A | **1.000** | 0.253 | **0.701** |
| 2 | D | 0.500 | **0.298** | 0.419 |
| 3 | C | 0.500 | 0.138 | 0.355 |

Pure keyword matching would already place **A** first, but PageRank still matters: it breaks ties between B and C, pushing the more authoritative **C** above **B**.

### Query: *"machine learning"*

| Rank | Page | k | r | Combined S |
|------|------|-----|-----|-----------|
| 1 | D | **1.000** | **0.298** | **0.719** |
| 2 | E | **1.000** | 0.190 | 0.676 |
| 3 | A | 0.000 | 0.253 | 0.101 |

Here **PageRank is decisive**: pages D and E tie on keywords, but D outranks E thanks to higher incoming trust, reflecting the idea that “who recommends you” beats “how many recommend you.”

---

## Why This Matters

Even on a six-node toy graph, PageRank captures *global structural signals* that keyword statistics alone cannot.  It rewards pages plugged into reputable neighborhoods and penalizes orphan or spam pages, illustrating **the dual lens of topicality and authority** that underpins modern search.

# Calculating Keyword Relevance Scores (k): The Text Matching Component

## Understanding the 'k' Score in Search Engines

In the PageRank example, we saw how search results combine **content relevance** (k) with **link authority** (PageRank) using the formula:

\[
S = 0.6 \times k + 0.4 \times r
\]

But how exactly is the keyword relevance score **k** calculated? This section explains the mathematical foundations behind text matching and relevance scoring used in modern search engines.

---

## Text Preprocessing: The Foundation

Before any relevance calculation can begin, both the search query and document content must be preprocessed to create a standardized, comparable format. This involves several critical steps:

### 1. Tokenization
Breaking text into individual words (tokens):
```
"Python Tutorial Hub" → ["Python", "Tutorial", "Hub"]
```

### 2. Case Normalization
Converting all text to lowercase for consistent matching:
```
["Python", "Tutorial", "Hub"] → ["python", "tutorial", "hub"]
```

### 3. Stopword Removal
Eliminating common words that carry little semantic meaning:
```
"Learn Python programming with the best guide" 
→ ["learn", "python", "programming", "best", "guide"]
```
Common stopwords: *the, and, or, but, in, on, at, to, for, of, with, by, a, an, is, are, was, were*

### 4. Stemming/Lemmatization
Reducing words to their root forms:
- **Stemming**: `running, runs, ran → run`
- **Lemmatization**: `better → good, mice → mouse`

---

## Simple Keyword Matching (Used in Our Example)

The cleanest approach, which produces the exact k values shown in our PageRank example, uses **proportional keyword matching**:

### Algorithm
```python
def calculate_k_score(query, document):
    # 1. Preprocess both query and document
    query_words = set(preprocess(query))
    doc_words = set(preprocess(document))
    
    # 2. Find intersection of words
    matched_words = query_words ∩ doc_words
    
    # 3. Calculate match ratio
    match_ratio = len(matched_words) / len(query_words)
    
    # 4. Convert to discrete scores
    if match_ratio >= 1.0:
        return 1.000  # Perfect match (all query words found)
    elif match_ratio >= 0.5:
        return 0.500  # Partial match (some query words found)
    else:
        return 0.000  # Poor match (few/no query words found)
```

### Example Calculations

**Query: "python tutorial"**
- Query words: `{python, tutorial}`
- Total query words: 2

| Page | Document Content | Matched Words | Ratio | k Score |
|------|------------------|---------------|-------|---------|
| A | Python Tutorial Hub... | `{python, tutorial}` | 2/2 = 1.0 | **1.000** |
| B | Basic Python Guide... | `{python}` | 1/2 = 0.5 | **0.500** |
| C | Advanced Python... | `{python}` | 1/2 = 0.5 | **0.500** |
| D | Data Science Blog... | `{python}` | 1/2 = 0.5 | **0.500** |
| E | ML Algorithms... | `{}` | 0/2 = 0.0 | **0.000** |
| F | Random Blog... | `{}` | 0/2 = 0.0 | **0.000** |

**Query: "machine learning"**
- Query words: `{machine, learning}`
- Total query words: 2

| Page | Document Content | Matched Words | Ratio | k Score |
|------|------------------|---------------|-------|---------|
| A | Python Tutorial Hub... | `{}` | 0/2 = 0.0 | **0.000** |
| B | Basic Python Guide... | `{}` | 0/2 = 0.0 | **0.000** |
| C | Advanced Python... | `{}` | 0/2 = 0.0 | **0.000** |
| D | Data Science Blog... | `{machine, learning}` | 2/2 = 1.0 | **1.000** |
| E | ML Algorithms... | `{machine, learning}` | 2/2 = 1.0 | **1.000** |
| F | Random Blog... | `{}` | 0/2 = 0.0 | **0.000** |

---

## Advanced Relevance Scoring: TF-IDF

For more sophisticated relevance calculation, search engines typically use **TF-IDF** (Term Frequency-Inverse Document Frequency), which considers both how often terms appear and how rare they are across the corpus.

### Term Frequency (TF)
Measures how frequently a term appears in a document:
  
TF(t, d) = (Number of times term *t* appears in document *d*) / (Total number of terms in document *d*)

### Inverse Document Frequency (IDF)
IDF(t) = log(Total number of documents / Number of documents containing term *t*)

### TF-IDF Score 
TF-IDF(t, d) = TF(t, d) × IDF(t)

### Example TF-IDF Calculation

For our 6-page corpus and the term "tutorial":

**Term Frequency:**
- Page A: "tutorial" appears 1 time in 10 words → TF = 1/10 = 0.100
- Pages B-F: "tutorial" appears 0 times → TF = 0.000

**Inverse Document Frequency:**
- Total documents: 6
- Documents containing "tutorial": 1 (only Page A)
- IDF = log(6/1) = log(6) ≈ 1.792

**TF-IDF Scores:**
- Page A: 0.100 × 1.792 = **0.179**
- Pages B-F: 0.000 × 1.792 = **0.000**

**Final Query Score:**
For query "python tutorial" against Page A:
- "python": TF-IDF = 0.081
- "tutorial": TF-IDF = 0.179
- **Combined score = (0.081 + 0.179) / 2 = 0.130**

---

## Modern Search Engine Enhancements

### 1. BM25 Algorithm
An improvement over TF-IDF that addresses document length bias:

\[
\text{BM25}(t,d) = IDF(t) \times \frac{TF(t,d) \times (k_1 + 1)}{TF(t,d) + k_1 \times (1 - b + b \times \frac{|d|}{avgdl})}
\]

Where:
- k₁ controls term frequency saturation (typically 1.2-2.0)
- b controls length normalization (typically 0.75)
- |d| is document length, avgdl is average document length

### 2. Semantic Matching
Modern systems use vector embeddings to match semantically similar terms:
- "car" matches "automobile"
- "ML" matches "machine learning"
- "AI" matches "artificial intelligence"

### 3. Query Expansion
Search engines expand queries with synonyms and related terms:
- Query: "python programming"
- Expanded: "python programming coding development scripting"

### 4. Field Weighting
Different document sections receive different importance weights:
- Title: 3.0× weight
- Headings: 2.0× weight
- Body text: 1.0× weight
- Meta tags: 0.5× weight

---

## Practical Implementation

Here's a complete Python implementation of keyword relevance scoring:

```python
import re
import math
from collections import Counter

def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase and extract words
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    # Remove stopwords
    stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    return [w for w in words if w not in stopwords]

def calculate_tfidf(documents):
    """Calculate TF-IDF scores for all documents"""
    # Preprocess all documents
    processed_docs = {doc_id: preprocess_text(text) for doc_id, text in documents.items()}
    
    # Calculate TF scores
    tf_scores = {}
    for doc_id, words in processed_docs.items():
        word_count = Counter(words)
        total_words = len(words)
        tf_scores[doc_id] = {word: count/total_words for word, count in word_count.items()}
    
    # Calculate IDF scores
    total_docs = len(documents)
    all_words = set(word for words in processed_docs.values() for word in words)
    idf_scores = {}
    for word in all_words:
        docs_containing = sum(1 for words in processed_docs.values() if word in words)
        idf_scores[word] = math.log(total_docs / docs_containing)
    
    # Calculate TF-IDF scores
    tfidf_scores = {}
    for doc_id in documents:
        tfidf_scores[doc_id] = {}
        for word in tf_scores[doc_id]:
            tfidf_scores[doc_id][word] = tf_scores[doc_id][word] * idf_scores[word]
    
    return tfidf_scores

def query_relevance(query, doc_tfidf):
    """Calculate relevance score for a query against a document"""
    query_words = preprocess_text(query)
    
    # Sum TF-IDF scores for all query words found in document
    total_score = sum(doc_tfidf.get(word, 0) for word in query_words)
    
    # Normalize by query length
    return total_score / len(query_words) if query_words else 0
```

---

## From Toy Example to Google Scale

While our 6-page example uses simple keyword matching, Google's production system scales these principles to billions of pages using:

1. **Distributed Computing**: TF-IDF calculations across massive server clusters
2. **Approximate Algorithms**: Probabilistic data structures for efficiency
3. **Machine Learning**: Neural networks for semantic understanding
4. **Real-time Updates**: Incremental index updates as new content appears
5. **Personalization**: User-specific relevance adjustments
6. **Context Awareness**: Location, device, and search history influence scoring

The fundamental principle remains the same: **measure how well document content matches user intent**, whether through simple keyword overlap or sophisticated semantic analysis.

This text relevance component (k), combined with authority signals like PageRank (r), creates the foundation of modern web search – turning the vast, unstructured web into a navigable information space.
## Computational Complexity and Scale

### Matrix Operations at Web Scale

Modern web graphs contain approximately 10¹² pages with ~10 outlinks each. This creates:
- Matrix size: 10¹² × 10¹² entries  
- Sparse structure: ~10¹³ non-zero entries
- Memory requirements: Terabytes for full matrix storage

### Sparse Matrix Optimization

The Google matrix's construction as G = dH + (1-d)E avoids explicit storage of the dense component. Instead, the matrix-vector multiplication becomes:

$$G\mathbf{r} = d(H\mathbf{r}) + \frac{(1-d)}{n}\mathbf{e}$$

where **e** is the vector of ones. This reduces each iteration from O(n²) to O(10n) operations.

### Production Implementation

Google's production PageRank system achieves remarkable scalability:
- 38 billion vertices, 3.1 trillion edges
- 34.4 seconds per iteration
- Distributed computation across thousands of machines
- Custom communication protocols minimize bandwidth requirements

## Advanced Mathematical Insights

### Personalized PageRank

The standard formulation extends to **personalized PageRank** by replacing the uniform jump vector with user-specific preferences:

$$G = dH + (1-d)\mathbf{v}\mathbf{e}^T$$

where **v** represents personalization weights, enabling topic-specific rankings.

### Matrix Analysis Properties

The Google matrix exhibits fascinating spectral properties:
- **Spectral radius**: ρ(G) = 1
- **Condition number**: Well-conditioned due to damping
- **Rank**: Full rank (irreducible construction)
- **Eigenvalue distribution**: Real dominant eigenvalue, complex subdominant eigenvalues

### Sensitivity Analysis

PageRank demonstrates robust stability under small perturbations. The algorithm's mathematical structure ensures that minor changes in link structure produce proportionally small changes in rankings, critical for handling the web's dynamic nature.

## Real-World Extensions Beyond Web Search

### Citation Networks  

Academic papers form directed graphs through citations, enabling PageRank-based impact metrics. The h-index and other bibliometrics pale beside eigenvector centrality measures for identifying influential research.

### Social Network Analysis

Social media platforms employ PageRank variants to identify influential users and content. Twitter's original "Who to Follow" algorithm directly implemented personalized PageRank with user interaction graphs.

### Recommendation Systems

Netflix, Amazon, and Spotify use PageRank principles to identify "influential" users whose preferences predict broader appeal. These systems construct bipartite graphs connecting users to content, then apply PageRank to both node types.

## The Algorithm That Built Modern Search

PageRank's mathematical elegance transformed web search from keyword matching to authority-based ranking. The algorithm's success stems from its theoretical foundations:

1. **Rigorous convergence guarantees** via Perron-Frobenius theory
2. **Efficient computation** through sparse matrix methods  
3. **Robust behavior** under web-scale perturbations
4. **Intuitive interpretation** as random surfer stationary distribution

While Google's current ranking algorithm incorporates hundreds of factors beyond PageRank, the fundamental insight—that link structure reveals authority—remains central to modern information retrieval. The mathematical framework pioneered by Page and Brin continues to influence everything from social network analysis to recommendation systems, demonstrating the profound impact of applying rigorous linear algebra to real-world problems.

The beauty of PageRank lies not in its computational complexity, but in its mathematical simplicity: the most important pages are those deemed important by other important pages, expressed through the elegant language of eigenvectors and Markov chains. This recursive definition, seemingly circular in natural language, finds precise mathematical expression through matrix theory—a testament to the power of linear algebra in solving complex, real-world problems.

## Implementation Code Example

```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

def pagerank(adjacency_matrix, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    """
    Compute PageRank using the power method.
    
    Parameters:
    - adjacency_matrix: n×n sparse matrix where A[i,j] = 1 if j links to i
    - damping_factor: probability of following links (default 0.85)
    - max_iterations: maximum power iterations
    - tolerance: convergence threshold
    
    Returns:
    - pagerank_vector: normalized PageRank scores
    """
    n = adjacency_matrix.shape[0]
    
    # Create hyperlink matrix (column stochastic)
    out_degrees = np.array(adjacency_matrix.sum(axis=0)).flatten()
    out_degrees[out_degrees == 0] = 1  # Handle dangling nodes
    
    H = adjacency_matrix / out_degrees
    
    # Initialize uniform PageRank vector
    r = np.ones(n) / n
    
    # Power iteration
    for iteration in range(max_iterations):
        r_new = damping_factor * (H @ r) + (1 - damping_factor) / n
        
        # Check convergence
        if np.linalg.norm(r_new - r) < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
            
        r = r_new
    
    return r / r.sum()  # Ensure normalization

# Example usage with 4-page network
A = np.array([[0, 0, 1, 1],
              [1, 0, 0, 0], 
              [1, 1, 0, 0],
              [0, 0, 1, 0]])

pagerank_scores = pagerank(A)
print("PageRank scores:", pagerank_scores)
```

This mathematical exploration reveals PageRank as far more than a search algorithm—it represents a fundamental breakthrough in using linear algebra to extract meaning from network structure, with applications spanning academic citation analysis to social media influence measurement[30][32]. The algorithm's enduring relevance demonstrates the power of mathematical theory in solving complex computational challenges at unprecedented scale.

## References

1. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank Citation Ranking: Bringing Order to the Web*. Stanford InfoLab Technical Report.  
   *[This is the seminal paper that introduced the PageRank algorithm. It is referenced in most discussions of web search and PageRank theory.]*

2. Langville, A. N., & Meyer, C. D. (2012). *Google’s PageRank and Beyond: The Science of Search Engine Rankings*. Princeton University Press.  
   *[This book provides a comprehensive mathematical overview of PageRank and related algorithms.]*

3. Brin, S., & Page, L. (1998). *The Anatomy of a Large-Scale Hypertextual Web Search Engine*. In *Proceedings of the 7th International World Wide Web Conference (WWW7)*, 107–117.  
   *[This paper gives the foundational technical details behind Google’s early search engine.]*

4. Meyer, C. D. (2000). *Matrix Analysis and Applied Linear Algebra*. Society for Industrial and Applied Mathematics (SIAM).  
   *[Used for the linear algebra and matrix theory underpinnings of PageRank.]*

5. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*. Johns Hopkins University Press.  
   *[A standard reference for matrix computations and eigenvalue problems.]*

6. Haveliwala, T. H. (2002). *Topic-Sensitive PageRank*. In *Proceedings of the 11th International World Wide Web Conference (WWW2002)*, 517–526.  
   *[Discusses personalized PageRank and its applications.]*

7. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.  
   *[Standard reference for information retrieval, including TF-IDF and BM25.]*

8. Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.  
   *[Details on BM25 and advanced relevance scoring.]*

9. Meyer, C. D. (1989). *Stochastic Complementarity, Uncoupling Markov Chains, and the Theory of Nearly Reducible Systems*. *SIAM Review*, 31(2), 240–272.  
   *[Background on Markov chains and irreducibility.]*

10. Seneta, E. (2006). *Non-negative Matrices and Markov Chains*. Springer Series in Statistics.  
    *[For Perron-Frobenius theory and stochastic matrices.]*
