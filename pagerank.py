import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000
THRESHOLD = 0.001


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_dist = {}

    links = corpus[page]
    num_links = len(links)
    if num_links != 0:

        prob_damp = (1 - damping_factor) / len(corpus)

        for html in corpus:

            if html in links:

                prob_link = damping_factor / num_links
                prob_dist[html] = prob_link + prob_damp

            else:
                prob_dist[html] = prob_damp

    else:
        for html in corpus:
            prob_dist[html] = 1 / len(corpus)

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    rank = {}
    for html in corpus:
        rank[html] = 0

    next_page = random.choice(list(corpus))
    rank[next_page] += 1

    for i in range(n - 1):
        prob_dist = transition_model(corpus, next_page, DAMPING)
        next_page = random.choices(list(prob_dist.keys()), weights=list(prob_dist.values()))[0]
        rank[next_page] += 1

    for page in rank:
        rank[page] = rank[page] / n

    return rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)

    rank = {}
    new = {}
    link_to = {}

    for html in corpus:
        rank[html] = 1 / N
        link_to[html] = []

    for html in corpus:
        for link in corpus[html]:
            link_to[link].append(html)

    while True:
        for html in corpus:
            sum = 0
            num_links_p = len(corpus[html])
            for i in link_to[html]:
                num_links_i = len(corpus[i])
                sum += rank[i] / num_links_i

            new[html] = ((1 - damping_factor) / N) + (damping_factor * sum)

        max_diff = 0
        for html in rank:
            diff = abs(rank[html] - new[html])
            if diff > max_diff:
                max_diff = diff

        if max_diff < THRESHOLD:
            return rank

        rank = new.copy()

if __name__ == "__main__":
    main()
