#!/usr/bin/env python3
"""Build the Anki-NeetCode Pro deck (Anki-NeetCode-Pro.apkg).

This is the committed, runnable version of the build cells in main.ipynb.
It regenerates the per-card test cases from the authoritative LeetCodeDataset
`test` field and writes the apkg plus inspectable test files under data/test-code/.

Run:  python3 build_pro_deck.py
"""
import json, re, ast, gzip, zipfile, sqlite3, tempfile, os, shutil
import genanki
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

DATASET = "data/LeetCodeDataset-v0.3.1-train.jsonl.gz"
NEETCODE_LIST = "neetcode-150-list.json"
APKG_OUT = "Anki-NeetCode-Pro.apkg"
TEST_CODE_DIR = "data/test-code"

# Premium problems are stored locally because LeetCode does not expose their
# full public question payload through the same unauthenticated path.
paidOnly = ['alien-dictionary', 'encode-and-decode-strings', 'graph-valid-tree',
            'meeting-rooms-ii', 'meeting-rooms',
            'number-of-connected-components-in-an-undirected-graph', 'walls-and-gates']

CODE_SNIPPET_OVERRIDES = {
    'alien-dictionary': (
        "class Solution:\n"
        "    def alienOrder(self, words: List[str]) -> str:\n"
        "        \n"
    ),
    'encode-and-decode-strings': (
        "class Solution:\n"
        "    def encode(self, strs: List[str]) -> str:\n"
        "        \n"
        "\n"
        "    def decode(self, s: str) -> List[str]:\n"
        "        \n"
    ),
    'graph-valid-tree': (
        "class Solution:\n"
        "    def validTree(self, n: int, edges: List[List[int]]) -> bool:\n"
        "        \n"
    ),
    'meeting-rooms': (
        "class Solution:\n"
        "    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:\n"
        "        \n"
    ),
    'meeting-rooms-ii': (
        "class Solution:\n"
        "    def minMeetingRooms(self, intervals: List[List[int]]) -> int:\n"
        "        \n"
    ),
    'number-of-connected-components-in-an-undirected-graph': (
        "class Solution:\n"
        "    def countComponents(self, n: int, edges: List[List[int]]) -> int:\n"
        "        \n"
    ),
    'walls-and-gates': (
        "class Solution:\n"
        "    def wallsAndGates(self, rooms: List[List[int]]) -> None:\n"
        "        \n"
    ),
}

BASIC_PROMPT = (
    "from typing import *\n"
    "from collections import *\n"
    "from heapq import *\n"
    "from bisect import *\n\n"
)

TREE_PROMPT = BASIC_PROMPT + (
    "class TreeNode:\n"
    "    def __init__(self, x):\n"
    "        self.val = x\n"
    "        self.left = None\n"
    "        self.right = None\n\n"
)

RANDOM_LIST_PROMPT = BASIC_PROMPT + (
    "class Node:\n"
    "    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):\n"
    "        self.val = int(x)\n"
    "        self.next = next\n"
    "        self.random = random\n\n"
)

GRAPH_PROMPT = BASIC_PROMPT + (
    "class Node:\n"
    "    def __init__(self, val = 0, neighbors = None):\n"
    "        self.val = val\n"
    "        self.neighbors = neighbors if neighbors is not None else []\n\n"
)


def _object_test_code(slug, cases):
    return "# Test cases for %s\n_test_cases = %r\n\n" % (slug, cases) + (
        "def check(cls):\n"
        "    passed = 0\n"
        "    failed = []\n"
        "    for case_num, (ops, args, expected) in enumerate(_test_cases, 1):\n"
        "        obj = None\n"
        "        actual = []\n"
        "        try:\n"
        "            for op, arg in zip(ops, args):\n"
        "                if obj is None:\n"
        "                    obj = cls(*arg)\n"
        "                    actual.append(None)\n"
        "                else:\n"
        "                    actual.append(getattr(obj, op)(*arg))\n"
        "            assert actual == expected\n"
        "            passed += 1\n"
        "        except AssertionError:\n"
        "            failed.append((case_num, expected, actual, 'wrong answer'))\n"
        "        except Exception as e:\n"
        "            failed.append((case_num, expected, actual, type(e).__name__ + ': ' + str(e)))\n"
        "    total = len(_test_cases)\n"
        "    print(f'{passed}/{total} tests passed')\n"
        "    if failed:\n"
        "        print(f'\\n{len(failed)} test(s) failed:')\n"
        "        for case_num, expected, actual, err in failed:\n"
        "            print(f'  Test #{case_num}: {err}')\n"
        "            print(f'    Expected: {expected}')\n"
        "            print(f'    Got: {actual}')\n"
        "    else:\n"
        "        print('All tests passed! \\u2713')\n"
        "    return not failed\n"
    )


ENCODE_DECODE_TEST_CODE = (
    "# Test cases for encode-and-decode-strings\n"
    "_encode_decode_cases = [\n"
    "    ['Hello', 'World'],\n"
    "    [''],\n"
    "    ['neet', 'code', 'love', 'you'],\n"
    "    ['we', 'say', ':', 'yes'],\n"
    "    ['#', '1#2', '', 'spaces are valid'],\n"
    "]\n\n"
    "def check(codec):\n"
    "    passed = 0\n"
    "    failed = []\n"
    "    for i, strs in enumerate(_encode_decode_cases, 1):\n"
    "        encoded = decoded = None\n"
    "        try:\n"
    "            encoded = codec.encode(strs)\n"
    "            decoded = codec.decode(encoded)\n"
    "            assert decoded == strs\n"
    "            passed += 1\n"
    "        except AssertionError:\n"
    "            failed.append((i, strs, decoded, 'wrong answer'))\n"
    "        except Exception as e:\n"
    "            failed.append((i, strs, decoded, type(e).__name__ + ': ' + str(e)))\n"
    "    total = len(_encode_decode_cases)\n"
    "    print(f'{passed}/{total} tests passed')\n"
    "    if failed:\n"
    "        print(f'\\n{len(failed)} test(s) failed:')\n"
    "        for i, strs, decoded, err in failed:\n"
    "            print(f'  Test #{i}: {err}')\n"
    "            print(f'    Input: {strs}')\n"
    "            print(f'    Decoded: {decoded}')\n"
    "    else:\n"
    "        print('All tests passed! \\u2713')\n"
    "    return not failed\n"
)

COPY_RANDOM_LIST_TEST_CODE = (
    "# Test cases for copy-list-with-random-pointer\n"
    "_test_cases = [\n"
    "    [[7, None], [13, 0], [11, 4], [10, 2], [1, 0]],\n"
    "    [[1, 1], [2, 1]],\n"
    "    [],\n"
    "]\n\n"
    "def _build_random_list(items):\n"
    "    nodes = [Node(v) for v, _ in items]\n"
    "    for i, node in enumerate(nodes[:-1]):\n"
    "        node.next = nodes[i + 1]\n"
    "    for node, (_, random_index) in zip(nodes, items):\n"
    "        node.random = None if random_index is None else nodes[random_index]\n"
    "    return (nodes[0] if nodes else None), nodes\n\n"
    "def _dump_random_list(head):\n"
    "    nodes = []\n"
    "    cur = head\n"
    "    while cur:\n"
    "        nodes.append(cur)\n"
    "        cur = cur.next\n"
    "    index = {node: i for i, node in enumerate(nodes)}\n"
    "    return [[node.val, None if node.random is None else index.get(node.random)] for node in nodes]\n\n"
    "def check(candidate):\n"
    "    passed = 0\n"
    "    failed = []\n"
    "    for i, items in enumerate(_test_cases, 1):\n"
    "        head, original_nodes = _build_random_list(items)\n"
    "        try:\n"
    "            cloned = candidate(head)\n"
    "            clone_nodes = []\n"
    "            cur = cloned\n"
    "            while cur:\n"
    "                clone_nodes.append(cur)\n"
    "                cur = cur.next\n"
    "            assert _dump_random_list(cloned) == items\n"
    "            assert all(node not in original_nodes for node in clone_nodes)\n"
    "            assert all(node.random is None or node.random not in original_nodes for node in clone_nodes)\n"
    "            passed += 1\n"
    "        except AssertionError:\n"
    "            failed.append((i, 'wrong answer'))\n"
    "        except Exception as e:\n"
    "            failed.append((i, type(e).__name__ + ': ' + str(e)))\n"
    "    print(f'{passed}/{len(_test_cases)} tests passed')\n"
    "    if failed:\n"
    "        print(f'\\n{len(failed)} test(s) failed:')\n"
    "        for i, err in failed:\n"
    "            print(f'  Test #{i}: {err}')\n"
    "    else:\n"
    "        print('All tests passed! \\u2713')\n"
    "    return not failed\n"
)

BST_LCA_TEST_CODE = (
    "# Test cases for lowest-common-ancestor-of-a-binary-search-tree\n"
    "_test_cases = [\n"
    "    ([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5], 2, 8, 6),\n"
    "    ([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5], 2, 4, 2),\n"
    "    ([2, 1], 2, 1, 2),\n"
    "]\n\n"
    "def _build_tree(values):\n"
    "    if not values:\n"
    "        return None, {}\n"
    "    nodes = [None if v is None else TreeNode(v) for v in values]\n"
    "    kids = nodes[::-1]\n"
    "    root = kids.pop()\n"
    "    for node in nodes:\n"
    "        if node:\n"
    "            if kids:\n"
    "                node.left = kids.pop()\n"
    "            if kids:\n"
    "                node.right = kids.pop()\n"
    "    return root, {node.val: node for node in nodes if node is not None}\n\n"
    "def check(candidate):\n"
    "    passed = 0\n"
    "    failed = []\n"
    "    for i, (values, p_val, q_val, expected) in enumerate(_test_cases, 1):\n"
    "        try:\n"
    "            root, nodes = _build_tree(values)\n"
    "            result = candidate(root, nodes[p_val], nodes[q_val])\n"
    "            assert result is not None and result.val == expected\n"
    "            passed += 1\n"
    "        except AssertionError:\n"
    "            failed.append((i, expected, None if 'result' not in locals() or result is None else result.val, 'wrong answer'))\n"
    "        except Exception as e:\n"
    "            failed.append((i, expected, None, type(e).__name__ + ': ' + str(e)))\n"
    "    print(f'{passed}/{len(_test_cases)} tests passed')\n"
    "    if failed:\n"
    "        print(f'\\n{len(failed)} test(s) failed:')\n"
    "        for i, expected, got, err in failed:\n"
    "            print(f'  Test #{i}: {err}; expected {expected}, got {got}')\n"
    "    else:\n"
    "        print('All tests passed! \\u2713')\n"
    "    return not failed\n"
)

SERIALIZE_TREE_TEST_CODE = (
    "# Test cases for serialize-and-deserialize-binary-tree\n"
    "_test_cases = [[1, 2, 3, None, None, 4, 5], [], [1], [1, 2]]\n\n"
    "def _build_tree(values):\n"
    "    if not values:\n"
    "        return None\n"
    "    nodes = [None if v is None else TreeNode(v) for v in values]\n"
    "    kids = nodes[::-1]\n"
    "    root = kids.pop()\n"
    "    for node in nodes:\n"
    "        if node:\n"
    "            if kids:\n"
    "                node.left = kids.pop()\n"
    "            if kids:\n"
    "                node.right = kids.pop()\n"
    "    return root\n\n"
    "def _tree_to_level(root):\n"
    "    out = []\n"
    "    q = deque([root]) if root else deque()\n"
    "    while q:\n"
    "        node = q.popleft()\n"
    "        if node is None:\n"
    "            out.append(None)\n"
    "            continue\n"
    "        out.append(node.val)\n"
    "        q.append(node.left)\n"
    "        q.append(node.right)\n"
    "    while out and out[-1] is None:\n"
    "        out.pop()\n"
    "    return out\n\n"
    "def check(codec):\n"
    "    passed = 0\n"
    "    failed = []\n"
    "    for i, values in enumerate(_test_cases, 1):\n"
    "        try:\n"
    "            root = _build_tree(values)\n"
    "            result = codec.deserialize(codec.serialize(root))\n"
    "            got = _tree_to_level(result)\n"
    "            assert got == values\n"
    "            passed += 1\n"
    "        except AssertionError:\n"
    "            failed.append((i, values, got, 'wrong answer'))\n"
    "        except Exception as e:\n"
    "            failed.append((i, values, None, type(e).__name__ + ': ' + str(e)))\n"
    "    print(f'{passed}/{len(_test_cases)} tests passed')\n"
    "    if failed:\n"
    "        print(f'\\n{len(failed)} test(s) failed:')\n"
    "        for i, expected, got, err in failed:\n"
    "            print(f'  Test #{i}: {err}; expected {expected}, got {got}')\n"
    "    else:\n"
    "        print('All tests passed! \\u2713')\n"
    "    return not failed\n"
)

CLONE_GRAPH_TEST_CODE = (
    "# Test cases for clone-graph\n"
    "_test_cases = [\n"
    "    [[2, 4], [1, 3], [2, 4], [1, 3]],\n"
    "    [[]],\n"
    "    [],\n"
    "]\n\n"
    "def _build_graph(adj):\n"
    "    if not adj:\n"
    "        return None, []\n"
    "    nodes = [Node(i + 1) for i in range(len(adj))]\n"
    "    for node, neighbors in zip(nodes, adj):\n"
    "        node.neighbors = [nodes[val - 1] for val in neighbors]\n"
    "    return nodes[0], nodes\n\n"
    "def _collect(node):\n"
    "    if node is None:\n"
    "        return []\n"
    "    seen = {}\n"
    "    q = deque([node])\n"
    "    while q:\n"
    "        cur = q.popleft()\n"
    "        if cur.val in seen:\n"
    "            continue\n"
    "        seen[cur.val] = cur\n"
    "        q.extend(cur.neighbors)\n"
    "    return [seen[k] for k in sorted(seen)]\n\n"
    "def _dump_graph(node):\n"
    "    nodes = _collect(node)\n"
    "    return [sorted(neighbor.val for neighbor in node.neighbors) for node in nodes]\n\n"
    "def check(candidate):\n"
    "    passed = 0\n"
    "    failed = []\n"
    "    for i, adj in enumerate(_test_cases, 1):\n"
    "        root, original_nodes = _build_graph(adj)\n"
    "        try:\n"
    "            cloned = candidate(root)\n"
    "            clone_nodes = _collect(cloned)\n"
    "            assert _dump_graph(cloned) == [sorted(n) for n in adj]\n"
    "            assert all(node not in original_nodes for node in clone_nodes)\n"
    "            passed += 1\n"
    "        except AssertionError:\n"
    "            failed.append((i, 'wrong answer'))\n"
    "        except Exception as e:\n"
    "            failed.append((i, type(e).__name__ + ': ' + str(e)))\n"
    "    print(f'{passed}/{len(_test_cases)} tests passed')\n"
    "    if failed:\n"
    "        print(f'\\n{len(failed)} test(s) failed:')\n"
    "        for i, err in failed:\n"
    "            print(f'  Test #{i}: {err}')\n"
    "    else:\n"
    "        print('All tests passed! \\u2713')\n"
    "    return not failed\n"
)

PROBLEM_OVERRIDES = {
    'clone-graph': {
        'prompt': GRAPH_PROMPT,
        'entry_point': "Solution().cloneGraph",
        'test_code': CLONE_GRAPH_TEST_CODE,
    },
    'copy-list-with-random-pointer': {
        'prompt': RANDOM_LIST_PROMPT,
        'entry_point': "Solution().copyRandomList",
        'test_code': COPY_RANDOM_LIST_TEST_CODE,
    },
    'design-add-and-search-words-data-structure': {
        'prompt': BASIC_PROMPT,
        'entry_point': "WordDictionary",
        'test_code': _object_test_code('design-add-and-search-words-data-structure', [
            (['WordDictionary', 'addWord', 'addWord', 'addWord', 'search', 'search', 'search', 'search'],
             [[], ['bad'], ['dad'], ['mad'], ['pad'], ['bad'], ['.ad'], ['b..']],
             [None, None, None, None, False, True, True, True]),
        ]),
    },
    'design-twitter': {
        'prompt': BASIC_PROMPT,
        'entry_point': "Twitter",
        'test_code': _object_test_code('design-twitter', [
            (['Twitter', 'postTweet', 'getNewsFeed', 'follow', 'postTweet', 'getNewsFeed', 'unfollow', 'getNewsFeed'],
             [[], [1, 5], [1], [1, 2], [2, 6], [1], [1, 2], [1]],
             [None, None, [5], None, None, [6, 5], None, [5]]),
        ]),
    },
    'detect-squares': {
        'prompt': BASIC_PROMPT,
        'entry_point': "DetectSquares",
        'test_code': _object_test_code('detect-squares', [
            (['DetectSquares', 'add', 'add', 'add', 'count', 'count', 'add', 'count'],
             [[], [[3, 10]], [[11, 2]], [[3, 2]], [[11, 10]], [[14, 8]], [[11, 2]], [[11, 10]]],
             [None, None, None, None, 1, 0, None, 2]),
        ]),
    },
    'encode-and-decode-strings': {
        'prompt': BASIC_PROMPT,
        'entry_point': "Solution()",
        'test_code': ENCODE_DECODE_TEST_CODE,
    },
    'find-median-from-data-stream': {
        'prompt': BASIC_PROMPT,
        'entry_point': "MedianFinder",
        'test_code': _object_test_code('find-median-from-data-stream', [
            (['MedianFinder', 'addNum', 'addNum', 'findMedian', 'addNum', 'findMedian'],
             [[], [1], [2], [], [3], []],
             [None, None, None, 1.5, None, 2.0]),
            (['MedianFinder', 'addNum', 'findMedian', 'addNum', 'findMedian'],
             [[], [-1], [], [-2], []],
             [None, None, -1.0, None, -1.5]),
        ]),
    },
    'implement-trie-prefix-tree': {
        'prompt': BASIC_PROMPT,
        'entry_point': "Trie",
        'test_code': _object_test_code('implement-trie-prefix-tree', [
            (['Trie', 'insert', 'search', 'search', 'startsWith', 'insert', 'search'],
             [[], ['apple'], ['apple'], ['app'], ['app'], ['app'], ['app']],
             [None, None, True, False, True, None, True]),
        ]),
    },
    'kth-largest-element-in-a-stream': {
        'prompt': BASIC_PROMPT,
        'entry_point': "KthLargest",
        'test_code': _object_test_code('kth-largest-element-in-a-stream', [
            (['KthLargest', 'add', 'add', 'add', 'add', 'add'],
             [[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]],
             [None, 4, 5, 5, 8, 8]),
            (['KthLargest', 'add', 'add'],
             [[1, []], [-3], [-2]],
             [None, -3, -2]),
        ]),
    },
    'lowest-common-ancestor-of-a-binary-search-tree': {
        'prompt': TREE_PROMPT,
        'entry_point': "Solution().lowestCommonAncestor",
        'test_code': BST_LCA_TEST_CODE,
    },
    'lru-cache': {
        'prompt': BASIC_PROMPT,
        'entry_point': "LRUCache",
        'test_code': _object_test_code('lru-cache', [
            (['LRUCache', 'put', 'put', 'get', 'put', 'get', 'put', 'get', 'get', 'get'],
             [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]],
             [None, None, None, 1, None, -1, None, -1, 3, 4]),
        ]),
    },
    'min-stack': {
        'prompt': BASIC_PROMPT,
        'entry_point': "MinStack",
        'test_code': _object_test_code('min-stack', [
            (['MinStack', 'push', 'push', 'push', 'getMin', 'pop', 'top', 'getMin'],
             [[], [-2], [0], [-3], [], [], [], []],
             [None, None, None, None, -3, None, 0, -2]),
            (['MinStack', 'push', 'push', 'getMin', 'top', 'pop', 'getMin'],
             [[], [1], [2], [], [], [], []],
             [None, None, None, 1, 2, None, 1]),
        ]),
    },
    'serialize-and-deserialize-binary-tree': {
        'prompt': TREE_PROMPT,
        'entry_point': "Codec()",
        'test_code': SERIALIZE_TREE_TEST_CODE,
    },
    'time-based-key-value-store': {
        'prompt': BASIC_PROMPT,
        'entry_point': "TimeMap",
        'test_code': _object_test_code('time-based-key-value-store', [
            (['TimeMap', 'set', 'get', 'get', 'set', 'get', 'get'],
             [[], ['foo', 'bar', 1], ['foo', 1], ['foo', 3], ['foo', 'bar2', 4], ['foo', 4], ['foo', 5]],
             [None, None, 'bar', 'bar', None, 'bar2', 'bar2']),
            (['TimeMap', 'get', 'set', 'get'],
             [[], ['missing', 1], ['love', 'high', 10], ['love', 5]],
             [None, '', None, '']),
        ]),
    },
}


# --------------------------------------------------------------------------
# Test-code generation
# --------------------------------------------------------------------------
# We keep the dataset's `test` asserts verbatim (keyword args + tree_node /
# list_node helpers that ship in `prompt`) but wrap them so the card reports a
# per-case pass/fail summary instead of dying on the first failed assert.
#
# The generated TestCode is stored as plain, editable Python in the note. The card
# (front-pro.html) reads it from a hidden <script type="text/x-python"> element, so
# arbitrary characters (backslashes, <, >, &, quotes) survive verbatim and users
# can still read / edit / add test cases directly in the TestCode field.
#
# Some problems accept MULTIPLE valid answers (the dataset inputs don't satisfy the
# "unique solution" guarantee) or return collections whose order is irrelevant. For
# those, exact `==` against the single reference output wrongly fails a correct
# solution. We rewrite their asserts to use a tolerant comparison:
#   'outer'       - top-level list order irrelevant (multiset); inner order kept
#   'outer_inner' - top-level and each nested list order irrelevant (multiset)
#   'two_sum_0' / 'two_sum_1' - validate returned indices sum to target (0/1-indexed)
COMPARE_MODES = {
    'two-sum': 'two_sum_0',
    'two-sum-ii-input-array-is-sorted': 'two_sum_1',
    'subsets': 'outer_inner',
    'subsets-ii': 'outer_inner',
    'combination-sum': 'outer_inner',
    'combination-sum-ii': 'outer_inner',
    '3sum': 'outer_inner',
    'group-anagrams': 'outer_inner',
    'permutations': 'outer',
    'palindrome-partitioning': 'outer',
    'generate-parentheses': 'outer',
    'letter-combinations-of-a-phone-number': 'outer',
    'word-search-ii': 'outer',
    'n-queens': 'outer',
    'top-k-frequent-elements': 'outer',
    'pacific-atlantic-water-flow': 'outer',
}

COMPARE_HELPERS = (
    "def _sk(o):\n"
    "    return (type(o).__name__, repr(o))\n"
    "def _canon(x, deep):\n"
    "    if isinstance(x, (list, tuple)):\n"
    "        e = [_canon(i, deep) for i in x]\n"
    "        if deep:\n"
    "            e = sorted(e, key=_sk)\n"
    "        return tuple(e)\n"
    "    return x\n"
    "def _eq_outer(a, b):\n"
    "    if a is None or b is None:\n"
    "        return a == b\n"
    "    return sorted((_canon(x, False) for x in a), key=_sk) == sorted((_canon(x, False) for x in b), key=_sk)\n"
    "def _eq_outer_inner(a, b):\n"
    "    if a is None or b is None:\n"
    "        return a == b\n"
    "    return sorted((_canon(x, True) for x in a), key=_sk) == sorted((_canon(x, True) for x in b), key=_sk)\n"
    "def _valid_two_sum(res, arr, target, one_indexed, expected):\n"
    "    if expected is None:\n"
    "        return res is None or res == [] or res == ()\n"
    "    if not res or len(res) != 2:\n"
    "        return False\n"
    "    off = 1 if one_indexed else 0\n"
    "    i, j = res[0] - off, res[1] - off\n"
    "    n = len(arr)\n"
    "    if not (0 <= i < n and 0 <= j < n and i != j):\n"
    "        return False\n"
    "    return arr[i] + arr[j] == target\n\n"
)


import copy as _copy


def _to_positional(call):
    """candidate(a=x, b=y) -> candidate(x, y) so any param naming works."""
    if call.keywords:
        call.args = list(call.args) + [k.value for k in call.keywords]
        call.keywords = []


class _PosCall(ast.NodeTransformer):
    def visit_Call(self, n):
        self.generic_visit(n)
        if isinstance(n.func, ast.Name) and n.func.id == 'candidate':
            _to_positional(n)
        return n


def _rewrite_assert(test_src, node, mode):
    """Return tolerant, positional-call source for one assert statement.

    Always converts candidate(kw=...) calls to positional so solutions with
    different parameter names (NeetCode vs LeetCode) still run. For flagged
    problems also swaps exact `==` for an order-insensitive / semantic check.
    """
    node = _copy.deepcopy(node)
    cmp = node.test
    is_eq_call = (isinstance(cmp, ast.Compare) and len(cmp.ops) == 1
                  and isinstance(cmp.ops[0], ast.Eq) and isinstance(cmp.left, ast.Call)
                  and isinstance(cmp.left.func, ast.Name) and cmp.left.func.id == 'candidate')
    if is_eq_call and mode in ('two_sum_0', 'two_sum_1'):
        kw = {k.arg: k.value for k in cmp.left.keywords}
        target_node = kw.get('target')
        arr_node = next(v for a, v in kw.items() if a != 'target')
        _to_positional(cmp.left)
        one = 'True' if mode == 'two_sum_1' else 'False'
        return (f"assert _valid_two_sum({ast.unparse(cmp.left)}, {ast.unparse(arr_node)}, "
                f"{ast.unparse(target_node)}, {one}, {ast.unparse(cmp.comparators[0])})")
    if is_eq_call and mode in ('outer', 'outer_inner'):
        _to_positional(cmp.left)
        fn = '_eq_outer' if mode == 'outer' else '_eq_outer_inner'
        return f"assert {fn}({ast.unparse(cmp.left)}, {ast.unparse(cmp.comparators[0])})"
    return ast.unparse(_PosCall().visit(node))


def build_test_code(slug, test_src):
    tree = ast.parse(test_src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check')
    mode = COMPARE_MODES.get(slug)
    lines = [_rewrite_assert(test_src, st, mode) for st in fn.body if isinstance(st, ast.Assert)]
    body = "# Test cases for %s\n" % slug
    if mode:
        body += COMPARE_HELPERS
    body += "_test_lines = [\n"
    for seg in lines:
        body += "    %r,\n" % seg
    body += "]\n\n"
    body += (
        "def check(candidate):\n"
        "    passed = 0\n"
        "    failed = []\n"
        "    ns = dict(globals())\n"
        "    ns['candidate'] = candidate\n"
        "    for i, line in enumerate(_test_lines, 1):\n"
        "        try:\n"
        "            exec(line, ns)\n"
        "            passed += 1\n"
        "        except AssertionError:\n"
        "            failed.append((i, line, 'wrong answer'))\n"
        "        except Exception as e:\n"
        "            failed.append((i, line, type(e).__name__ + ': ' + str(e)))\n"
        "    total = len(_test_lines)\n"
        "    print(f'{passed}/{total} tests passed')\n"
        "    if failed:\n"
        "        print(f'\\n{len(failed)} test(s) failed:')\n"
        "        for i, line, err in failed[:10]:\n"
        "            print(f'  Test #{i}: {err}')\n"
        "            print(f'    {line}')\n"
        "        if len(failed) > 10:\n"
        "            print(f'  ... and {len(failed) - 10} more')\n"
        "    else:\n"
        "        print('All tests passed! \\u2713')\n"
        "    return not failed\n"
    )
    return body


# --------------------------------------------------------------------------
# Load dataset -> per-slug prompt / entry_point / test_code, and dump test files
# --------------------------------------------------------------------------
def load_dataset():
    os.makedirs(TEST_CODE_DIR, exist_ok=True)
    out = {}
    with gzip.open(DATASET, "rt") as f:
        for line in f:
            it = json.loads(line)
            slug = it["task_id"]
            tc = build_test_code(slug, it["test"])
            out[slug] = {"prompt": it["prompt"], "entry_point": it["entry_point"], "test_code": tc}
    return out


def dump_test_files(lcd, slugs):
    """Write inspectable, runnable test files for the 150 deck problems."""
    for slug in slugs:
        if slug not in lcd:
            override = PROBLEM_OVERRIDES.get(slug)
            if not override:
                continue
            prompt = override["prompt"]
            test_code = override["test_code"]
            entry_point = override["entry_point"]
        else:
            prompt = lcd[slug]["prompt"].replace("from sortedcontainers import SortedList", "")
            test_code = lcd[slug]["test_code"]
            entry_point = lcd[slug]["entry_point"]
        content = (
            "# Auto-generated by build_pro_deck.py — do not edit by hand.\n"
            "# Paste your Solution class where indicated, then run to self-test.\n\n"
            + prompt + "\n\n"
            "# ==== YOUR SOLUTION HERE ====\n\n\n"
            + test_code
            + "\ncheck(%s)\n" % entry_point
        )
        with open(os.path.join(TEST_CODE_DIR, slug + ".py"), "w") as f:
            f.write(content)


# --------------------------------------------------------------------------
# LeetCode question data + NeetCode solution HTML loaders (from main.ipynb)
# --------------------------------------------------------------------------
def getLeetCodeData(title_slug, isPaid=False):
    path = f"data/paidOnly/{title_slug}.json" if isPaid else f"data/leetcode-json-data/{title_slug}.json"
    with open(path) as f:
        qdata = json.load(f)
    q = qdata["data"]["question"]
    topicTagsNew, tags = {}, []
    for t in q["topicTags"]:
        tags.append(t["slug"]); topicTagsNew[t["slug"]] = t["name"]
    hints = q["hints"]; hints_html = ""
    if hints:
        hints_html = '<div class="hints-section" style="margin-top: 20px;">\n'
        for idx, hint in enumerate(hints, 1):
            hints_html += f'''<details style="margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
        <summary style="cursor: pointer; font-weight: bold; padding: 5px;">Hint {idx}</summary>
        <p style="margin-top: 10px; padding: 10px;">{hint}</p>
    </details>
    '''
        hints_html += "</div>"
    cs = q["codeSnippets"]
    code = cs[2]["code"] if cs and len(cs) > 2 and "code" in cs[2] else ""
    if not code and title_slug in CODE_SNIPPET_OVERRIDES:
        code = CODE_SNIPPET_OVERRIDES[title_slug]
    return {
        "Id": q["questionId"], "Title": q["title"], "TitleSlug": q["titleSlug"],
        "TopicTags": json.dumps(topicTagsNew), "Difficulty": q["difficulty"],
        "Description": q["content"], "Notes": "",
        "CodeSnippets": code,
        "Hints": hints_html, "Tags": tags,
    }


def getNeetCodeSolutionHTML(title_slug):
    removeli = '<div _ngcontent-ng-c3350875783="" class="tabs is-left" style="width: 100%; overflow-x: hidden; margin-bottom: 0px;"><ul _ngcontent-ng-c3350875783="" class="tabs-list" style="width: 100%; margin-left: 0px; margin-top: 0px; margin-bottom: 0px;"><li _ngcontent-ng-c3350875783="" class="tabs-list-item my-active-tab my-active-code-tab" style="margin: 0px;"><a _ngcontent-ng-c3350875783="" role="button" tabindex="0"><span _ngcontent-ng-c3350875783="" class="tab-header font-inter bold-font" style="font-size: 16px;">Python</span></a></li><li _ngcontent-ng-c3350875783="" class="tabs-list-item" style="margin: 0px;"><a _ngcontent-ng-c3350875783="" role="button" tabindex="0"><span _ngcontent-ng-c3350875783="" class="tab-header font-inter light-text" style="font-size: 16px;">Java</span></a></li><li _ngcontent-ng-c3350875783="" class="tabs-list-item" style="margin: 0px;"><a _ngcontent-ng-c3350875783="" role="button" tabindex="0"><span _ngcontent-ng-c3350875783="" class="tab-header font-inter light-text" style="font-size: 16px;">C++</span></a></li><li _ngcontent-ng-c3350875783="" class="tabs-list-item" style="margin: 0px;"><a _ngcontent-ng-c3350875783="" role="button" tabindex="0"><span _ngcontent-ng-c3350875783="" class="tab-header font-inter light-text" style="font-size: 16px;">JavaScript</span></a></li><li _ngcontent-ng-c3350875783="" class="tabs-list-item" style="margin: 0px;"><a _ngcontent-ng-c3350875783="" role="button" tabindex="0"><span _ngcontent-ng-c3350875783="" class="tab-header font-inter light-text" style="font-size: 16px;">C#</span></a></li><li _ngcontent-ng-c3350875783="" class="tabs-list-item" style="margin: 0px;"><a _ngcontent-ng-c3350875783="" role="button" tabindex="0"><span _ngcontent-ng-c3350875783="" class="tab-header font-inter light-text" style="font-size: 16px;">Go</span></a></li><li _ngcontent-ng-c3350875783="" class="tabs-list-item" style="margin: 0px;"><a _ngcontent-ng-c3350875783="" role="button" tabindex="0"><span _ngcontent-ng-c3350875783="" class="tab-header font-inter light-text" style="font-size: 16px;">Kotlin</span></a></li><li _ngcontent-ng-c3350875783="" class="tabs-list-item" style="margin: 0px;"><a _ngcontent-ng-c3350875783="" role="button" tabindex="0"><span _ngcontent-ng-c3350875783="" class="tab-header font-inter light-text" style="font-size: 16px;">Swift</span></a></li><!-- --></ul></div>'
    removebtn = '<button class="copy-btn has-tooltip-left" data-tooltip="Copy"><fa-icon class="ng-fa-icon"><svg aria-hidden="true" class="svg-inline--fa fa-copy" data-icon="copy" data-prefix="fas" focusable="false" role="img" viewBox="0 0 448 512" xmlns="http://www.w3.org/2000/svg"><path d="M208 0L332.1 0c12.7 0 24.9 5.1 33.9 14.1l67.9 67.9c9 9 14.1 21.2 14.1 33.9L448 336c0 26.5-21.5 48-48 48l-192 0c-26.5 0-48-21.5-48-48l0-288c0-26.5 21.5-48 48-48zM48 128l80 0 0 64-64 0 0 256 192 0 0-32 64 0 0 48c0 26.5-21.5 48-48 48L48 512c-26.5 0-48-21.5-48-48L0 176c0-26.5 21.5-48 48-48z" fill="currentColor"></path></svg></fa-icon></button>'
    removebtn1 = removebtn.replace("viewBox", "viewbox")
    with open(f"data/neetcode-solution-html/{title_slug}.html") as f:
        html = f.read()
    html = html.replace(removeli, "").replace(removebtn, "").replace(removebtn1, "")
    html = html.replace("<!-- -->", "")
    html = html.replace('<h1 _ngcontent-ng-c3055955716="" style="font-size: 26px; margin-top: 24px; margin-bottom: 20px;">Prerequisites</h1>', "")
    html = re.sub(r"<app-prereq-cards[^>]*>.*?</app-prereq-cards>", "", html, flags=re.DOTALL)
    return html


# --------------------------------------------------------------------------
# Reuse stable model_id + deck_ids from the existing apkg so re-imports update
# instead of creating duplicate decks.
# --------------------------------------------------------------------------
def existing_ids():
    if not os.path.exists(APKG_OUT):
        return None, {}
    z = zipfile.ZipFile(APKG_OUT)
    tmp = tempfile.mkdtemp(); z.extract("collection.anki2", tmp)
    db = sqlite3.connect(os.path.join(tmp, "collection.anki2"))
    models, decks = db.execute("select models, decks from col").fetchone()
    models, decks = json.loads(models), json.loads(decks)
    mid = int(next(iter(models)))
    deck_by_name = {d["name"]: int(did) for did, d in decks.items()}
    db.close(); shutil.rmtree(tmp)
    return mid, deck_by_name


def main():
    lcd = load_dataset()
    data = json.load(open(NEETCODE_LIST))
    slugs = [urlparse(data[d][q]["url"]).path.strip("/").split("/")[-1] for d in data for q in data[d]]
    dump_test_files(lcd, slugs)

    model_id, deck_by_name = existing_ids()
    if model_id is None:
        model_id = 1189842311  # keep stable across fresh builds

    with open("card-template/front-pro.html") as f: front = f.read()
    with open("card-template/back.html") as f: back = f.read()
    with open("card-template/card.css") as f: css = f.read()

    model = genanki.Model(
        model_id=model_id, name="Basic - Anki-NeetCode - Pro",
        fields=[{"name": n} for n in ["Id", "Title", "TitleSlug", "TopicTags", "Difficulty",
                "Description", "Notes", "CodeSnippets", "Hints", "Solution", "EntryPoint", "TestCode", "Prompt"]],
        templates=[{"name": "Card 1", "qfmt": front, "afmt": back}], css=css)

    class SlugNote(genanki.Note):
        @property
        def guid(self):
            return genanki.guid_for(self.fields[2])  # stable on title slug

    decks = []
    missing = []
    for index, d in enumerate(data):
        index_str = f"0{index+1}" if index < 9 else str(index + 1)
        deck_name = f"Anki - NeetCode Pro::{index_str}. {d}"
        deck_id = deck_by_name.get(deck_name)
        if deck_id is None:
            deck_id = int(genanki.guid_for(deck_name).encode().hex()[:8], 16) | (1 << 30)
        sub = genanki.Deck(deck_id, deck_name)
        for q in data[d]:
            slug = urlparse(data[d][q]["url"]).path.strip("/").split("/")[-1]
            lc = getLeetCodeData(slug, isPaid=slug in paidOnly)
            sol = getNeetCodeSolutionHTML(slug)
            if slug not in lcd:
                override = PROBLEM_OVERRIDES.get(slug)
                if override:
                    prompt = override["prompt"]
                    entry = override["entry_point"]
                    test_code = override["test_code"]
                else:
                    missing.append(slug); prompt = entry = test_code = ""
            else:
                prompt = lcd[slug]["prompt"].replace("from sortedcontainers import SortedList", "")
                entry = lcd[slug]["entry_point"]
                test_code = lcd[slug]["test_code"]
            fields = [str(lc["Id"]), lc["Title"], lc["TitleSlug"], lc["TopicTags"], lc["Difficulty"],
                      lc["Description"], lc["Notes"], lc["CodeSnippets"], lc["Hints"], sol,
                      entry, test_code, prompt]
            sub.add_note(SlugNote(model=model, fields=fields, tags=lc["Tags"]))
        decks.append(sub)

    media_files = [
        "card-template/css/_atom-one-dark.min.css", "card-template/css/_codemirror.css",
        "card-template/css/_katex.css", "card-template/css/_nord.css",
        "card-template/js/_codemirror.js", "card-template/js/_katex.min.js",
        "card-template/js/_highlight.min.js", "card-template/js/_pyodide.js",
        "card-template/js/_python-codemirror.js", "card-template/js/_python.min.js",
        "card-template/fonts/_Material_Symbols_Outlined.woff2",
    ]
    genanki.Package(decks, media_files=media_files).write_to_file(APKG_OUT)
    print(f"WROTE {APKG_OUT}  model_id={model_id}  decks={len(decks)}")
    print(f"test files -> {TEST_CODE_DIR}/  ({len(slugs) - len(missing)} written)")
    print(f"cards without test coverage: {len(missing)} -> {missing}")


if __name__ == "__main__":
    main()
