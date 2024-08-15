# Python3 Template

from typing import *
from bisect import bisect_left, bisect_right
from heapq import heappop, heappush
from math import inf, log2
import sys

"""Binary Trie"""
class BinaryTrie:
    class Node:
        def __init__(self, bit: bool = False):
            self.bit = bit # Stores the current bit (False if 0, True if 1)
            self.children = []
            self.count = 0 # stores number of keys finishing at this bit
            self.counter = 1 # stores number of keys with this bit as prefix

    def __init__(self, size):
        self.root = BinaryTrie.Node()
        self.size = size # Maximum size of each key

    def convert(self, key):
        """Converts key from string/integer to a list of boolean values!"""
        bits = []
        if isinstance(key, int):
            key = bin(key)[2:]
        if isinstance(key, str):
            for i in range(self.size - len(key)):
                bits += [False]
            for i in key:
                if i == "0":
                    bits += [False]
                else:
                    bits += [True]
        else:
            return list(key)
        return bits

    def add(self, key):
        """Add a key to the trie!"""
        node = self.root
        bits = self.convert(key)
        for bit in bits:
            found_in_child = False
            for child in node.children:
                if child.bit == bit:
                    child.counter += 1
                    node = child
                    found_in_child = True
                    break
            if not found_in_child:
                new_node = BinaryTrie.Node(bit)
                node.children.append(new_node)
                node = new_node
        node.count += 1

    def remove(self, key):
        """Removes a key from the trie! If there are multiple occurrences, it removes only one of them."""
        node = self.root
        bits = self.convert(key)
        nodelist = [node]
        for bit in bits:
            for child in node.children:
                if child.bit == bit:
                    node = child
                    node.counter -= 1
                    nodelist.append(node)
                    break
        node.count -= 1
        if not node.children and not node.count:
            for i in range(len(nodelist) - 2, -1, -1):
                nodelist[i].children.remove(nodelist[i + 1])
                if nodelist[i].children or nodelist[i].count:
                    break

    def query(self, prefix, root=None):
        """Search for a prefix in the trie! Returns the node if found, otherwise 0."""
        if not root: root = self.root
        node = root
        if not root.children:
            return 0
        for bit in prefix:
            bit_not_found = True
            for child in node.children:
                if child.bit == bit:
                    bit_not_found = False
                    node = child
                    break
            if bit_not_found:
                return 0
        return node

"""Trie"""
class Trie:
    class Node:
        def __init__(self, char: str = "*"):
            self.char = char
            self.children = []
            self.count = 0 # stores count of words finishing at this point
            self.counter = 1 # stores count of words which have this prefix

    def __init__(self):
        self.root = Trie.Node()

    def add(self, word: str):
        """Add a word to the trie!"""
        node = self.root
        for char in word:
            found_in_child = False
            for child in node.children:
                if child.char == char:
                    child.counter += 1
                    node = child
                    found_in_child = True
                    break
            if not found_in_child:
                new_node = Trie.Node(char)
                node.children.append(new_node)
                node = new_node
        node.count += 1

    def remove(self, word: str):
        """Removes a word from the trie! If there are multiple occurrences, it removes only one of them."""
        node = self.root
        nodelist = [node]
        for char in word:
            for child in node.children:
                if child.char == char:
                    node = child
                    node.counter -= 1
                    nodelist.append(node)
                    break
        node.count -= 1
        if not node.children and not node.count:
            for i in range(len(nodelist) - 2, -1, -1):
                nodelist[i].children.remove(nodelist[i + 1])
                if nodelist[i].children or nodelist[i].count:
                    break

    def query(self, prefix, root=None):
        """Search for a prefix in the trie! Returns the node if found, otherwise 0."""
        if not root: root = self.root
        node = root
        if not root.children:
            return 0
        for char in prefix:
            char_not_found = True
            for child in node.children:
                if child.char == char:
                    char_not_found = False
                    node = child
                    break
            if char_not_found:
                return 0
        return node

"""Segment Tree"""
class LazySegmentTree:
    def __init__(self, array, func=max):
        self.n = len(array)
        self.size = 2**(int(log2(self.n-1))+1) if self.n != 1 else 1
        self.func = func
        self.default = 0 if self.func != min else inf
        self.data = [self.default] * (2 * self.size)
        self.lazy = [0] * (2 * self.size)
        self.process(array)

    def process(self, array):
        self.data[self.size : self.size+self.n] = array
        for i in range(self.size-1, -1, -1):
            self.data[i] = self.func(self.data[2*i], self.data[2*i+1])

    def push(self, index):
        """Push the information of the root to its children!"""
        self.lazy[2*index] += self.lazy[index]
        self.lazy[2*index+1] += self.lazy[index]
        self.data[2 * index] += self.lazy[index]
        self.data[2 * index + 1] += self.lazy[index]
        self.lazy[index] = 0

    def build(self, index):
        """Build data with the new changes!"""
        index >>= 1
        while index:
            self.data[index] = self.func(self.data[2*index], self.data[2*index+1]) + self.lazy[index]
            index >>= 1

    def query(self, alpha, omega):
        """Returns the result of function over the range (inclusive)!"""
        res = self.default
        alpha += self.size
        omega += self.size + 1
        for i in range(len(bin(alpha)[2:])-1, 0, -1):
            self.push(alpha >> i)
        for i in range(len(bin(omega-1)[2:])-1, 0, -1):
            self.push((omega-1) >> i)
        while alpha < omega:
            if alpha & 1:
                res = self.func(res, self.data[alpha])
                alpha += 1
            if omega & 1:
                omega -= 1
                res = self.func(res, self.data[omega])
            alpha >>= 1
            omega >>= 1
        return res

    def update(self, alpha, omega, value):
        """Increases all elements in the range (inclusive) by given value!"""
        alpha += self.size
        omega += self.size + 1
        l, r = alpha, omega
        while alpha < omega:
            if alpha & 1:
                self.data[alpha] += value
                self.lazy[alpha] += value
                alpha += 1
            if omega & 1:
                omega -= 1
                self.data[omega] += value
                self.lazy[omega] += value
            alpha >>= 1
            omega >>= 1
        self.build(l)
        self.build(r-1)

class SegmentTree:
    def __init__(self, array):
        self.n = len(array)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.func = lambda x, y: x if x[0] < y[0] else y
        self.default = (10**9, -1)
        self.data = [self.default] * (2 * self.size)
        self.process(array)

    def process(self, array):
        self.data[self.size : self.size+self.n] = array
        for i in range(self.size-1, -1, -1):
            self.data[i] = self.func(self.data[2*i], self.data[2*i+1])

    def query(self, alpha, omega):
        """Returns the result of function over the range (inclusive)!"""
        if alpha == omega:
            return self.data[alpha + self.size]
        res = self.default
        alpha += self.size
        omega += self.size + 1
        while alpha < omega:
            if alpha & 1:
                res = self.func(res, self.data[alpha])
                alpha += 1
            if omega & 1:
                omega -= 1
                res = self.func(res, self.data[omega])
            alpha >>= 1
            omega >>= 1
        return res

    def update(self, index, value):
        """Updates the element at index to given value!"""
        index += self.size
        self.data[index] = value
        index >>= 1
        while index:
            self.data[index] = self.func(self.data[2*index], self.data[2*index+1])
            index >>= 1

"""LCA"""
class LCA:
    def __init__(self, graph, root):
        self.graph = graph
        self.n = len(graph)
        self.euler = []
        self.first = [-1]*self.n
        self.st = None
        self.process(root)

    def process(self, root):
        visited, parents, heights = [False]*self.n, [-1]*self.n, [1]*self.n
        stack = [root]
        while stack:
            v = stack[-1]
            if not visited[v]:
                visited[v] = True
                self.euler += [v]
                if self.first[v] == -1:
                    self.first[v] = len(self.euler) - 1
                for u in self.graph[v]:
                    if not visited[u]:
                        stack.append(u)
                        parents[u], heights[u] = v, heights[v] + 1
            else:
                self.euler += [parents[stack.pop()]]
        self.euler = [(heights[k], k) for k in self.euler]
        self.st = SegmentTree(self.euler)

    def query(self, x, y):
        """Returns the lowest common ancestor of nodes x and y!"""
        p, q = min(self.first[x], self.first[y]), max(self.first[x], self.first[y])
        return self.st.query(p, q)[1]

"""Graph"""
def dijkstra(graph, alpha):
    """Calculates Shortest Distance from a source node to all nodes"""
    n = len(graph)
    distance = [float("inf")]*n
    distance[alpha] = 0
    parents = [-1]*n
    queue = []
    heappush(queue, (0, alpha))
    while queue:
        length, v = heappop(queue)
        for x, dist in graph[v]:
            if dist + length < distance[x]:
                distance[x] = dist + length
                parents[x] = v
                heappush(queue, (dist+length, x))
    return distance, parents

"""DSU"""
class DisjointSetUnion:
    def __init__(self, n):
        self.parent = [*range(n+1)]
        self.size = [1]*(n+1)
        self.min, self.max = [*range(n+1)], [*range(n+1)]
        self.count = n

    def get(self, a):
        """Returns the identifier (parent) of the set to which a belongs to!"""
        if self.parent[a] == a:
            return a
        x = a
        while a != self.parent[a]:
            a = self.parent[a]
        while x != self.parent[x]:
            self.parent[x], x = a, self.parent[x]
        return a

    def union(self, a, b):
        """Join two sets that contain a and b!"""
        a, b = self.get(a), self.get(b)
        if a != b:
            if self.size[a] > self.size[b]:
                a, b = b, a
            self.parent[a] = b
            self.size[b] += self.size[a]
            self.min[b] = min(self.min[a], self.min[b])
            self.max[b] = max(self.max[a], self.max[b])
            self.count -= 1

    def count_sets(self):
        """Returns the number of disjoint sets!"""
        return self.count

class Result:
    def __init__(self, index, value):
        self.index = index
        self.value = value

"""Binary Search"""
class BinarySearch:
    def __init__(self):
        pass

    @staticmethod
    def greater_than(num: int, func, size: int = 1):
        """Searches for smallest element greater than num!"""
        if isinstance(func, list):
            index = bisect_right(func, num)
            if index == len(func):
                return Result(None, None)
            else:
                return Result(index, func[index])
        else:
            alpha, omega = 0, size - 1
            if func(omega) <= num:
                return Result(None, None)
            while alpha < omega:
                if func(alpha) > num:
                    return Result(alpha, func(alpha))
                if omega == alpha + 1:
                    return Result(omega, func(omega))
                mid = (alpha + omega) // 2
                if func(mid) > num:
                    omega = mid
                else:
                    alpha = mid

    @staticmethod
    def less_than(num: int, func, size: int = 1):
        """Searches for largest element less than num!"""
        if isinstance(func, list):
            index = bisect_left(func, num) - 1
            if index == -1:
                return Result(None, None)
            else:
                return Result(index, func[index])
        else:
            alpha, omega = 0, size - 1
            if func(alpha) >= num:
                return Result(None, None)
            while alpha < omega:
                if func(omega) < num:
                    return Result(omega, func(omega))
                if omega == alpha + 1:
                    return Result(alpha, func(alpha))
                mid = (alpha + omega) // 2
                if func(mid) < num:
                    alpha = mid
                else:
                    omega = mid

"""Matrix"""
class Matrix(object):
    def __init__(self, n, m, values, modulo=10**9+7):
        """Initialize a nxm matrix with given set of values!"""
        self.n = n
        self.m = m
        self.modulo = modulo
        if len(values) == n*m:
            self.values = values
        else:
            self.values = [i for k in values for i in k]

    def get_element(self, i, j):
        """Returns element A[i][j]"""
        return self.values[i*self.n + j]

    def __add__(self, other):
        """Add two matrices if they have the same order!"""
        if other.n != self.n or other.m != self.m:
            return None
        return Matrix(self.n, self.m,
                      [(self.values[i] + other.values[i])%self.modulo
                       for i in range(self.n*self.m)],
                      self.modulo)

    def __sub__(self, other):
        """Subtract two matrices if they have the same order!"""
        if other.n != self.n or other.m != self.m:
            return None
        return Matrix(self.n, self.m,
                      [(self.values[i] - other.values[i]) % self.modulo
                       for i in range(self.n*self.m)],
                      self.modulo)

    def __mul__(self, other):
        """Multiply two matrices if multiplication is possible!"""
        if self.m != other.n:
            return None
        result = [0]*(self.n * other.m)
        for i in range(self.n):
            for j in range(self.m):
                for k in range(other.m):
                    result[i*self.n + k] = \
                        (result[i*self.n + k] +
                         self.values[i*self.n + j] * other.values[j*other.n + k]) % self.modulo
        return Matrix(self.n, other.m, result, self.modulo)

    def __pow__(self, power):
        """Raises a matrix to some power!"""
        result = None
        cur = self
        b = bin(power)[2:]
        for i in range(len(b) - 1, -1, -1):
            if b[i] == '1':
                result = cur if result is None else result*cur
            cur *= cur
        return result

    def __repr__(self):
        return "Matrix({0})".format([[self.values[i*self.n+j] for j in range(self.m)]
                                     for i in range(self.n)])

"""Gray Code"""
def gray_code(n):
    """Finds Gray Code of n!"""
    return n ^ (n >> 1)

def reverse_gray_code(g):
    """Restores number n from the gray code!"""
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n

def get_sequence(k):
    """Gets sequence of gray codes for k-bit numbers!"""
    result = []
    for i in range(2**k):
        next = bin(gray_code(i))[2:]
        result += ["0"*(k - len(next)) + next]
    return result

"""Linear Recurrence"""
class LinearRecurrence:
    def __init__(self, coefficients, values, mod=10 ** 9 + 7):
        self.coefficients = coefficients
        self.values = values
        self.mod = mod
        self.k = len(values)
        self.a = []
        self.b = []
        self.process()

    def process(self):
        self.a = [self.values]
        self.b = [[0] * self.k for _ in range(self.k)]
        i = 1
        while i < self.k:
            self.b[i][i - 1] = 1
            i += 1
        for j in range(self.k):
            self.b[j][self.k - 1] = self.coefficients[-j - 1]

    def multiply(self, A, B):
        return [[sum(i * j for i, j in zip(row, col)) % self.mod for col in zip(*B)] for row in A]

    def get_power(self, p):
        res = [[0] * self.k for _ in range(self.k)]
        cur = [list(x) for x in self.b]
        for i in range(self.k):
            res[i][i] = 1
        i = 0
        while p:
            if p & (1 << i):
                res = list(self.multiply(res, cur))
                p ^= (1 << i)
            cur = list(self.multiply(cur, cur))
            i += 1
        return res

    def get_term(self, n):
        if n <= self.k:
            return self.values[n - 1] % self.mod
        return self.multiply(self.a, self.get_power(n - self.k))[0][-1]

"""LinkedList """
class LinkedList:
    class ListNode:
        def __init__(self, val=0, nextElement=None):
            self.val = val
            self.next = nextElement

    @staticmethod
    def generateLinkedListFromArray(self, arr=None) -> ListNode | None:
        if arr is None:
            arr = [Type[int]]
        if not len(arr):
            return None
        head = LinkedList.ListNode(arr[0])
        current = head
        for i in range(1,len(arr)):
            current.next = LinkedList.ListNode(arr[i])
            current = current.next
        return head

    @staticmethod
    def generateLinkedList(self,length) -> ListNode:
        """Generates LinkedList of specified Length"""
        l1 = LinkedList.ListNode()
        h1 = l1
        while length:
            h1.next = LinkedList.ListNode(length)
            h1 = h1.next
            length -= 1
        return l1.next

    @staticmethod
    def printLinkedList(self,ll: List[ListNode]) -> None:
        """Prints LinkedList in a single Line"""
        p = ll
        while p:
            print(p.val, end=" ")
            p = p.next

def SieveOfEratosthenes(limit=10**6):
    """Returns all primes not greater than limit"""
    isPrime = [True]*(limit+1)
    isPrime[0] = isPrime[1] = False
    primes = []
    for i in range(2, limit+1):
        if not isPrime[i]:
            continue
        primes += [i]
        for j in range(i*i, limit+1, i):
            isPrime[j] = False
    return primes

"""Fast Input Functions"""
Input = sys.stdin.readline

def inputInt():
    return int(Input())
def inputList(string=False):
    if string:
        return list(map(str,Input().split()))
    return list(map(int,Input().split()))
def inputString():
    string = Input()
    return list(string[:len(string)-1])


