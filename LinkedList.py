from typing import *

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Generates LinkedList of specified Length
def generateLinkedList(length) -> ListNode:
    l1 = ListNode()
    h1 = l1
    while length:
        h1.next = ListNode(length)
        h1 = h1.next
        length -= 1
    return l1.next

# Prints LinkedList in a single Line
def printLinkedList(ll):
    p = ll
    while p:
        print(p.val,end=" ")
        p = p.next


