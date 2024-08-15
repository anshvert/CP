class SegmentTree:
    def __init__(self, arr):
        self.tree = [0]*(4*len(arr))
        self.n = len(arr)
        self.arr = arr

    def build(self, v=1, lt=0, rt=None):
        if rt is None:
            rt = self.n - 1
        if lt == rt:
            self.tree[v] = self.arr[lt]
        else:
            mid = (lt + rt)//2
            self.build(v*2, lt, mid)
            self.build(v*2 + 1, mid+1, rt)
            self.tree[v] = self.tree[v*2] + self.tree[v*2 + 1]

    def sum(self,l, r, v=1, lt=0, rt=None):
        if rt is None:
            rt = self.n - 1
        if l > r: return 0
        if lt == l and rt == r:
            return self.tree[v]
        mid = (lt + rt) // 2
        lSum = self.sum(l,min(r,mid),v*2, lt, mid)
        rSum = self.sum(max(mid+1,l),r,v*2 + 1, mid+1, rt)
        print(lSum,rSum)
        return lSum + rSum

    def update(self, ind, val, v=1, lt=0, rt=None):
        if rt is None:
            rt = self.n - 1
        if lt == rt:
            self.tree[v] = val
        else:
            mid = (lt + rt) // 2
            if ind <= mid:
                self.update(ind,val,v*2,lt,mid)
            else:
                self.update(ind,val,v*2 + 1,mid+1,rt)
            self.tree[v] = self.tree[v*2] + self.tree[v*2 + 1]

seg = SegmentTree([1,2,3,4,5])
seg.build()
print(seg.tree)
print(seg.sum(2,4))
seg.update(0,5)
print(seg.sum(0,2))



