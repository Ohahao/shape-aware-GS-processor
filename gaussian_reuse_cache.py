class GaussianCache:
    def __init__(self, num_sets=1024, ways=8):
        self.num_sets = num_sets
        self.ways = ways
        # Cache storage: each set holds up to 8 Gaussian IDs
        self.set_entries = [ [None]*ways for _ in range(num_sets) ]
        # P-LRU tree bits for each set (for 8-way cache, 7 bits per set)
        self.plru_bits = [ [0]*(ways-1) for _ in range(num_sets) ]
        # Track number of valid entries in each set
        self.set_count = [0]*num_sets

    def _get_victim_way(self, set_idx):
        """Traverse the PLRU tree to find the least-recently-used way index."""
        bits = self.plru_bits[set_idx]
        # Tree traversal (0 = go left, 1 = go right)
        way = 0
        # Level 0 (1 bit: index 0 covers ways [0-7])
        direction = bits[0]   # 0 -> left(ways0-3), 1 -> right(ways4-7)
        way |= (direction << 2)           # set MSB of way (bit 2) based on root
        node_index = 1 + direction       # go to node1 (left) or node2 (right)
        # Level 1 (2 bits: node_index 1 covers ways0-3, index2 covers ways4-7)
        direction = bits[node_index]     # 0-> left half, 1-> right half of that node
        way |= (direction << 1)          # set next bit of way index
        node_index = 3 + (node_index-1)*2 + direction  # compute child node index
        # Level 2 (4 bits: node_index 3..6 each cover 2 ways)
        direction = bits[node_index]     # 0->left of that pair, 1->right
        way |= direction                 # set least significant bit of way
        return way  # integer 0-7

    def _update_plru(self, set_idx, way_used):
        """Update PLRU bits along the path of the accessed way."""
        bits = self.plru_bits[set_idx]
        # Determine binary path for this way (3-bit for 8 ways)
        b2 = 0 if way_used < 4 else 1       # MSB: which half (0=ways0-3, 1=ways4-7)
        b1 = 0 if way_used % 4 < 2 else 1   # next bit: within half (0=lower half, 1=upper half)
        b0 = 0 if way_used % 2 == 0 else 1  # LSB: even or odd index within that pair
        # Update bits: set each node bit to indicate the opposite branch (LRU) from the one taken
        bits[0] = 1 - b2               # root: if used left (b2=0) set bit=1, else 0
        bits[1 + b2] = 1 - b1          # node1 or node2: update the one corresponding to this half
        # Determine which node at level2 (index 3-6) corresponds to the half:
        node_idx = 3 if b2==0 and b1==0 else \
                   4 if b2==0 and b1==1 else \
                   5 if b2==1 and b1==0 else 6
        bits[node_idx] = 1 - b0        # update that node's bit (which pair)
        # (Note: P-LRU uses a binary tree where each bit points to the LRU subtree. 
        # Here we set the bit to the opposite of the branch taken, marking the other branch as LRU.)
    
    def access(self, gaussian_id):
        """Access a Gaussian in the cache. Return True if hit, False if miss (requiring memory fetch)."""
        set_idx = gaussian_id % self.num_sets
        # Check if gaussian is already in cache (cache hit)
        if gaussian_id in self.set_entries[set_idx]:
            way = self.set_entries[set_idx].index(gaussian_id)
            self._update_plru(set_idx, way)   # update PLRU on hit
            return True
        # Cache miss: need to load Gaussian from memory
        if self.set_count[set_idx] < self.ways:
            # Empty slot available – fill it
            way = self.set_entries[set_idx].index(None)
            self.set_entries[set_idx][way] = gaussian_id
            self.set_count[set_idx] += 1
        else:
            # Evict the P-LRU victim
            way = self._get_victim_way(set_idx)
            self.set_entries[set_idx][way] = gaussian_id
        self._update_plru(set_idx, way)
        return False
