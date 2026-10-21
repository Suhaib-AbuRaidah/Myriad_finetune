"""Scene shuffle followed by every joint in stable ordinal order."""
import hashlib
import json
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inventory(dataset):
    rows = []
    for i, (scene, pair) in enumerate(dataset.examples):
        path = dataset.files[scene]
        st = path.stat()
        rows.append(dict(index=i, scene=scene, path=str(path.resolve()), size=st.st_size,
                         mtime_ns=st.st_mtime_ns, parent=pair[0], child=pair[1], joint_index=pair[2]))
    return dict(version=1, pairs=rows, sha256=digest(rows))


class PairCursor:
    def __init__(self, examples, seed, *, epoch=0, offset=0):
        self.examples = [(int(s), tuple(p)) for s, p in examples]
        self.seed, self.epoch, self.offset = seed, epoch, offset
        self.signature = digest(self.examples)
        if not self.examples or epoch < 0 or not 0 <= offset <= len(examples):
            raise ValueError("Invalid pair cursor")

    def order(self):
        groups = {}
        for i, (scene, _) in enumerate(self.examples):
            groups.setdefault(scene, []).append(i)
        scenes = sorted(groups)
        order = torch.randperm(len(scenes), generator=torch.Generator().manual_seed(self.seed+self.epoch))
        return [i for s in order.tolist() for i in groups[scenes[s]]]

    def next_indices(self, batch_size):
        if batch_size < 1:
            raise ValueError("Positive batch size required")
        if self.offset == len(self.examples):
            self.epoch += 1
            self.offset = 0
        stop = min(len(self.examples), self.offset+batch_size)
        result = self.order()[self.offset:stop]
        self.offset = stop
        return result

    def state_dict(self):
        return dict(version=1, epoch=self.epoch, offset=self.offset, seed=self.seed,
                    signature=self.signature, size=len(self.examples))

    @classmethod
    def from_state(cls, state, examples, seed):
        obj = cls(examples, seed, epoch=state["epoch"], offset=state["offset"])
        if state != obj.state_dict():
            raise ValueError("Pair inventory, traversal policy or seed changed")
        return obj
