"""Ground only explicitly observable cyber facts."""
import json
from ..pddl.model import Atom
from ..store import Delta

class CyberBattleAdapter:
    def ground(self, observation, store, first=False):
        data = json.loads(observation) if observation.strip() else {}
        def atoms(key):
            return [Atom(item['pred'], tuple(item.get('args', []))) for item in data.get(key, [])]
        return Delta(add=atoms('facts'), delete=atoms('delete'), types=dict(data.get('types', {})))
