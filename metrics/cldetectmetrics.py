from avalanche.evaluation import Metric

class DetectionForgetting(Metric[float]):
    def __init__(self):
        self.initial_map = {}
        self.last_map = {}

    def update(self, strategy):
        exp_id = strategy.experience.current_experience
        coco_metrics = strategy.evaluation_plugin.last_metrics
        current_map = coco_metrics.get("DetectionMetrics/bbox/AP", 0)
        
        if exp_id not in self.initial_map:
            self.initial_map[exp_id] = current_map
        self.last_map[exp_id] = current_map

    def result(self):
        return sum(
            self.initial_map[eid] - self.last_map.get(eid, 0) 
            for eid in self.initial_map
        ) / max(1, len(self.initial_map))

    def reset(self):
        self.initial_map.clear()
        self.last_map.clear()
