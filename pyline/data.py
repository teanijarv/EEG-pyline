class PylineData:
    def __init__(self, raw_data, filename, tasktype = None):
        self.raw = raw_data if raw_data else None
        self.filtered = None
        self.resampled = None
        self.ica = None
        self.csd = None
        self.pyline = None
        self.events = None
        self.filename = filename if filename else None
        if tasktype is not None and tasktype not in ['task', 'resting']:
            raise ValueError('tasktype must be either "task" or "resting"')
        self.tasktype = tasktype