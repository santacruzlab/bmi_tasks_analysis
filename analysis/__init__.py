

def get(id, task_specific=True, **kwargs):
	if task_specific:
		import performance
		return performance._get_te(id, **kwargs)
	else:
		from db import dbfunctions
		return dbfunctions.TaskEntry(id)