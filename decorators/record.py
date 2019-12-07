from gym.wrappers import Monitor


def record(env, enabled):
    def wrapper_record(func):
        if enabled:
            monitor = Monitor(env, './videos', force=True)
            func()
            monitor.close()
        else:
            func()
    return wrapper_record
