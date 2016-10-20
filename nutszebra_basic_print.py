import inspect
import tqdm


# store builtin print
old_print = print


def new_print(*args, **kwargs):
    # if tqdm.tqdm.write raises error, use builtin print
    try:
        tqdm.tqdm.write(*args, **kwargs)
    except:
        old_print(*args, ** kwargs)

# globaly replace print with new_print
inspect.builtins.print = new_print
