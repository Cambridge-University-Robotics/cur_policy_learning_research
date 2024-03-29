from . import passive_hand
import inspect
import collections

_DOMAINS = {name: module for name, module in locals().items()
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}


def _get_tasks(tag):
    """Returns a sequence of (domain name, task name) pairs for the given tag."""
    result = []

    for domain_name in sorted(_DOMAINS.keys()):

        domain = _DOMAINS[domain_name]

        if tag is None:
            tasks_in_domain = domain.SUITE
        else:
            tasks_in_domain = domain.SUITE.tagged(tag)

        for task_name in tasks_in_domain.keys():
            result.append((domain_name, task_name))

    return tuple(result)


def _get_tasks_by_domain(tasks):
    """Returns a dict mapping from task name to a tuple of domain names."""
    result = collections.defaultdict(list)

    for domain_name, task_name in tasks:
        result[domain_name].append(task_name)

    return {k: tuple(v) for k, v in result.items()}


# A sequence containing all (domain name, task name) pairs.
ALL_TASKS = _get_tasks(tag=None)

# Subsets of ALL_TASKS, generated via the tag mechanism.
BENCHMARKING = _get_tasks('benchmarking')
EASY = _get_tasks('easy')
HARD = _get_tasks('hard')
EXTRA = tuple(sorted(set(ALL_TASKS) - set(BENCHMARKING)))
NO_REWARD_VIZ = _get_tasks('no_reward_visualization')
REWARD_VIZ = tuple(sorted(set(ALL_TASKS) - set(NO_REWARD_VIZ)))

# A mapping from each domain name to a sequence of its task names.
TASKS_BY_DOMAIN = _get_tasks_by_domain(ALL_TASKS)


def load(domain_name, task_name, task_kwargs=None, environment_kwargs=None):
    """Returns an environment from a domain name, task name and optional settings.
    ```python
    env = suite.load('cartpole', 'balance')
    ```
    Args:
      domain_name: A string containing the name of a domain.
      task_name: A string containing the name of a task.
      task_kwargs: Optional `dict` of keyword arguments for the task.
      environment_kwargs: Optional `dict` specifying keyword arguments for the
        environment.
    Returns:
      The requested environment.
    """
    return build_environment(domain_name, task_name, task_kwargs,
                             environment_kwargs)


def build_environment(domain_name, task_name, task_kwargs=None,
                      environment_kwargs=None, visualize_reward=False):
    """Returns an environment from the suite given a domain name and a task name.
    Args:
      domain_name: A string containing the name of a domain.
      task_name: A string containing the name of a task.
      task_kwargs: Optional `dict` specifying keyword arguments for the task.
      environment_kwargs: Optional `dict` specifying keyword arguments for the
        environment.
    Raises:
      ValueError: If the domain or task doesn't exist.
    Returns:
      An instance of the requested environment.
    """
    if domain_name not in _DOMAINS:
        raise ValueError('Domain {!r} does not exist.'.format(domain_name))

    domain = _DOMAINS[domain_name]
    if task_name not in domain.SUITE:
        raise ValueError('Level {!r} does not exist in domain {!r}.'.format(
            task_name, domain_name))

    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = dict(task_kwargs, environment_kwargs=environment_kwargs)
    env = domain.SUITE[task_name](**task_kwargs)
    return env
